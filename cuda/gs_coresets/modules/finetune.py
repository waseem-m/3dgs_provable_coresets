#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import gc
import os
import sys
from random import randint

import torch
import numpy as np  # NEW: for moving average
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR

# NOTE: import matplotlib (backend decision later); DO NOT import pyplot here
import matplotlib

# Ensure <PROJECT_DIR> and <GraphDECO_DIR> are both on project path before imports
from gs_coresets.utils.default_paths import ensure_all_on_syspath
ensure_all_on_syspath()

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr  # noqa: F401 (used inside logger_utils.training_report)
from arguments import ModelParams, PipelineParams, OptimizationParams
from gs_coresets.utils.io_utils import ensure_dir_for
from gs_coresets.utils.logger_utils import training_report, prepare_output_and_logger


try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# =========================
# Plot helpers (interactive detection + smoothing)
# =========================

# You can tweak this to control smoothing strength (must be >= 1)
SMOOTH_WINDOW = 20  # trailing moving average window in iterations

def _safe_isatty(stream) -> bool:
    try:
        isatty = getattr(stream, "isatty", None)
        if callable(isatty):
            return bool(isatty())
        fileno = getattr(stream, "fileno", None)
        if callable(fileno):
            return bool(os.isatty(fileno()))
    except Exception:
        pass
    return False

def _is_interactive_session() -> bool:
    """Return True for Jupyter/IPython or a real TTY with a GUI display."""
    # IPython/Jupyter?
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    # Plain terminal with a TTY and display
    if _safe_isatty(sys.stdin) and _safe_isatty(sys.stdout):
        if sys.platform == "win32":
            return True
        return "DISPLAY" in os.environ and os.environ["DISPLAY"] != ""
    return False

def _moving_average_trailing(values: Sequence[float], window: int) -> Tuple[np.ndarray, int]:
    """
    Trailing moving average of 'values' with window 'window' (>=1).
    Returns (smoothed_values, used_window). Length is len(values) - window + 1.
    """
    n = len(values)
    if n == 0:
        return np.asarray([]), 0
    w = max(1, min(window, n))
    if w == 1:
        return np.asarray(values, dtype=float), 1
    # cumulative-sum trick: O(n)
    arr = np.asarray(values, dtype=float)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    # avg over [i-w+1, i]
    ma = (cs[w:] - cs[:-w]) / w
    return ma, w


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    resume_state,
    start_iteration,
    debug_from,
    args,
):
    """Fine-tune an existing Gaussian scene WITHOUT pruning.
    Records (iteration, loss) each step; at the end it ALWAYS saves a PNG,
    and shows a single Matplotlib figure (if interactive):
      - x = iteration
      - y = total loss (raw + moving average)
    """
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    resume_kind, resume_path = resume_state
    first_iter = int(start_iteration)
    resume_iteration = first_iter

    # --- Load state (checkpoint OR explicit point cloud) ---
    if resume_kind == "checkpoint":
        gaussians.training_setup(opt)
        loaded_model, recorded_iter = torch.load(resume_path)
        if recorded_iter != first_iter:
            print(
                f"[finetune] Requested start_iteration={first_iter}, "
                f"but checkpoint records {recorded_iter}; using recorded value.",
                flush=True,
            )
            first_iter = int(recorded_iter)
            resume_iteration = first_iter
        gaussians.restore(loaded_model, opt)
    elif resume_kind == "pointcloud":
        gaussians.load_ply(Path(resume_path).as_posix())
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros(
            (gaussians.get_xyz.shape[0]),
            device=gaussians.get_xyz.device,
        )
    else:
        raise ValueError(f"Unsupported resume state '{resume_kind}'")

    print(f"\nInitial Model: Number of Gaussians is {len(gaussians.get_xyz)}", flush=True)

    # --- Background ---
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- Timing ---
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # --- Loss logs (no resume) ---
    loss_log: list[tuple[int, float]] = []
    l1_log: list[tuple[int, float]] = []

    # --- Loop state ---
    viewpoint_stack = None
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Fine-tuning progress",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.0,
        smoothing=0.0,
    )
    first_iter += 1
    lr_iter = first_iter
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    finished = False  # so we can plot after breaking out

    # --- Main loop ---
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # LR & SH schedules
        gaussians.update_learning_rate(lr_iter)
        if lr_iter % 1000 == 0:
            gaussians.oneupSHdegree()
        if lr_iter % 400 == 0:
            gaussians.scheduler.step()

        # Sample a camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Record raw losses (no guards)
        l1_log.append((iteration, float(Ll1.detach().cpu())))
        loss_log.append((iteration, float(loss.detach().cpu())))

        loss.backward()
        iter_end.record()
        torch.cuda.synchronize()

        if tb_writer:
            tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
            tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
            tb_writer.add_scalar("iter_time", iter_start.elapsed_time(iter_end), iteration)

        # Save/Checkpoint if requested
        if iteration in saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians", flush=True)
            scene.save(iteration)

        if iteration in checkpoint_iterations:
            print(f"\n[ITER {iteration}] Saving Checkpoint", flush=True)
            ensure_dir_for(scene.model_path, treat_as_file=False)
            torch.save(
                (gaussians.capture(), iteration),
                os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
            )

        training_report(
            tb_writer,
            0,  # prune_idx (fixed, since we don't prune)
            iteration,
            Ll1,
            loss,
            l1_loss,
            iter_start.elapsed_time(iter_end),
            testing_iterations,
            scene,
            render,
            (pipe, background),
            after_prune=False,
            init_report=(iteration == testing_iterations[0] if testing_iterations else False),
        )

        # Optimizer step / loop end
        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
        else:
            finished = True
            break

        # Optional "environment refresh"
        if lr_iter == opt.position_lr_max_steps:
            print(f"\n[ITER {iteration}] Refreshing environment", flush=True)
            point_cloud_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
            if os.path.exists(point_cloud_path):
                print(f"Loading {point_cloud_path}/point_cloud.ply", flush=True)
                # Clean and reload
                try:
                    del gaussians.scheduler
                except Exception:
                    pass
                del gaussians.optimizer
                del gaussians.xyz_gradient_accum
                del gaussians.denom
                del gaussians
                del scene.gaussians
                del scene
                gc.collect()
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians)
                gaussians.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
            gaussians.training_setup(opt)
            gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
            gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)
            viewpoint_stack = None
            lr_iter = first_iter
        else:
            lr_iter += 1

        # Progress bar
        CHUNK = 10
        if iteration % CHUNK == 0:
            progress_bar.update(CHUNK)
            try:
                progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})
            except Exception:
                pass

    # fill any remainder and close
    if progress_bar.n < opt.iterations:
        progress_bar.update(opt.iterations - progress_bar.n + 1)
    progress_bar.close()

    # -------- Plot: raw + moving average (always save; show only if interactive) --------
    if loss_log:
        iters = [i for (i, _) in loss_log]
        losses = [v for (_, v) in loss_log]

        # Compute trailing moving average
        ma_vals, used_w = _moving_average_trailing(losses, SMOOTH_WINDOW)
        if used_w > 1 and len(ma_vals) > 0:
            iters_ma = iters[used_w - 1:]
        else:
            iters_ma = iters  # if window=1 or too short, just reuse raw x

        # Create plots directory under the model path
        model_path = Path(getattr(dataset, "model_path", getattr(args, "model_path", ".")))
        plots_dir = model_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        png_path = plots_dir / "loss_vs_iter.png"

        # Headless safety: choose backend before importing pyplot
        if not _is_interactive_session():
            matplotlib.use("Agg")  # non-GUI backend

        import matplotlib.pyplot as plt  # import AFTER backend decision

        plt.figure()
        # Raw (light)
        plt.plot(iters, losses, linewidth=1, alpha=0.3, label="Raw loss")
        # Smoothed (bold) if window > 1
        if used_w > 1 and len(ma_vals) > 0:
            plt.plot(iters_ma, ma_vals, linewidth=2, label=f"Moving average (w={used_w})")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Fine-tune Loss vs Iteration")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="best")

        # --- Horizontal lines for min/max of the moving average (fallback to raw if needed) ---
        if used_w > 1 and len(ma_vals) > 0:
            ma_arr = np.asarray(ma_vals, dtype=float)
            min_idx = int(np.argmin(ma_arr))
            max_idx = int(np.argmax(ma_arr))
            min_y = float(ma_arr[min_idx])
            max_y = float(ma_arr[max_idx])
            min_x = iters_ma[min_idx]
            max_x = iters_ma[max_idx]
            min_label = f"MA min (w={used_w}) = {min_y:.6f}"
            max_label = f"MA max (w={used_w}) = {max_y:.6f}"
        else:
            # Fallback to raw losses if MA is effectively window=1
            raw_arr = np.asarray(losses, dtype=float)
            min_idx = int(np.argmin(raw_arr))
            max_idx = int(np.argmax(raw_arr))
            min_y = float(raw_arr[min_idx])
            max_y = float(raw_arr[max_idx])
            min_x = iters[min_idx]
            max_x = iters[max_idx]
            min_label = f"Raw min = {min_y:.6f}"
            max_label = f"Raw max = {max_y:.6f}"

        # draw the lines
        plt.axhline(y=min_y, linestyle="--", linewidth=1, alpha=0.7, label=min_label)
        plt.axhline(y=max_y, linestyle="--", linewidth=1, alpha=0.7, label=max_label)

        # annotate near those positions for readability
        plt.annotate(f"{min_y:.6f}", xy=(min_x, min_y), xytext=(5, 5),
                     textcoords="offset points", ha="left", va="bottom", fontsize=9)
        plt.annotate(f"{max_y:.6f}", xy=(max_x, max_y), xytext=(5, 5),
                     textcoords="offset points", ha="left", va="bottom", fontsize=9)

        plt.ylim(0, 0.3)

        # Always save the PNG, then show if interactive
        plt.savefig(png_path.as_posix(), dpi=180, bbox_inches="tight")
        print(f"[finetune] Saved loss plot → {png_path}", flush=True)

        if _is_interactive_session():
            plt.show()
        else:
            plt.close()

    if not finished:
        print("[finetune] Training loop ended prematurely (no 'finished' flag).", flush=True)

    return resume_iteration


def _extract_param_groups(args: Namespace) -> Tuple[object, object, object]:
    """
    Instantiate GraphDECO argument groups and extract their namespaces without
    mutating the shared CLI parser.
    """
    dummy_parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(dummy_parser)
    op = OptimizationParams(dummy_parser)
    pp = PipelineParams(dummy_parser)
    defaults = dummy_parser.parse_args([])
    merged = vars(defaults).copy()
    merged.update(vars(args))
    combined = argparse.Namespace(**merged)
    return lp.extract(combined), op.extract(combined), pp.extract(combined)


def _resolve_start_iteration(args: Namespace) -> int:
    if getattr(args, "start_iteration", None) is not None:
        return int(args.start_iteration)
    if getattr(args, "first_iter", None) is not None:
        return int(args.first_iter)
    maybe_checkpoint = getattr(args, "start_checkpoint", None)
    if maybe_checkpoint:
        try:
            return int(Path(maybe_checkpoint).stem.replace("chkpnt", ""))
        except Exception:
            pass
    maybe_pointcloud = getattr(args, "start_pointcloud", None)
    if maybe_pointcloud:
        try:
            return int(Path(maybe_pointcloud).parent.name.split("_")[-1])
        except Exception:
            pass
    return 30_000


def _resolve_resume_state(
    model_dir: Path,
    start_iteration: int,
    args: Namespace,
) -> Tuple[str, Path]:
    checkpoint_path = model_dir / f"chkpnt{int(start_iteration)}.pth"
    if checkpoint_path.exists():
        return "checkpoint", checkpoint_path
    pointcloud_path = model_dir / "point_cloud" / f"iteration_{int(start_iteration)}" / "point_cloud.ply"
    if pointcloud_path.exists():
        return "pointcloud", pointcloud_path

    fallback_checkpoint = getattr(args, "start_checkpoint", None)
    if fallback_checkpoint:
        fallback_checkpoint_path = Path(fallback_checkpoint)
        if fallback_checkpoint_path.exists():
            print(
                "[finetune] Detected legacy --start_checkpoint usage; this interface is deprecated.",
                flush=True,
            )
            return "checkpoint", fallback_checkpoint_path

    fallback_pointcloud = getattr(args, "start_pointcloud", None)
    if fallback_pointcloud:
        fallback_pointcloud_path = Path(fallback_pointcloud)
        if fallback_pointcloud_path.exists():
            print(
                "[finetune] Detected legacy --start_pointcloud usage; this interface is deprecated.",
                flush=True,
            )
            return "pointcloud", fallback_pointcloud_path

    raise RuntimeError(
        (
            f"Iteration {start_iteration} was not found under {model_dir}. "
            f"Expected checkpoint '{checkpoint_path.name}' or point cloud "
            f"'point_cloud/iteration_{start_iteration}/point_cloud.ply'. "
            "Finetuning aborted."
        ),
    )


def _prepare_iteration_lists(
    args: Namespace,
    opt_params: object,
    start_iteration: int,
) -> Tuple[int, int, Sequence[int], Sequence[int], Sequence[int]]:
    target = int(getattr(opt_params, "iterations", 35_000))

    explicit_tests = getattr(args, "test_iterations", None)
    if explicit_tests is None:
        mid = start_iteration + max((target - start_iteration) // 2, 1)
        tests = sorted({start_iteration, mid, target})
    else:
        tests = list(explicit_tests)

    explicit_saves = getattr(args, "save_iterations", None)
    if explicit_saves is None:
        saves = [target]
    else:
        saves = list(explicit_saves)

    explicit_checkpoints = getattr(args, "checkpoint_iterations", None)
    if explicit_checkpoints is None:
        checkpoints = [target]
    else:
        checkpoints = list(explicit_checkpoints)

    #return start_iteration, target, tests, saves, checkpoints
    return start_iteration, target, [0], saves, [0]

def run_finetune_cli(args: Namespace) -> Tuple[Dict[str, object] | None, int]:
    safe_state(bool(getattr(args, "quiet", False)))
    torch.autograd.set_detect_anomaly(bool(getattr(args, "detect_anomaly", False)))

    dataset_params, opt_params, pipe_params = _extract_param_groups(args)

    model_path = Path(getattr(dataset_params, "model_path", getattr(args, "model_path", "."))).resolve()
    dataset_params.model_path = model_path.as_posix()
    print(f"Optimizing {model_path}", flush=True)

    start_iteration = _resolve_start_iteration(args)
    try:
        resume_state = _resolve_resume_state(model_path, start_iteration, args)
    except RuntimeError as exc:
        print(f"[finetune] error: {exc}", file=sys.stderr, flush=True)
        return None, 1

    start_iteration, target_iter, test_iterations, save_iterations, checkpoint_iterations = _prepare_iteration_lists(
        args,
        opt_params,
        start_iteration,
    )
    test_iterations = list(test_iterations)
    save_iterations = list(save_iterations)
    checkpoint_iterations = list(checkpoint_iterations)

    if not test_iterations or all(item <= start_iteration for item in test_iterations):
        print(
            f"[finetune] Test iterations {test_iterations or []} do not exceed start iteration {start_iteration}; no evaluations will be run.",
            flush=True,
        )

    if not save_iterations or all(item <= start_iteration for item in save_iterations):
        print(
            f"[finetune] Save iterations {save_iterations or []} do not exceed start iteration {start_iteration}; the scene will not be saved.",
            flush=True,
        )

    if not checkpoint_iterations or all(item <= start_iteration for item in checkpoint_iterations):
        print(
            f"[finetune] Checkpoint iterations {checkpoint_iterations or []} do not exceed start iteration {start_iteration}; no checkpoints will be saved.",
            flush=True,
        )

    if target_iter <= start_iteration:
        print(
            f"[finetune] Final iteration {target_iter} is not greater than start iteration {start_iteration}; skipping finetuning.",
            flush=True,
        )
        iteration_dir = model_path / f"point_cloud/iteration_{start_iteration}"
        point_cloud_ply = iteration_dir / "point_cloud.ply"
        if not point_cloud_ply.exists() and resume_state[0] == "pointcloud":
            point_cloud_ply = Path(resume_state[1])
            iteration_dir = point_cloud_ply.parent
        payload: Dict[str, object] = {
            "model_path": str(model_path),
            "final_iteration": int(start_iteration),
            "point_cloud_dir": str(iteration_dir),
            "point_cloud_ply": str(point_cloud_ply),
            "test_iterations": test_iterations,
            "save_iterations": save_iterations,
            "checkpoint_iterations": checkpoint_iterations,
            "start_iteration": int(start_iteration),
            "resume_kind": resume_state[0],
            "resume_artifact": str(resume_state[1]),
        }
        return payload, 0

    try:
        effective_start = training(
            dataset_params,
            opt_params,
            pipe_params,
            test_iterations,
            save_iterations,
            checkpoint_iterations,
            resume_state,
            start_iteration,
            getattr(args, "debug_from", -1),
            args,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI caller
        print(f"[finetune] error: {exc}", file=sys.stderr, flush=True)
        return None, 1

    model_path = Path(dataset_params.model_path).resolve()
    final_iteration = int(getattr(opt_params, "iterations", target_iter))
    point_cloud_dir = model_path / f"point_cloud/iteration_{final_iteration}"
    payload = {
        "model_path": str(model_path),
        "final_iteration": final_iteration,
        "point_cloud_dir": str(point_cloud_dir),
        "point_cloud_ply": str(point_cloud_dir / "point_cloud.ply"),
        "test_iterations": test_iterations,
        "save_iterations": save_iterations,
        "checkpoint_iterations": checkpoint_iterations,
        "start_iteration": int(effective_start),
        "resume_kind": resume_state[0],
        "resume_artifact": str(resume_state[1]),
    }
    print("\nFine-tuning complete.", flush=True)
    return payload, 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    from gs_coresets.utils.args_utils import build_commands_parser

    parser = build_commands_parser(selected=("finetune",))
    tokens = ["finetune", *(argv if argv is not None else sys.argv[1:])]
    args = parser.parse_args(tokens)
    _, exit_code = run_finetune_cli(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

# Single-scene CLI workflow

Start with a pretrained GraphDECO model and its dataset. Use an entirely new
output directory and keep the input model unchanged. The commands below start
at iteration 30,000, retain 10% of the Gaussians by Top-K sensitivity, and do
not fine-tune unless you execute the optional section.

Install the package first, following its [README](README.md). The dataset must
be readable by GraphDECO, and the pretrained model directory must contain
`point_cloud/iteration_30000/point_cloud.ply`, `cameras.json`, and `cfg_args`.
Replace the example paths below with your own.

The sensitivity and selection examples use `--device cuda`. Replace it with
`--device cpu` in those commands to run them on CPU. Camera extraction,
GraphDECO rendering/evaluation, and fine-tuning have separate requirements;
the full workflow below requires a CUDA environment.

## Paths and camera extraction

```bash
DATASET=/path/to/scene
BASELINE=/path/to/pretrained/model
BASELINE_PLY="$BASELINE/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT=/path/to/new/output
MODEL="$OUTPUT/model"

# The parent of OUTPUT must exist. Refuse an existing directory.
mkdir -- "$OUTPUT" || exit 1
mkdir -p "$OUTPUT/cameras" "$MODEL/point_cloud/iteration_30000"

python -m gs_coresets_pytorch.cli sens_cams \
  --source "$DATASET" --model "$BASELINE" \
  --out "$OUTPUT/cameras" \
  --extract-data-device cpu --extract-resolution -1
```

## Sensitivities

This example computes RGB L1 sensitivities from real training cameras. The
PyTorch command writes all six granularity tensors; the selection step below
uses the scene-level tensor.

```bash
python -m gs_coresets_pytorch.cli sens \
  --model "$BASELINE_PLY" \
  --cams-train "$OUTPUT/cameras/extract_cameras_train.json" \
  --use-train-only --out "$OUTPUT/sensitivities" \
  --sensitivity-norm l1 --sensitivity-reduce max \
  --no-sensitivity-nocolor --device cuda
```

The expected selection input is `per_scene_l1_max.pt`. Never reuse a
sensitivity directory for a different backend or definition. A directory
input to `coreset` resolves one exact filename from the requested granularity,
norm, reduction, and color option. It does not verify backend metadata for you;
select the directory produced by the intended sensitivity run.

## Top-K selection with exact source rows

```bash
python -m gs_coresets_pytorch.cli coreset \
  --model "$BASELINE_PLY" --sens "$OUTPUT/sensitivities" \
  --sens-granularity per_scene --sens-norm l1 --sens-reduce max \
  --no-sens-nocolor --prune-ratio 0.90 --sampling-mode topk \
  --no-allow-duplicates --no-apply-ss-weights --no-normalize-weights \
  --preserve-raw-parameters --device cuda \
  --out "$MODEL/point_cloud/iteration_30000/point_cloud.ply"

cp "$BASELINE/cameras.json" "$BASELINE/cfg_args" "$MODEL/"
```

Retained count is `max(1, round((1 - prune_ratio) * original_count))`, capped
at the source count. Top-K normalizes the scores and ranks them in descending
order, breaking equal scores by ascending original Gaussian index. It does
not use a seed. This stable tie rule can differ from native `torch.topk`,
which does not specify membership among tied cutoff scores. Output vertex
rows are ordered by original index.

`--preserve-raw-parameters` copies complete binary PLY vertex records,
including non-unit stored quaternions and additional scalar properties.
It requires unique, unit-weight selection and enough positive score support;
nonfinite/negative scores, ASCII source PLYs and list-valued vertex properties
are rejected. This option is off by default. Without it, the output is
constructed from loaded Gaussian tensors rather than copied vertex records;
additional source properties are not preserved.

Raw selection writes `point_cloud.ply.selection.json` alongside the PLY.
The manifest records selected original indices, source/score/output hashes,
counts, selection mode, and the stable tie rule. Override its location with
`--selection-manifest PATH` only together with raw preservation.

The coreset CLI refuses existing output files. Files are staged and validated
before no-replace publication; the raw-selection manifest is published last
as a completion marker. A handled publication failure removes only newly
published files from that invocation. A killed process can leave an incomplete
PLY without a manifest: inspect it and choose a new destination rather than
overwriting it.

For seeded sampling instead, use `--sampling-mode multinomial --seed 0`
with no duplicates or weights when retaining raw parameters. Sampling and
Top-K are different selection rules. `--uniform` replaces the sensitivity
arguments with uniform scores, not the choice of selection mode.

## Render and evaluate without fine-tuning

```bash
python -m gs_coresets_pytorch.cli render \
  --source_path "$DATASET" --model_path "$MODEL" \
  --iteration 30000 --resolution -1 --skip_train
python -m gs_coresets_pytorch.cli metrics --model_paths "$MODEL"
```

These commands run the pinned stock GraphDECO scripts, not the renderer used
inside sensitivity computation. Results are written under `$MODEL`:

- `test/ours_30000/renders/`: rendered test views.
- `test/ours_30000/gt/`: matching reference images.
- `results.json`: aggregate PSNR, SSIM, and LPIPS.
- `per_view.json`: per-view measurements.

Check that render/reference counts match and all metric values are finite;
a successful command exit alone does not establish a complete evaluation.

## Optional fine-tuning

To run 100 additional training iterations after selection:

```bash
python -m gs_coresets_pytorch.cli finetune \
  --source_path "$DATASET" --model_path "$MODEL" \
  --start_iteration 30000 --iterations 30100 \
  --save_iterations 30100 --eval
python -m gs_coresets_pytorch.cli render \
  --source_path "$DATASET" --model_path "$MODEL" \
  --iteration 30100 --resolution -1 --skip_train
python -m gs_coresets_pytorch.cli metrics --model_paths "$MODEL"
```

Use a new copied model directory if you want to preserve an earlier evaluation:
GraphDECO metrics commands may replace the model directory's metric JSON files.
No operation in this guide should target the pretrained input directory.

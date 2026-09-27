#!/usr/bin/env python3
# gs_coresets.utils/camera_generator.py
# -----------------------------------------------------------------------------
# ROI-aware CameraGenerator with robust placement + diagnostics.
# - Prints scene AABB coordinates and volume
# - Prints ROI bbox coordinates and volumes
# - Prints selected anchor coordinates (and distance to nearest gaussian)
# - Prints accepted camera eye positions, the Fibonacci direction used, and the
#   actual view direction (eye->anchor), plus a visibility hit count
# - For OPEN scenes, constrains the ROI to the convex hull of the Gaussians if
#   SciPy is available; otherwise uses an AABB-based fallback
# - Visibility gate ensures each camera sees geometry inside its FOV
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from gs_coresets.utils.io_utils import load_gaussian_ply, write_json
from gs_coresets.utils.types_devices_utils import torch_ctx, as_4x4_list
from gs_coresets.utils.camera_utils import Camera, look_at_c2w, projected_covariance_2d
from gs_coresets.utils.scene_utils import (
    fibonacci_sphere,
    grid_size_from_scene_logscale as grid_size_from_scene,
    world_to_index,
    index_to_world,
    voxel_size_world,
    clamp_indices_np,
    robust_aabb,
    AABB_DEFAULTS,
)
from gs_coresets.utils.debug_utils import (
    print_debug,
    IS_DEBUG_MODE,
    debug_gauss_histograms,
    debug_scene_plots,
)

# Optional convex hull (preferred for open scenes)
try:
    from scipy.spatial import ConvexHull, Delaunay  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# --------------------------------- ROI type ----------------------------------

@dataclass
class ROI:
    kind: str  # "closed" or "open"
    center: Tuple[float, float, float]
    aabb_min: Tuple[float, float, float]
    aabb_max: Tuple[float, float, float]
    voxel_count: int  # only for closed
    label_id: int  # component id in interior labels (0 for open)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------- Main class ----------------------------------

class CameraGenerator:
    """
    Constructor:
        CameraGenerator(ply_path: str, out_json_path: str, seed: Optional[int] = None)

    Methods:
        generate(num_cameras: int, target_coverage: float = 0.99) -> List[Camera]
    """

    # Camera Intrinsics:
    DEFAULT_W = 800
    DEFAULT_H = 800
    DEFAULT_FOV_DEG = 60.0  # Decides fx/fy

    # Grid / walls
    LOW_PCT = AABB_DEFAULTS.pct_bounds[0]
    HIGH_PCT = AABB_DEFAULTS.pct_bounds[1]
    MUL_FACT = AABB_DEFAULTS.mul_fact
    DILATE_VOXELS = 2  # Thicken walls before flood-fill during ROI discovery (dilate it by a Chebyshev radius dil)

    # Hard minimum eye–splat distances:
    # - Eyes must be at least: max( MIN_EYE_TO_GAUSS_FRAC * diag , MIN_CLEAR_SIGMA_MULT * sigma_median_world )
    #   from the nearest Gaussian (“world-scale” and “sigma-scale” guards).
    # - Enforced by pushing the eye outward along its ray until the constraint holds.
    #
    # Practical effect:
    # - If getting lots of “too_close” rejections, raise either knob.
    # - If cameras feel too far from geometry (tiny splats on screen):
    #   Lower the sigma multiplier first; The frac term is a GLOBAL safety rail.
    MIN_EYE_TO_GAUSS_FRAC = 0.01  # Closest Gaussian distance at least (MIN_EYE_TO_GAUSS_FRAC * ROI_diag).
    MIN_CLEAR_SIGMA_MULT = 3.0  # Closest Gaussian distance at least (MIN_CLEAR_SIGMA_MULT * median-sigma).

    ## Placement -- Per-camera depth range relative to distance.
    # This bounds what counts as “in front” for visibility/coverage tests and rendering metadata:
    # - Smaller NEAR admits very close geometry.
    # - Larger FAR_MULT accepts farther hits (useful outdoors).
    FAR_MULT = 8.0  # For each candidate: far = FAR_MULT * dist
    NEAR = 0.01  # For each candidate: near = max(NEAR, 0.01 * dist)

    ## Eye Sampling / Target sampling.
    # EYE_OVERSAMPLE_FACTOR:
    #   How many candidate eye positions (camera centers) to generate for every one ultimately kept.
    #   Oversample first, then prune with the diversity gates below.
    #   Higher = better coverage/quality, more compute.
    EYE_OVERSAMPLE_FACTOR = 3  # OLD=5
    # FIB_SAMPLES_PER_EYE:
    #   For each eye, how many look directions to try (uniform-ish on the sphere via Fibonacci sampling).
    #   More samples = more ways to aim that same camera center at different parts of the scene.
    FIB_SAMPLES_PER_EYE = 12  # OLD=24
    # TARGET_CLUSTER_COUNT:
    #   K for clustering scene geometry (e.g., Gaussian means) into “points of interest” (POIs).
    #   Targets that we look at are drawn from these cluster centers.
    #   Larger K captures finer structure but costs more when scoring visibility.
    TARGET_CLUSTER_COUNT = 192  # OLD=128
    # TARGET_POISSON_RADIUS_FRAC:
    #   A Poisson-disk angular spacing on the target directions chosen per eye (expressed as a fraction of the
    #   unit-sphere scale that's used).
    #   Prevents two look-at targets being nearly collinear from the same eye.
    #   Increase if seeing many near-duplicate aim directions per eye.
    TARGET_POISSON_RADIUS_FRAC = 0.03

    # ---- Adaptive dirs/eye + budgets + early-stop ----
    GLOBAL_CANDIDATE_BUDGET_MUL = 8  # hard cap on total built candidates (pre-selection) ; OLD=4
    DIRS_PER_EYE_MIN = 3  # floor for per-eye directions ; OLD=4
    DIRS_PER_EYE_MAX = 8  # ceiling for per-eye directions ; OLD=12
    INWARD_FADE_VOX = 6  # Number of voxels after NEAR_BOUNDARY_VOX for fade-to-0 inward-dot threshold

    ## Diversity gates (near-duplicate pose filtering)
    # EYE_POISSON_RADIUS_FRAC:
    #   Poisson-disk-sampling spatial spacing for the eyes (camera centers).
    #   It’s normalized by the RIO scale (usually RIO diagonal).
    #   Larger values force cameras farther apart.
    EYE_POISSON_RADIUS_FRAC = 0.05  # OLD=0.08
    # PER_EYE_TARGET_ANG_DEG:
    #   Minimum angular separation (in degrees) between any two targets selected from the same eye.
    #   This is a simple, cheap gate in addition to the target Poisson disk.
    PER_EYE_TARGET_ANG_DEG = 22.0  # OLD=18.0
    # POSE_SIM_EYE_DIST_FRAC:
    #   Two candidate poses are considered “near” if their eye positions are within 0.05 × scene_diag.
    #   Poses that are near and look similar are penalized/rejected.
    POSE_SIM_EYE_DIST_FRAC = 0.04
    # POSE_SIM_LOOK_ANG_DEG:
    #   “Look-vector” similarity threshold in degrees.
    #   Combined with the eye-distance threshold above to decide if two poses are effectively the same view.
    POSE_SIM_LOOK_ANG_DEG = 8.0  # OLD=12.0
    # POSE_SIM_JACCARD_THRESH:
    #   Content-overlap test:
    #       Compute Jaccard similarity between sets of visible items (e.g., Gaussian IDs) for two poses.
    #       If ≥ POSE_SIM_JACCARD_THRESH, they’re considered redundant.
    #   Helps catch duplicates even when the camera moved a little.
    POSE_SIM_JACCARD_THRESH = 0.6  # OLD=0.8
    # POSE_SIM_PENALTY_WEIGHT:
    #   How strongly to downweight a candidate pose’s score when it’s similar to already accepted poses.
    #   If seeing too many lookalikes slipping through, bump this up (or lower the Jaccard threshold).
    POSE_SIM_PENALTY_WEIGHT = 0.7  # OLD=0.35
    # POSE_SIM_RECENT_LIMIT:
    #   For speed, compare each candidate only against the last N accepted poses instead of all historical ones.
    #   Increase for stricter global diversity; keep modest to avoid O(N²) blowups.
    POSE_SIM_RECENT_LIMIT = 64  # OLD: POSE_SIM_RECENT_LIMIT = 16

    ## Visibility Gating.
    # For each candidate, run a fast visibility sample against _xyz_vis with the near/far gates and FoV.
    # If (hits < VIS_MIN_HITS), it’s dropped (unless ENFORCE_VISIBILITY=False).
    # This filters poses that don’t actually see geometry even if they’re “legal” by distance/angle alone.
    ENFORCE_VISIBILITY = True  # set False if only want diagnostics.
    VIS_SAMPLE_MAX = 12000
    VIS_MIN_HITS = 60
    # A small angular slack in the FOV gate:
    #   Add VIS_FOV_MARGIN_DEG to the half-FOV when checking angles, so near-edge geometry isn’t spuriously
    #   rejected due to discretization.
    VIS_FOV_MARGIN_DEG = 6.0

    ## Coverage check.
    # Coverage uses the screen-space covariance helper and counts a Gaussian as covered if its ellipse (at k-sigma) is
    # inside the frame and in front of the camera, with a minimum pixel sigma to avoid degenerate sub-pixel acceptance.
    # That’s what drives “target_coverage” reporting and the greedy candidate picker’s gain vector.
    COV_K_SIGMA = 3.0  # How many sigmas count a Gaussian as "in pixel" for coverage.
    MIN_SIGMA_PX = 0.5  # Minimum pixel sigma.

    # Boundary "inward" test if near boundary ("Point inward when you’re near a wall").
    # When an eye is near the boundary, form a surface-inward normal/gradient and reject look directions
    # whose dot with the inward vector is below INWARD_DOT_THRESH (i.e., glancing or pointing outward).
    INWARD_DOT_THRESH = 0.25  # Used for the dot-test.
    NEAR_BOUNDARY_VOX = 3  # Voxel proximity guard (when to apply the dot-test).

    # CDIST_CHUNK_COLS:
    #   cdist memory guard (columns processed per chunk when matching against many gaussians)
    #   When you compute big distance matrices (e.g., torch.cdist between many candidates and many Gaussians), do it in
    #   column chunks of this size to cap memory.
    #   Rough memory per chunk (float32) ≈ rows × CDIST_CHUNK_COLS × 4 bytes.
    #   Lower this if hitting OOM; raise it to go faster if got memory headroom.
    CDIST_CHUNK_COLS = 4096  # 4096 cols × 2000 rows ≈ 32M floats ≈ 128MB per chunk

    ## Hull backend
    # HULL_BACKEND:
    #   How to approximate the scene’s outer shell:
    #   -   "kdop" = k-Discrete Oriented Polytope (project onto K directions → build a tight-ish convex polytope).
    #           Fast, GPU-friendly, robust.
    #   -   "scipy" = exact convex hull via Qhull (nice but memory/point-count sensitive).
    #   -   "auto" = prefilter then try SciPy; if it looks too big/slow, fall back to k-DOP.
    HULL_BACKEND = "kdop"  # {"kdop", "auto", "scipy"} ; "auto" tries SciPy on a reduced set, falls back to k-DOP
    # KDOP_K:
    #   Number of directions on the sphere for the k-DOP.
    #   Higher → tighter hull, more projection work.
    #   96–256 is a good sweet spot; 160 is nicely balanced.
    KDOP_K = 160  # directions on S^2
    # KDOP_MARGIN_FRAC:
    #   Grows all hull planes outward by margin × scene_diag to avoid floating-point edge cases
    #   (e.g., rays grazing the hull).
    #   Push this up slightly if you see eyes clamped “inside” by accident.
    KDOP_MARGIN_FRAC = 1e-3  # grows planes outward by this * scene_diag to avoid FP edge cases
    # KDOP_CHUNK:
    #   Project points onto directions in chunks of this many points to bound GPU/CPU memory during hull construction.
    #   Lower if tight on memory.
    KDOP_CHUNK = 65536  # project N points in chunks to bound memory
    # EXTREME_FILTER_K:
    #   When using "auto"/"scipy", first do a GPU pass:
    #       => For a set of random directions, keep only the extreme points (min/max along each direction).
    #   This reduces the point set fed to Qhull. Higher keeps more candidate extremes (slower but safer).
    EXTREME_FILTER_K = 128  # when "auto"/"scipy": GPU prefilter directions
    # MAX_QHULL_POINTS:
    #   Hard safety cap on points passed to SciPy’s Qhull.
    #   If exceeded, the code will fall back to k-DOP to avoid crashes/timeouts.
    MAX_QHULL_POINTS = 200000  # safety cap if you ever call SciPy

    # ===== Distinctiveness & multi-direction coverage knobs =====
    # Spherical directional bins per TARGET (cluster) to encourage multi-view coverage.
    DIR_BIN_COUNT = 24  # number of bins on the sphere for each target
    DIR_NOVELTY_WEIGHT = 0.8  # how strongly we reward unseen direction bins per target

    # Distance stratification around targets (relative to ROI diagonal of the candidate's ROI).
    # Bin edges are fractions of the ROI diagonal (e.g., 0.05 => 5% of ROI diag).
    DIST_BIN_EDGES_FRAC = (0.06, 0.12, 0.25)  # yields 4 bins: [0,0.06), [0.06,0.12), [0.12,0.25), [0.25,+inf)
    DIST_BALANCE_WEIGHT = 0.45  # reward under-sampled distance bins per target

    # Encourage covering targets that currently have fewer accepted views.
    TARGET_BALANCE_WEIGHT = 0.25  # reward for targets with low total selection count

    # Set to False to disable the whole distinctiveness layer (for A/B checks).
    ENABLE_DIRECTIONAL_DIVERSITY = True


    # ----------------------------- Inward-fade helpers (DRY; no numeric change) -----------------------------
    # Note:
    #   reset the cache once per grid build.
    #   Add this single line after finishing _build_rois_and_fields:
    #       self._inward_cache = None

    def _inward_params(self) -> tuple[float, float]:
        """
        Cached conversion of 'NEAR_BOUNDARY_VOX' and 'INWARD_FADE_VOX' into world units
        for the current grid. Invalidated whenever grids are (re)built.
        Returns: (near_thr_world, fade_world)
        """
        cache = getattr(self, "_inward_cache", None)
        if cache is not None:
            return cache
        vsz = voxel_size_world(self._aabb_min, self._aabb_max, self._grid["dims"]).detach().cpu().numpy()
        mean_v = float(np.mean(vsz))
        near_thr_world = float(self.NEAR_BOUNDARY_VOX) * mean_v
        fade_world = float(self.INWARD_FADE_VOX) * mean_v
        self._inward_cache = (near_thr_world, fade_world)
        return self._inward_cache

    def _effective_inward_thresh(self, eye_world: torch.Tensor) -> float:
        """
        Distance-aware inward dot threshold:
          - d <= near_thr_world           -> INWARD_DOT_THRESH
          - d >= near_thr_world+fade     -> 0
          - otherwise linearly interpolated to 0
        """
        near_thr_world, fade_world = self._inward_params()
        d_world = self._dist_to_boundary_world(eye_world)
        base = float(self.INWARD_DOT_THRESH)
        if d_world <= near_thr_world:
            return base
        if d_world >= (near_thr_world + fade_world):
            return 0.0
        frac = 1.0 - float((d_world - near_thr_world) / max(fade_world, 1e-9))
        return base * max(0.0, min(1.0, frac))


    # ---- Single-source ROI diagonal helper (DRY; used everywhere) ----
    @staticmethod
    def __roi_diag_len(roi: "ROI") -> float:
        a = np.asarray(roi.aabb_min, dtype=np.float32)
        b = np.asarray(roi.aabb_max, dtype=np.float32)
        return float(np.linalg.norm(b - a))

    # ---- Near/Far helper (DRY; used in visibility + camera metadata) ----
    def _near_far_for_dist(self, dist: float) -> tuple[float, float]:
        """
        Given a target distance, produce per-candidate near/far planes:
          near = max(NEAR, 0.01 * dist)
          far  = FAR_MULT * max(1.0, dist)
        """
        d = float(dist)
        return (max(float(self.NEAR), 0.01 * d), float(self.FAR_MULT) * max(1.0, d))

    # ----------------------------- Debug counters -----------------------------

    def _reset_dbg(self) -> None:
        self._dbg = {
            "eyes_total": 0,
            "candidate_attempts": 0,
            "candidates_total": 0,
            "accepted": 0,
            "rejected_outside_roi": 0,
            "rejected_too_close_gauss": 0,
            "rejected_boundary_dir": 0,
            "rejected_no_geom": 0,
        }

    # ------------ k-DOP builder and membership helpers ------------

    @staticmethod
    def __kdop_project_minmax(pts: torch.Tensor, dirs: torch.Tensor, chunk: int) -> tuple[
        torch.Tensor, torch.Tensor]:
        """
        Compute min/max projections of pts along each direction in dirs.
        Works in chunks to avoid allocating (N x K) all at once.
        pts: (N,3)  dirs: (K,3)  returns (bmin[K], bmax[K])
        """
        dev, dt = pts.device, pts.dtype
        K = dirs.shape[0]
        bmin = torch.full((K,), float("inf"), device=dev, dtype=dt)
        bmax = torch.full((K,), float("-inf"), device=dev, dtype=dt)
        for s in range(0, pts.shape[0], chunk):
            blk = pts[s:s + chunk]  # (B,3)
            proj = blk @ dirs.t()  # (B,K)
            bmin = torch.minimum(bmin, proj.min(dim=0).values)
            bmax = torch.maximum(bmax, proj.max(dim=0).values)
        return bmin, bmax

    def _build_open_polytope(self, k: int | None = None, margin_frac: float | None = None) -> None:
        """
        Build a k-DOP H-rep (dirs, bmin, bmax) for open scenes using all gaussians (or a subset).
        """
        if k is None: k = self.KDOP_K
        if margin_frac is None: margin_frac = self.KDOP_MARGIN_FRAC
        dev, dt = self.device, self.dtype

        # Directions on the sphere (reuse your existing Fibonacci utility)
        dirs = fibonacci_sphere(k, dev, dt)
        dirs = dirs / (torch.linalg.norm(dirs, dim=1, keepdim=True) + 1e-9)  # (K,3)

        # Min/max supports along each direction
        bmin, bmax = CameraGenerator.__kdop_project_minmax(self._xyz, dirs, self.KDOP_CHUNK)

        # Grow a tiny margin to be robust to FP & unseen anchors
        eps_grow = float(margin_frac) * float(self._diag)
        self._kdop_dirs = dirs  # (K,3)
        self._kdop_bmin = bmin - eps_grow  # (K,)
        self._kdop_bmax = bmax + eps_grow  # (K,)

    def _inside_open_polytope(self, p: torch.Tensor) -> bool:
        """
        Scalar point-in-polytope test for open ROI usage.
        """
        if hasattr(self, "_kdop_dirs"):
            s = (p.view(1, 3) @ self._kdop_dirs.t()).squeeze(0)  # (K,)
            return bool(((s <= self._kdop_bmax) & (s >= self._kdop_bmin)).all())

        if self._hull_delaunay is not None:
            return self._inside_open_hull(p)

        return bool(torch.all(p >= self._aabb_min) & torch.all(p <= self._aabb_max))

    # ----------------------------- ROI discovery -----------------------------
    ## Why a scene gets “open” vs “closed”:
    ####  You get closed ROIs only if some interior exists after dilation + flood-fill (i.e., there are enclosed voids).
    ####  You get open if either:
    ####  * The occupied mass is too sparse or thin (no enclosed pockets after dilation), or
    ####  * dilation is so aggressive it seals to the boundary and eats all interior (interior becomes empty), or
    ####  * any enclosed pockets are below the size filter and get dropped.
    ## Algorithm Summary:
    ####  Build occupancy → dilate → flood-fill boundary to mark outside → (interior = enclosed air).
    ####  Connected-components on interior → each is a closed ROI with world-space AABB & center.
    ####  If none found, emit one open ROI covering the scene AABB and (optionally) a hull for geometry queries.
    ####  Also compute a distance field inside interiors for downstream placement logic.
    ## Performance:
    ####  dilation is O(#occ · dil³)
    ####  BFS/labeling are O(#voxels).
    ####  ==> Keep grid sizes reasonable.
    def _build_rois_and_fields(self) -> List[ROI]:
        xyz = self._xyz
        N = xyz.shape[0]
        aabb_min = self._aabb_min
        aabb_max = self._aabb_max

        # --- Local helpers (refactor; no behavior change) ---
        def _boundary_voxel_count(nx: int, ny: int, nz: int) -> int:
            """Total boundary-voxel count for a (nx,ny,nz) grid."""
            return int(2 * (nx * ny + nx * nz + ny * nz))

        def _enqueue_boundary_free(free: np.ndarray, outside: np.ndarray):
            """Initialize queue with all boundary voxels that are free.
            Returns (queue, boundary_free_count)."""
            from collections import deque
            nx, ny, nz = free.shape
            q = deque()
            cnt = 0

            def push_if_free(x: int, y: int, z: int):
                nonlocal cnt
                if free[x, y, z] and not outside[x, y, z]:
                    outside[x, y, z] = True
                    q.append((x, y, z))
                    cnt += 1

            for x in range(nx):
                for y in range(ny):
                    push_if_free(x, y, 0)
                    push_if_free(x, y, nz - 1)
            for x in range(nx):
                for z in range(nz):
                    push_if_free(x, 0, z)
                    push_if_free(x, ny - 1, z)
            for y in range(ny):
                for z in range(nz):
                    push_if_free(0, y, z)
                    push_if_free(nx - 1, y, z)
            return q, cnt

        def _flood_fill_outside(free: np.ndarray) -> tuple[np.ndarray, int, int]:
            """6-connected flood fill from boundary free space.
            Returns (outside_mask, boundary_free_count, boundary_total).
            """
            nx, ny, nz = free.shape
            outside = np.zeros_like(free, dtype=np.bool_)
            q, boundary_free = _enqueue_boundary_free(free, outside)
            # BFS
            while q:
                x, y, z = q.popleft()
                for xx, yy, zz in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1),
                                   (x, y, z - 1)):
                    if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                        if free[xx, yy, zz] and not outside[xx, yy, zz]:
                            outside[xx, yy, zz] = True
                            q.append((xx, yy, zz))
            return outside, int(boundary_free), _boundary_voxel_count(nx, ny, nz)

        # 1) Discretize the scene into a voxel grid:
        ## A) Get grid size from scene
        nx, ny, nz = grid_size_from_scene(N)
        _dbg_grid_size = nx * ny * nz
        print_debug(f"grid_size_from_scene: {nx}*{ny}*{nz} = {_dbg_grid_size:,}")
        # Occupancy grid
        occ = np.zeros((nx, ny, nz), dtype=np.bool_)
        ## B) Map Gaussians means from world-coords to grid-indices
        ix_t, iy_t, iz_t = world_to_index(xyz, aabb_min, aabb_max, (nx, ny, nz))
        ## C) Clamp indices to grid's actual dims
        ix = ix_t.detach().cpu().numpy()
        iy = iy_t.detach().cpu().numpy()
        iz = iz_t.detach().cpu().numpy()
        ix, iy, iz = clamp_indices_np(ix, iy, iz, nx, ny, nz)
        ## D) Mark grid cells as occupied
        occ[ix, iy, iz] = True

        if IS_DEBUG_MODE:
            print_debug("---------------------- PRE-DILATION ----------------------")
            ## Fast Gaussian histograms (visual sanity checks)
            debug_gauss_histograms(xyz, low_pct=self.LOW_PCT, high_pct=self.HIGH_PCT, mul_fact=self.MUL_FACT)

            ## Plot 3 scene plots for visual sanity
            # Flood-fill for interior/boundary detection
            free = ~occ
            total = int(nx * ny * nz)
            free_num = int(np.count_nonzero(free))
            print_debug(
                f"Grid voxels pre-dilation: {total:,} total; {free_num:,} free ({100 * float(free_num) / max(total, 1):.2f}%).")
            outside, bfree, btotal = _flood_fill_outside(free)
            print_debug(
                f"Boundary voxels pre-dilation: {btotal:,} total; {bfree:,} free ({100 * float(bfree) / max(btotal, 1):.2f}%).")
            interior = free & (~outside)
            # Plot results
            dev = self._xyz.device
            self._grid_torch = {
                "occ": torch.as_tensor(occ, device=dev, dtype=torch.bool),
                "free": torch.as_tensor(free, device=dev, dtype=torch.bool),
                "outside": torch.as_tensor(outside, device=dev, dtype=torch.bool),
                "interior": torch.as_tensor(interior, device=dev, dtype=torch.bool),
                "dims": (nx, ny, nz),
            }
            self._room_air_aabb, self._room_wall_aabb, _, _, _ = debug_scene_plots(
                self._grid_torch, self._aabb_min, self._aabb_max, draw_boxes=True, figsize=(8, 12), show=True
            )
        # 2) “Thicken” the occupied set (dilation):
        # 2.1) Estimate by how many voxels to dilate:
        ## A) Compute average voxel edge length in world units (3D vec, then use mean on components):
        vsz = voxel_size_world(aabb_min, aabb_max, (nx, ny, nz)).detach().cpu().numpy()
        print_debug(f"[DIL] voxel_per_edge_length in world units: {vsz}")
        mean_vsz = float(np.mean(vsz))
        print_debug(f"[DIL] mean voxel_edge_length in world units: {mean_vsz}")
        ## B) Compare avg_voxel_edge to median_gaussian_sigma:
        sigma_med_world = self._sigma_med_world  # Median Gaussian sigma in world units:
        print_debug(f"[DIL] median_gaussian_sigma in world units: {sigma_med_world}")
        # avg_voxels_per_gaussian = median_gaussian_sigma / mean_voxel_edge_length:
        avg_vox_per_gauss = round(sigma_med_world / max(1e-12, mean_vsz))
        print_debug(f"[DIL] avg_vox_per_gauss = round(sigma_med_world / mean_vsz)")
        print_debug(
            f"[DIL] ==> avg_vox_per_gauss = "
            f"round({sigma_med_world / max(1e-12, mean_vsz):.2f}) = "
            f"{avg_vox_per_gauss}"
        )
        ## C) Set dil = avg_vox_per_gauss, but in ranges [DILATE_VOXELS, ...] -- (DILATE_VOXELS >= 1)
        # Clip min avg_voxels_per_gaussian by DILATE_VOXELS:
        dil = self.DILATE_VOXELS * int(max(1, avg_vox_per_gauss))
        # print_debug(f"[DIL] dil = max(self.DILATE_VOXELS, avg_vox_per_gauss)")
        # print_debug(
        #     f"[DIL] ==> dil = max({self.DILATE_VOXELS},{round(sigma_med_world / max(1e-12, mean_vsz))}) "
        #     f"= {dil}"
        # )
        # Clip min avg_voxels_per_gaussian by 1:
        dil = int(max(1, dil))
        print_debug(f"[DIL] dil = dil.clipmin(1) = {dil}")

        # 2.2) for every occupied voxel, mark all neighbors within a Chebyshev radius dil (i.e., the (2dil+1)^3 cube)
        # as occupied. ➜ Purpose: small gaps and thin walls in splat clouds get “sealed” so the flood-fill in the
        # next step behaves robustly.
        occ_dil = occ.copy()
        ix_all, iy_all, iz_all = np.where(occ)
        for dx in range(-dil, dil + 1):
            for dy in range(-dil, dil + 1):
                for dz in range(-dil, dil + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    sx = np.clip(ix_all + dx, 0, nx - 1)
                    sy = np.clip(iy_all + dy, 0, ny - 1)
                    sz = np.clip(iz_all + dz, 0, nz - 1)
                    occ_dil[sx, sy, sz] = True
        occ = occ_dil

        print_debug("---------------------- POST-DILATION ----------------------")
        # 3) Mark the outside (free space reachable from the grid boundary):
        free = ~occ
        total = int(nx * ny * nz)
        free_num = int(np.count_nonzero(free))
        print_debug(
            f"Grid voxels post-dilation: {total:,} total; {free_num:,} free ({100 * float(free_num) / max(total, 1):.2f}%).")

        outside, bfree, btotal = _flood_fill_outside(free)
        print_debug(
            f"Boundary voxels post-dilation: {btotal:,} total; {bfree:,} free ({100 * float(bfree) / max(btotal, 1):.2f}%).")

        interior = free & (~outside)
        print_debug(f"==> Free voxels: {np.count_nonzero(free):,}.")
        print_debug(f"==> Not outside: {np.count_nonzero(~outside):,}.")
        print_debug(f"==> Interior = Free & ~outside: {np.count_nonzero(interior):,}.")

        if IS_DEBUG_MODE:
            ## Plot 3 scene plots for visual sanity
            dev = self._xyz.device
            self._grid_torch = {
                "occ": torch.as_tensor(occ, device=dev, dtype=torch.bool),
                "free": torch.as_tensor(free, device=dev, dtype=torch.bool),
                "outside": torch.as_tensor(outside, device=dev, dtype=torch.bool),
                "interior": torch.as_tensor(interior, device=dev, dtype=torch.bool),
                "dims": (nx, ny, nz),
            }
            self._room_air_aabb, self._room_wall_aabb, _, _, _ = debug_scene_plots(
                self._grid_torch, self._aabb_min, self._aabb_max, draw_boxes=True, figsize=(8, 12), show=True
            )

        print_debug("---------------------- Done building grids! ----------------------")

        # Label connected components inside interior (CLOSED rooms)
        labels = np.zeros_like(interior, dtype=np.int32)
        rois: List[ROI] = []
        label_id = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if not interior[x, y, z] or labels[x, y, z] != 0:
                        continue
                    comp = []
                    labels[x, y, z] = label_id + 1
                    comp.append((x, y, z))
                    q = [(x, y, z)]
                    while q:
                        xx, yy, zz = q.pop()
                        for nb in (
                                (xx + 1, yy, zz),
                                (xx - 1, yy, zz),
                                (xx, yy + 1, zz),
                                (xx, yy - 1, zz),
                                (xx, yy, zz + 1),
                                (xx, yy, zz - 1),
                        ):
                            nx_, ny_, nz_ = nb
                            if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                                if interior[nx_, ny_, nz_] and labels[nx_, ny_, nz_] == 0:
                                    labels[nx_, ny_, nz_] = label_id + 1
                                    q.append((nx_, ny_, nz_))
                                    comp.append((nx_, ny_, nz_))
                    count = len(comp)
                    if count < max(20, (nx * ny * nz) // 20000):
                        for (xx, yy, zz) in comp:
                            labels[xx, yy, zz] = 0
                        continue
                    xs = np.array([p[0] for p in comp])
                    ys = np.array([p[1] for p in comp])
                    zs = np.array([p[2] for p in comp])
                    ix0, ix1 = int(xs.min()), int(xs.max())
                    iy0, iy1 = int(ys.min()), int(ys.max())
                    iz0, iz1 = int(zs.min()), int(zs.max())
                    wmin = index_to_world(ix0, iy0, iz0, self._aabb_min, self._aabb_max, (nx, ny, nz)).astype(
                        np.float32)
                    wmax = index_to_world(ix1, iy1, iz1, self._aabb_min, self._aabb_max, (nx, ny, nz)).astype(
                        np.float32)
                    c_idx = np.array([0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1), 0.5 * (iz0 + iz1)], dtype=np.float64)
                    c_world = index_to_world(
                        int(round(c_idx[0])),
                        int(round(c_idx[1])),
                        int(round(c_idx[2])),
                        self._aabb_min,
                        self._aabb_max,
                        (nx, ny, nz),
                    ).astype(np.float32)
                    rois.append(
                        ROI(
                            kind="closed",
                            center=(float(c_world[0]), float(c_world[1]), float(c_world[2])),
                            aabb_min=(float(wmin[0]), float(wmin[1]), float(wmin[2])),
                            aabb_max=(float(wmax[0]), float(wmax[1]), float(wmax[2])),
                            voxel_count=count,
                            label_id=label_id + 1,
                        )
                    )
                    label_id += 1

        rois.sort(key=lambda r: r.voxel_count, reverse=True)

        # If no closed regions: OPEN scene — use convex hull as ROI if possible
        if len(rois) == 0:
            c = self._center.detach().cpu().numpy().astype(np.float32)
            rois.append(
                ROI(
                    kind="open",
                    center=(float(c[0]), float(c[1]), float(c[2])),
                    aabb_min=(float(self._aabb_min[0]), float(self._aabb_min[1]), float(self._aabb_min[2])),
                    aabb_max=(float(self._aabb_max[0]), float(self._aabb_max[1]), float(self._aabb_max[2])),
                    voxel_count=0,
                    label_id=0,
                )
            )

            # Build convex hull for the open scene (if available)
            self._hull = None
            self._hull_delaunay = None

            # Choose backend for open-ROI:
            backend = getattr(self, "HULL_BACKEND", "kdop")

            if backend in ("kdop", "auto"):
                # Always build the k-DOP polytope; used as primary (kdop) or fallback (auto).
                self._build_open_polytope()

            if backend in ("scipy", "auto") and _HAS_SCIPY:
                # (Optional) GPU extreme-point prefilter -> then SciPy on survivors
                try:
                    # 1) Extreme-point reduction on GPU
                    dirs = fibonacci_sphere(self.EXTREME_FILTER_K, self.device, self.dtype)
                    proj = self._xyz @ dirs.t()  # (N,K)
                    pmin, imin = proj.min(dim=0)
                    pmax, imax = proj.max(dim=0)
                    keep = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=self.device)
                    keep[imin] = True
                    keep[imax] = True

                    # small tolerance band around extremes
                    eps_tol = 1e-6
                    band = ((proj <= (pmin + eps_tol * self._diag)) | (proj >= (pmax - eps_tol * self._diag))).any(dim=1)
                    keep |= band
                    pts_small = self._xyz[keep].detach().cpu().numpy()

                    if pts_small.shape[0] >= 4 and pts_small.shape[0] <= self.MAX_QHULL_POINTS:
                        from scipy.spatial import ConvexHull, Delaunay
                        hull = ConvexHull(pts_small)
                        self._hull = hull
                        self._hull_delaunay = Delaunay(pts_small[hull.vertices])
                        print(
                            f"Open-ROI hull (SciPy w/ GPU filter): verts={len(hull.vertices)}, "
                            f"vol≈{float(hull.volume):.6g}")
                    else:
                        if backend == "scipy":
                            print("[WARN] Skipping SciPy hull (too many/few points). Using k-DOP instead.")
                except Exception as e:
                    if backend == "scipy":
                        print(f"[WARN] SciPy hull failed ({e}). Using k-DOP instead.")
            else:
                if backend == "scipy":
                    print("[WARN] SciPy unavailable. Using k-DOP instead.")

        # Don't calculate hull if found closed ROIs
        else:
            self._hull = None
            self._hull_delaunay = None

        # Distance-to-boundary via BFS on interior boundary
        dist = np.full((nx, ny, nz), fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        from collections import deque

        dq = deque()
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if not interior[x, y, z]:
                        continue
                    is_boundary = False
                    for xx, yy, zz in (
                            (x + 1, y, z),
                            (x - 1, y, z),
                            (x, y + 1, z),
                            (x, y - 1, z),
                            (x, y, z + 1),
                            (x, y, z - 1),
                    ):
                        if (
                                xx < 0
                                or xx >= nx
                                or yy < 0
                                or yy >= ny
                                or zz < 0
                                or zz >= nz
                                or (not interior[xx, yy, zz])
                        ):
                            is_boundary = True
                            break
                    if is_boundary:
                        dist[x, y, z] = 0
                        dq.append((x, y, z))
        while dq:
            x, y, z = dq.popleft()
            d0 = dist[x, y, z]
            for xx, yy, zz in (
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y + 1, z),
                    (x, y - 1, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
            ):
                if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                    if interior[xx, yy, zz] and dist[xx, yy, zz] > d0 + 1:
                        dist[xx, yy, zz] = d0 + 1
                        dq.append((xx, yy, zz))

        self._grid = {
            "dims": (nx, ny, nz),
            "interior": interior,
            "labels": labels,
            "dist": dist,
        }

        # invalidate inward-fade cache when grid changes
        self._inward_cache = None
        return rois

    # ---------------------- Grid / field convenience methods -------------------

    def _inside_open_hull(self, p: torch.Tensor) -> bool:
        if self._hull_delaunay is None:
            # SciPy missing or failed — fall back to AABB-clip
            return bool(torch.all(p >= self._aabb_min) and torch.all(p <= self._aabb_max))
        x = p.detach().cpu().numpy()[None, :]
        return bool(self._hull_delaunay.find_simplex(x) >= 0)

    def _inside_roi(self, p: torch.Tensor, roi: ROI) -> bool:
        if roi.kind == "open":
            return self._inside_open_polytope(p)
        ix_t, iy_t, iz_t = world_to_index(p.view(1, 3), self._aabb_min, self._aabb_max, self._grid["dims"])
        ix = int(ix_t.item())
        iy = int(iy_t.item())
        iz = int(iz_t.item())
        nx, ny, nz = self._grid["dims"]
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        iz = max(0, min(nz - 1, iz))
        interior = self._grid["interior"][ix, iy, iz]
        label_ok = self._grid["labels"][ix, iy, iz] == roi.label_id
        return bool(interior and label_ok)

    def _dist_to_boundary_world(self, p: torch.Tensor) -> float:
        ix_t, iy_t, iz_t = world_to_index(p.view(1, 3), self._aabb_min, self._aabb_max, self._grid["dims"])
        ix = int(ix_t.item())
        iy = int(iy_t.item())
        iz = int(iz_t.item())
        nx, ny, nz = self._grid["dims"]
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        iz = max(0, min(nz - 1, iz))
        d_vox = self._grid["dist"][ix, iy, iz]
        vsz = voxel_size_world(self._aabb_min, self._aabb_max, self._grid["dims"]).detach().cpu().numpy()
        return float(d_vox * float(np.mean(vsz)))

    def _grad_dist_vox(self, ix: int, iy: int, iz: int) -> np.ndarray:
        nx, ny, nz = self._grid["dims"]
        D = self._grid["dist"]

        def get(x: int, y: int, z: int) -> float:
            x = max(0, min(nx - 1, x))
            y = max(0, min(ny - 1, y))
            z = max(0, min(nz - 1, z))
            return float(D[x, y, z])

        gx = 0.5 * (get(ix + 1, iy, iz) - get(ix - 1, iy, iz))
        gy = 0.5 * (get(ix, iy + 1, iz) - get(ix, iy - 1, iz))
        gz = 0.5 * (get(ix, iy, iz + 1) - get(ix, iy, iz - 1))
        return np.array([gx, gy, gz], dtype=np.float32)

    def _grad_dist_world(self, p: torch.Tensor) -> torch.Tensor:
        ix_t, iy_t, iz_t = world_to_index(p.view(1, 3), self._aabb_min, self._aabb_max, self._grid["dims"])
        ix = int(ix_t.item())
        iy = int(iy_t.item())
        iz = int(iz_t.item())
        g = self._grad_dist_vox(ix, iy, iz)
        vsz = voxel_size_world(self._aabb_min, self._aabb_max, self._grid["dims"]).detach().cpu().numpy()
        g_world = g * np.array(
            [1.0 / max(vsz[0], 1e-9), 1.0 / max(vsz[1], 1e-9), 1.0 / max(vsz[2], 1e-9)],
            dtype=np.float32,
        )
        g_t = torch.tensor(g_world, device=self.device, dtype=self.dtype)
        n = torch.linalg.norm(g_t) + 1e-9
        return g_t / n

    def _min_dist_to_set(self, pts: torch.Tensor, ref: torch.Tensor, cols_per_chunk: int | None = None) -> torch.Tensor:
        """
        Return per-point min distance from pts (M,3) to the reference set ref (N,3),
        evaluating in column chunks to avoid allocating an (M×N) matrix.
        """
        if cols_per_chunk is None:
            cols_per_chunk = self.CDIST_CHUNK_COLS
        M = pts.shape[0]
        best = torch.full((M,), float("inf"), device=pts.device, dtype=pts.dtype)
        for s in range(0, ref.shape[0], cols_per_chunk):
            r = ref[s:s + cols_per_chunk]  # (C,3)
            # distances for a manageable block: (M,C)
            d = torch.cdist(pts, r, p=2.0)
            best = torch.minimum(best, d.min(dim=1).values)
        return best

    def _nearest_gaussian_distance(self, p: torch.Tensor) -> float:
        d = torch.cdist(p.view(1, 3), self._xyz, p=2.0).min().item()
        return float(d)

    def _min_clearance_world(self, sigma_mult: Optional[float] = None) -> float:
        # Combine scale-aware and scene-scale-aware clearances
        diag_term = float(self.MIN_EYE_TO_GAUSS_FRAC * float(self._diag))
        sigma_term = float((self.MIN_CLEAR_SIGMA_MULT if sigma_mult is None else sigma_mult) * self._sigma_med_world)
        return max(diag_term, sigma_term)

    def _min_clearance_for_roi(self, roi, relax_scale: float = 1.0, sigma_mult: float | None = None) -> float:
        """ROI-aware version of min_abs.
        Uses ROI bbox diagonal (not world diag), plus a sigma term, both scaled by relax_scale.
        """
        roi_diag = float(CameraGenerator.__roi_diag_len(roi))
        base_diag_term = self.MIN_EYE_TO_GAUSS_FRAC * roi_diag
        sm = self.MIN_CLEAR_SIGMA_MULT if sigma_mult is None else sigma_mult
        base_sigma_term = sm * float(self._sigma_med_world)
        return float(relax_scale * max(base_diag_term, base_sigma_term))

    # --------------------------- Capacity estimation ---------------------------

    def _estimate_capacity(self, voxel_vol: float) -> Tuple[int, int]:
        """
        Estimate a *realistic* capacity that is consistent with the eye-placement sampler:

          • For each ROI we estimate the effective interior volume V_eff that actually passes
            the min-clearance gate (distance to visible gaussians ≥ min_abs for the ROI).
          • We then bound the number of mutually well-spaced eye sites using a simple packing
            upper bound at the same Poisson radius you will use in sampling:
                N_pack ≈ 1.414 * V_eff / r^3, where r = EYE_POISSON_RADIUS_FRAC * roi_diag.
          • Finally, we multiply by an estimate for “looks per eye” to get a camera capacity band.

        Returns:
            (cap_min, cap_max) as integers.
        """
        nx, ny, nz = self._grid["dims"]
        interior = self._grid["interior"]
        labels = self._grid["labels"]
        D = self._grid["dist"]

        total_min = 0
        total_max = 0
        eps = 1e-6

        for roi in self._rois:
            # ---------------- Open scene: rough packing on its AABB ----------------
            if roi.kind == "open":
                rio_diag = max(CameraGenerator.__roi_diag_len(roi), float(self._diag))
                r = max(eps, float(self.EYE_POISSON_RADIUS_FRAC * rio_diag))
                a = np.asarray(roi.aabb_min, dtype=np.float32)
                b = np.asarray(roi.aabb_max, dtype=np.float32)
                V = float(np.prod(b - a))  # world units^3
                # Packing upper bound (see docstring)
                N_pack = int(max(1, (1.414 * V) / (r ** 3)))

                # Looks per eye: keep conservative to avoid over-promising
                looks_min = max(6, int(0.50 * self.FIB_SAMPLES_PER_EYE))
                looks_max = max(looks_min + 4, int(0.85 * self.FIB_SAMPLES_PER_EYE))

                total_min += int(0.60 * N_pack) * looks_min  # conservative
                total_max += int(1.10 * N_pack) * looks_max  # slightly generous
                continue

            # ---------------- Closed scene: interior mask + clearance fraction ----
            label_id = roi.label_id
            coords = np.argwhere((interior) & (labels == label_id))  # (K,3) in index space
            if coords.size == 0:
                continue

            # Interior volume (world units^3)
            V_int = float(coords.shape[0]) * float(voxel_vol)

            # Clearance fraction: sample at most 4000 interior voxels, measure fraction passing min_abs
            sel = np.random.choice(coords.shape[0], size=min(4000, coords.shape[0]), replace=False)
            sampled = coords[sel]
            wpos = np.stack(
                [index_to_world(ix, iy, iz, self._aabb_min, self._aabb_max, (nx, ny, nz)).astype(np.float32)
                 for (ix, iy, iz) in sampled],
                axis=0,
            )
            wpos_t = torch.tensor(wpos, device=self.device, dtype=self.dtype)
            min_abs_roi = self._min_clearance_for_roi(roi, relax_scale=1.0)
            d = self._min_dist_to_set(wpos_t, self._xyz_vis).detach().cpu().numpy()
            frac_far = float((d >= float(min_abs_roi)).mean()) if d.size > 0 else 0.25

            # Effective interior that survives clearance
            V_eff = max(0.0, V_int * frac_far)

            # Packing bound at the Poisson radius we'll use in sampling
            rio_diag = max(CameraGenerator.__roi_diag_len(roi), eps)
            r = max(eps, float(self.EYE_POISSON_RADIUS_FRAC * rio_diag))
            N_pack = int(max(1, (1.414 * V_eff) / (r ** 3)))

            # Looks per eye: scale a bit with mean distance-to-boundary (headroom)
            Dvals = D[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
            mean_d = float(np.mean(Dvals)) if Dvals.size > 0 else 1.0
            looks_min = max(6, int(0.50 * self.FIB_SAMPLES_PER_EYE))
            looks_max = max(looks_min + 4, int(0.85 * self.FIB_SAMPLES_PER_EYE + 0.5 * mean_d))

            total_min += int(0.80 * N_pack) * looks_min  # conservative
            total_max += int(1.20 * N_pack) * looks_max  # slightly generous

        total_min = int(max(0, min(total_min, 500_000)))
        total_max = int(max(total_min, min(total_max, 1_000_000)))
        return total_min, total_max

    # ------------------------------ Coverage check -----------------------------

    def _estimate_coverage(self, cams: List[Camera]) -> Tuple[float, float]:
        """
        Composite coverage = geometric mean of three pieces:
          1) Pixel coverage over Gaussians (k-sigma ellipse in frame).
          2) Directional-bin coverage per target (fraction of *viable* view bins hit).
          3) Distance-bin coverage per target (fraction of *viable* distance bins hit).

        Returns:
          (composite_unweighted, composite_weighted)
        Also populates self._last_cov_breakdown with the three legs + composite.
        """
        if not cams:
            self._last_cov_breakdown = {
                "pixel_unweighted": 0.0, "pixel_weighted": 0.0,
                "dir_unweighted": 0.0, "dir_weighted": 0.0,
                "dist_unweighted": 0.0, "dist_weighted": 0.0,
                "composite_unweighted": 0.0, "composite_weighted": 0.0,
            }
            return 0.0, 0.0

        g = self._model
        dev, dt = self.device, self.dtype
        eps = 1e-8

        # ========= A) Pixel coverage over ALL Gaussians =========
        N = g.means.shape[0]
        alpha_full = torch.sigmoid(g.opacity_logits.view(-1)).to(device=dev, dtype=dt)
        alpha_sum_full = float(alpha_full.sum().item()) + eps
        covered_any_full = torch.zeros((N,), dtype=torch.bool, device=dev)

        ksig = float(self.COV_K_SIGMA)
        for cam in cams:
            proj = projected_covariance_2d(
                means3d=g.means, scales=g.scales, rotations=g.rotations_wxyz,
                camera=cam, min_sigma_px=self.MIN_SIGMA_PX, max_sigma_px=None,
            )
            uc, vc, Z = proj["uc"], proj["vc"], proj["Z"]
            sig = proj["sigmas"]
            in_front = Z > 0.0
            left_ok = (uc + ksig * sig[:, 0]) >= 0.0
            right_ok = (uc - ksig * sig[:, 0]) <= (cam.width - 1.0)
            top_ok = (vc + ksig * sig[:, 1]) >= 0.0
            bottom_ok = (vc - ksig * sig[:, 1]) <= (cam.height - 1.0)
            in_img = in_front & left_ok & right_ok & top_ok & bottom_ok
            covered_any_full |= in_img

        pix_unweighted = float(covered_any_full.float().mean().item())
        pix_weighted = float((alpha_full * covered_any_full.float()).sum().item() / alpha_sum_full)

        # ========= B) Multi-view coverage over TARGETS (direction & distance bins) =========
        targets = self._make_targets()  # (T,3)
        T = int(targets.shape[0])
        if T == 0:
            self._last_cov_breakdown = {
                "pixel_unweighted": pix_unweighted, "pixel_weighted": pix_weighted,
                "dir_unweighted": 0.0, "dir_weighted": 0.0,
                "dist_unweighted": 0.0, "dist_weighted": 0.0,
                "composite_unweighted": pix_unweighted, "composite_weighted": pix_weighted,
            }
            return pix_unweighted, pix_weighted

        # Per-target ROI info
        def _roi_for_point(p: torch.Tensor) -> int:
            for i, r in enumerate(self._rois):
                if self._inside_roi(p, r):
                    return i
            return 0

        roi_index = torch.empty((T,), dtype=torch.long, device=dev)
        roi_diag = torch.empty((T,), dtype=dt, device=dev)
        min_abs_t = torch.empty((T,), dtype=dt, device=dev)
        for i in range(T):
            ri = _roi_for_point(targets[i])
            roi_index[i] = int(ri)
            r = self._rois[int(ri)]
            a = torch.tensor(r.aabb_min, device=dev, dtype=dt)
            b = torch.tensor(r.aabb_max, device=dev, dtype=dt)
            roi_diag[i] = torch.linalg.norm(b - a).clamp_min(1e-8)
            min_abs_t[i] = float(self._min_clearance_for_roi(r, relax_scale=1.0))

        # Assign each Gaussian to nearest target (chunked)
        means = g.means.to(device=dev, dtype=dt)
        t_assign = torch.empty((N,), dtype=torch.long, device=dev)
        CHUNK = 8192
        for s in range(0, N, CHUNK):
            dist = torch.cdist(means[s:s + CHUNK], targets, p=2.0)
            t_assign[s:s + CHUNK] = torch.argmin(dist, dim=1)

        # Opacity mass per target
        t_weights = torch.zeros((T,), dtype=dt, device=dev)
        t_weights.index_add_(0, t_assign, torch.sigmoid(g.opacity_logits.view(-1)).to(device=dev, dtype=dt))

        # Direction + distance bins
        B_dir = int(self.DIR_BIN_COUNT)
        dir_bins = fibonacci_sphere(B_dir, dev, dt)  # (B,3)
        edges_frac = torch.tensor(list(self.DIST_BIN_EDGES_FRAC), device=dev, dtype=dt)
        K_dst = int(edges_frac.numel() + 1)

        dir_viable = torch.zeros((T, B_dir), dtype=torch.bool, device=dev)
        dist_viable = torch.zeros((T, K_dst), dtype=torch.bool, device=dev)

        def _feasible_eye(ti: int, v_dir: torch.Tensor) -> Optional[torch.Tensor]:
            roi = self._rois[int(roi_index[ti].item())]
            t = targets[ti]
            r_init = float(roi_diag[ti].item())
            return self._find_feasible_eye_on_ray(
                anchor=t, v=v_dir, r_init=r_init, roi=roi,
                min_abs=float(min_abs_t[ti].item()),
                inward_grad=None, inward_thresh=float(self.INWARD_DOT_THRESH),
            )

        with torch.no_grad():
            for ti in range(T):
                rmax = 0.0
                for bj in range(B_dir):
                    v = dir_bins[bj] / (torch.linalg.norm(dir_bins[bj]) + 1e-9)
                    eye = _feasible_eye(ti, v)
                    if eye is None:
                        continue
                    dir_viable[ti, bj] = True
                    rmax = max(rmax, float(torch.linalg.norm(eye - targets[ti]).item()))
                if rmax > 0.0:
                    r_min = float(min_abs_t[ti].item())
                    eabs = (edges_frac * roi_diag[ti]).detach().cpu().numpy().tolist()
                    bounds = []
                    if len(eabs) == 0:
                        bounds = [(0.0, float("inf"))]
                    else:
                        bounds = [(0.0, eabs[0])]
                        for ii in range(len(eabs) - 1):
                            bounds.append((eabs[ii], eabs[ii + 1]))
                        bounds.append((eabs[-1], float("inf")))
                    for kb, (lo, hi) in enumerate(bounds):
                        if (rmax > lo) and (r_min < hi):
                            dist_viable[ti, kb] = True

        # Observed bins from accepted cameras
        obs_dir = torch.zeros((T, B_dir), dtype=torch.bool, device=dev)
        obs_dst = torch.zeros((T, K_dst), dtype=torch.bool, device=dev)

        def _cam_eye(cam_obj) -> torch.Tensor:
            cc = getattr(cam_obj, "camera_center", None)
            if cc is not None:
                return cc.to(device=dev, dtype=dt)
            c2w = getattr(cam_obj, "camera_to_world", None)
            if c2w is not None:
                c2w_t = torch.as_tensor(c2w, device=dev, dtype=dt).reshape(4, 4)
                return c2w_t[:3, 3]
            w2c = getattr(cam_obj, "world_view_transform", None)
            if w2c is not None:
                w2c_t = torch.as_tensor(w2c, device=dev, dtype=dt).reshape(4, 4)
                return torch.linalg.inv(w2c_t)[:3, 3]
            raise RuntimeError("Camera missing camera_center / camera_to_world / world_view_transform.")

        for cam in cams:
            proj = projected_covariance_2d(
                means3d=g.means, scales=g.scales, rotations=g.rotations_wxyz,
                camera=cam, min_sigma_px=self.MIN_SIGMA_PX, max_sigma_px=None,
            )
            uc, vc, Z = proj["uc"], proj["vc"], proj["Z"]
            sig = proj["sigmas"]
            in_front = Z > 0.0
            left_ok = (uc + ksig * sig[:, 0]) >= 0.0
            right_ok = (uc - ksig * sig[:, 0]) <= (cam.width - 1.0)
            top_ok = (vc + ksig * sig[:, 1]) >= 0.0
            bottom_ok = (vc - ksig * sig[:, 1]) <= (cam.height - 1.0)
            in_img = in_front & left_ok & right_ok & top_ok & bottom_ok
            if not bool(in_img.any().item()):
                continue

            t_seen = t_assign[in_img]
            if t_seen.numel() == 0:
                continue
            t_unique = torch.unique(t_seen)

            eye = _cam_eye(cam)
            for ti in t_unique.tolist():
                ti = int(ti)
                tpos = targets[ti]
                v = (eye - tpos)
                r = torch.linalg.norm(v) + 1e-9
                vhat = v / r
                # Direction bin
                b = int(torch.argmax(vhat @ dir_bins.T).item())
                obs_dir[ti, b] = True
                # Distance bin (relative to that target's ROI diag)
                eabs = (edges_frac * roi_diag[ti]).detach().cpu().numpy()
                rr = float(r.item())
                if eabs.size == 0:
                    kb = 0
                elif rr < eabs[0]:
                    kb = 0
                elif eabs.size > 1 and rr < eabs[1]:
                    kb = 1
                elif eabs.size > 2 and rr < eabs[2]:
                    kb = 2
                else:
                    kb = int(eabs.size)
                obs_dst[ti, kb] = True

        # ===== Ratios per target; ONLY count observed ∩ viable =====
        dir_viable_counts = dir_viable.sum(dim=1)  # (T,)
        dst_viable_counts = dist_viable.sum(dim=1)  # (T,)

        has_dir = dir_viable_counts > 0
        has_dst = dst_viable_counts > 0

        dir_ratios = torch.zeros((T,), dtype=dt, device=dev)
        dst_ratios = torch.zeros((T,), dtype=dt, device=dev)
        if bool(has_dir.any().item()):
            dir_hits = (obs_dir & dir_viable).sum(dim=1).to(dtype=dt)
            dir_ratios[has_dir] = dir_hits[has_dir] / (dir_viable_counts[has_dir].to(dtype=dt) + eps)
        if bool(has_dst.any().item()):
            dst_hits = (obs_dst & dist_viable).sum(dim=1).to(dtype=dt)
            dst_ratios[has_dst] = dst_hits[has_dst] / (dst_viable_counts[has_dst].to(dtype=dt) + eps)

        # Clamp to [0,1] for numerical safety
        dir_ratios = torch.clamp(dir_ratios, 0.0, 1.0)
        dst_ratios = torch.clamp(dst_ratios, 0.0, 1.0)
        pix_unweighted = float(max(0.0, min(1.0, pix_unweighted)))
        pix_weighted = float(max(0.0, min(1.0, pix_weighted)))

        # Unweighted & opacity-weighted (by per-target mass)
        t_w = t_weights
        wsum_dir = float(t_w[has_dir].sum().item()) + eps
        wsum_dst = float(t_w[has_dst].sum().item()) + eps

        dir_unweighted = float(dir_ratios[has_dir].mean().item()) if bool(has_dir.any().item()) else 0.0
        dir_weighted = float((dir_ratios[has_dir] * t_w[has_dir].to(dtype=dt)).sum().item() / wsum_dir) if bool(
            has_dir.any().item()) else 0.0

        dst_unweighted = float(dst_ratios[has_dst].mean().item()) if bool(has_dst.any().item()) else 0.0
        dst_weighted = float((dst_ratios[has_dst] * t_w[has_dst].to(dtype=dt)).sum().item() / wsum_dst) if bool(
            has_dst.any().item()) else 0.0

        # ========= C) Composite via geometric mean (≤ 1 by construction) =========
        comp_unw = (max(pix_unweighted, eps) * max(dir_unweighted, eps) * max(dst_unweighted, eps)) ** (1.0 / 3.0)
        comp_w = (max(pix_weighted, eps) * max(dir_weighted, eps) * max(dst_weighted, eps)) ** (1.0 / 3.0)

        # Final safety clamp (just in case of float noise)
        comp_unw = float(min(1.0, max(0.0, comp_unw)))
        comp_w = float(min(1.0, max(0.0, comp_w)))

        # Optional debug
        print_debug(
            f"[COVERAGE] pix(u/w)={pix_unweighted * 100:.1f}/{pix_weighted * 100:.1f}  "
            f"dir(u/w)={dir_unweighted * 100:.1f}/{dir_weighted * 100:.1f}  "
            f"dist(u/w)={dst_unweighted * 100:.1f}/{dst_weighted * 100:.1f}  "
            f"=> composite(u/w)={comp_unw * 100:.1f}/{comp_w * 100:.1f}"
        )

        self._last_cov_breakdown = {
            "pixel_unweighted": pix_unweighted, "pixel_weighted": pix_weighted,
            "dir_unweighted": dir_unweighted, "dir_weighted": dir_weighted,
            "dist_unweighted": dst_unweighted, "dist_weighted": dst_weighted,
            "composite_unweighted": comp_unw,
            "composite_weighted": comp_w,
            "config": {
                "DIR_BIN_COUNT": int(self.DIR_BIN_COUNT),
                "DIST_BIN_EDGES_FRAC": list(self.DIST_BIN_EDGES_FRAC),
                "MIN_SIGMA_PX": float(self.MIN_SIGMA_PX),
                "COV_K_SIGMA": float(self.COV_K_SIGMA),
            },
        }
        return float(comp_unw), float(comp_w)

    # ------------------------ Feasible eye search per ray ----------------------

    def _find_feasible_eye_on_ray(
            self,
            anchor: torch.Tensor,
            v: torch.Tensor,
            r_init: float,
            roi: ROI,
            min_abs: float,
            inward_grad: Optional[torch.Tensor],
            inward_thresh: float,
    ) -> Optional[torch.Tensor]:
        r = r_init
        for _ in range(16):
            eye = anchor + r * v
            if not self._inside_roi(eye, roi):
                r *= 0.85
                continue
            if self._nearest_gaussian_distance(eye) < min_abs:
                r *= 0.85
                continue
            if inward_grad is not None:
                look_dir = anchor - eye
                look_dir = look_dir / (torch.linalg.norm(look_dir) + 1e-9)
                if float(torch.dot(look_dir, inward_grad)) < inward_thresh:
                    r *= 0.85
                    continue
            return eye
        return None

    def _max_open_polytope_radius(self, anchor: torch.Tensor, v: torch.Tensor, r_limit: float) -> float:
        """Clamp the ray length to remain inside the open-scene k-DOP envelope."""
        if not hasattr(self, "_kdop_dirs"):
            return r_limit

        dirs = self._kdop_dirs
        bmin = self._kdop_bmin
        bmax = self._kdop_bmax
        anchor_proj = torch.mv(dirs, anchor)
        dir_proj = torch.mv(dirs, v)

        eps = 1e-9
        mask_zero = dir_proj.abs() < eps
        if mask_zero.any():
            inside = (anchor_proj[mask_zero] >= bmin[mask_zero]) & (anchor_proj[mask_zero] <= bmax[mask_zero])
            if not bool(inside.all()):
                return 0.0

        r_entry = torch.full_like(anchor_proj, -float("inf"))
        r_exit = torch.full_like(anchor_proj, float("inf"))

        mask = ~mask_zero
        if mask.any():
            denom = dir_proj[mask]
            t1 = (bmin[mask] - anchor_proj[mask]) / denom
            t2 = (bmax[mask] - anchor_proj[mask]) / denom
            r_entry[mask] = torch.minimum(t1, t2)
            r_exit[mask] = torch.maximum(t1, t2)

        lo = float(r_entry.max().item())
        hi = float(r_exit.min().item())

        if hi <= 0.0 or lo > hi:
            return 0.0

        entry_pos = max(0.0, lo)
        if entry_pos > hi:
            return 0.0

        r_hi = min(r_limit, hi)
        if r_hi <= 0.0:
            return 0.0

        return r_hi

    # --------------------------- Eye sampling helpers ------------------------

    def _poisson_sample_points(
            self, pts: torch.Tensor, radius: float, limit: Optional[int] = None
    ) -> List[torch.Tensor]:
        if pts.shape[0] == 0:
            return []
        if radius <= 0.0:
            if limit is None:
                return [pts[i] for i in range(pts.shape[0])]
            return [pts[i] for i in range(min(limit, pts.shape[0]))]

        order = torch.randperm(pts.shape[0], device=pts.device)
        selected: List[torch.Tensor] = []
        r2 = radius * radius
        for idx in order.tolist():
            p = pts[idx]
            if not selected:
                selected.append(p)
            else:
                stacked = torch.stack(selected)
                d2 = torch.sum((stacked - p) ** 2, dim=1)
                if torch.all(d2 >= r2):
                    selected.append(p)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def _sample_open_eye_positions(self, roi: ROI, n: int, min_abs: float) -> List[torch.Tensor]:
        if n <= 0:
            return []
        dev, dt = self.device, self.dtype
        roi_diag = float(np.linalg.norm(np.asarray(roi.aabb_max) - np.asarray(roi.aabb_min)))
        radius = float(self.EYE_POISSON_RADIUS_FRAC * roi_diag)
        shell_pts: List[torch.Tensor] = []
        volume_pts: List[torch.Tensor] = []
        center = torch.tensor(roi.center, device=dev, dtype=dt)
        attempts = max(400, n * 40)
        diag = float(self._diag)
        for _ in range(attempts):
            direction = torch.randn(3, device=dev, dtype=dt)
            direction = direction / (torch.linalg.norm(direction) + 1e-9)
            r_limit = diag
            r_max = self._max_open_polytope_radius(center, direction, r_limit)
            if r_max <= min_abs * 1.1:
                continue
            shell_r = max(min_abs * 1.1, r_max * 0.9)
            shell = center + shell_r * direction
            if self._inside_roi(shell, roi) and self._nearest_gaussian_distance(shell) >= min_abs:
                shell_pts.append(shell)
            t = torch.rand(1, device=dev, dtype=dt).item()
            vol_r = max(min_abs * 1.1, r_max * t)
            volume = center + vol_r * direction
            if self._inside_roi(volume, roi) and self._nearest_gaussian_distance(volume) >= min_abs:
                volume_pts.append(volume)
            if len(shell_pts) >= 6 * n and len(volume_pts) >= 6 * n:
                break

        candidates: List[torch.Tensor] = []
        if shell_pts:
            shell_tensor = torch.stack(shell_pts)
            candidates.extend(self._poisson_sample_points(shell_tensor, radius, max(1, n // 2)))
        if volume_pts:
            volume_tensor = torch.stack(volume_pts)
            candidates.extend(self._poisson_sample_points(volume_tensor, radius, n))

        if len(candidates) < n:
            # fallback: random interior points filtered by clearance
            aabb_min = self._aabb_min
            aabb_max = self._aabb_max
            extra = []
            for _ in range(max(800, n * 60)):
                p = aabb_min + torch.rand(3, device=dev, dtype=dt) * (aabb_max - aabb_min)
                if not self._inside_roi(p, roi):
                    continue
                if self._nearest_gaussian_distance(p) < min_abs:
                    continue
                extra.append(p)
                if len(extra) >= n * 4:
                    break
            if extra:
                extra_tensor = torch.stack(extra)
                candidates.extend(self._poisson_sample_points(extra_tensor, radius, n))

        return candidates[:n]

    def _sample_closed_eye_positions(self, roi: ROI, n: int, min_abs: float) -> List[torch.Tensor]:
        """
        Sample eye positions inside a CLOSED ROI.

        Changes vs. previous version:
          • Use ONE global Poisson selection over (shell ∪ core), so spacing is enforced
            across all eyes, not separately in two pools.
          • If the request n is infeasible at the current Poisson radius (by packing bound),
            adaptively shrink the radius just enough to make n feasible (bounded to 25% of
            the original radius).
            - If prefer strict spacing, remove the shrink and cap n:
                # Cap target to feasible bound instead of shrinking the radius
                n = min(n, N_pack)
        """
        dev, dt = self.device, self.dtype
        if n <= 0:
            return []

        nx, ny, nz = self._grid["dims"]
        interior = self._grid["interior"]
        labels = self._grid["labels"]
        D = self._grid["dist"]

        # Index-space coords of this ROI's interior
        coords = np.argwhere((interior) & (labels == roi.label_id))
        if coords.size == 0:
            return []

        # Speed cap
        rng = np.random.default_rng(self.seed or 0)
        if coords.shape[0] > 8000:
            sel = rng.choice(coords.shape[0], size=8000, replace=False)
            coords = coords[sel]

        # Shell/Core split by distance-to-boundary quantile (for diversity biasing)
        dvals = D[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
        shell_thresh = np.quantile(dvals, 0.35)
        shell_mask = dvals <= shell_thresh
        core_mask = ~shell_mask

        # Convert to world coords
        pts_world = [
            index_to_world(ix, iy, iz, self._aabb_min, self._aabb_max, (nx, ny, nz)).astype(np.float32)
            for ix, iy, iz in coords
        ]
        pts_tensor = torch.tensor(np.asarray(pts_world), device=dev, dtype=dt)

        # Clearance filter (distance to visible gaussians >= min_abs)
        d_clear = self._min_dist_to_set(pts_tensor, self._xyz_vis)
        keep = d_clear >= float(min_abs)
        pts_tensor = pts_tensor[keep]
        mask_keep = keep.detach().cpu().numpy()

        if pts_tensor.shape[0] == 0:
            return []

        shell_tensor = pts_tensor[torch.tensor(shell_mask[mask_keep], device=dev, dtype=torch.bool)]
        core_tensor = pts_tensor[torch.tensor(core_mask[mask_keep], device=dev, dtype=torch.bool)]

        # Poisson radius tied to ROI diagonal (single-source)
        roi_diag = float(CameraGenerator.__roi_diag_len(roi))
        pois_r = float(self.EYE_POISSON_RADIUS_FRAC * max(roi_diag, 1e-6))

        # Estimate feasibility (packing bound) at current radius using the *kept* volume
        vx = voxel_size_world(self._aabb_min, self._aabb_max, (nx, ny, nz)).detach().cpu().numpy()
        voxel_vol = float(vx[0] * vx[1] * vx[2])
        V_eff = float(pts_tensor.shape[0]) * voxel_vol

        def packing_bound(V: float, r: float) -> int:
            r = max(r, 1e-6)
            return int(max(1, (1.414 * V) / (r ** 3)))

        N_pack = packing_bound(V_eff, pois_r)

        # If the request n exceeds the bound, shrink r just enough to make it feasible
        if n > N_pack:
            scale = (max(1, N_pack) / float(n)) ** (1.0 / 3.0)  # r_new = r_old * (N_pack/n)^(1/3)
            pois_r = max(0.25 * pois_r, float(pois_r * scale))

        # Global candidate pool (put shell first to add a slight bias toward near-walls)
        all_pts = torch.cat([shell_tensor, core_tensor], dim=0) if core_tensor.numel() > 0 else shell_tensor
        if all_pts.shape[0] == 0:
            return []

        # Global Poisson selection with randomized order
        order = torch.randperm(all_pts.shape[0], device=dev)
        selected: List[torch.Tensor] = []
        r2 = float(pois_r * pois_r)

        for idx in order.tolist():
            p = all_pts[idx]
            if not selected:
                selected.append(p)
            else:
                S = torch.stack(selected)
                if torch.all(torch.sum((S - p) ** 2, dim=1) >= r2):
                    selected.append(p)
            if len(selected) >= n:
                break

        return selected[:n]


    def _sample_eye_positions(self, roi: ROI, n: int, min_abs: float) -> List[torch.Tensor]:
        if roi.kind == "open":
            return self._sample_open_eye_positions(roi, n, min_abs)
        return self._sample_closed_eye_positions(roi, n, min_abs)

    # --------------------------- Target construction --------------------------

    def _make_targets(self) -> torch.Tensor:
        dev, dt = self.device, self.dtype
        if self._xyz_vis.shape[0] == 0:
            return torch.zeros((0, 3), device=dev, dtype=dt)

        k = min(self.TARGET_CLUSTER_COUNT, int(self._xyz_vis.shape[0]))
        idxs: List[int] = []
        first = int(torch.randint(0, self._xyz_vis.shape[0], (1,), device=dev).item())
        idxs.append(first)
        d2 = torch.sum((self._xyz_vis - self._xyz_vis[first]) ** 2, dim=1)
        while len(idxs) < k:
            next_idx = int(torch.argmax(d2).item())
            if next_idx in idxs:
                break
            idxs.append(next_idx)
            d2 = torch.minimum(d2, torch.sum((self._xyz_vis - self._xyz_vis[next_idx]) ** 2, dim=1))
        cluster_pts = self._xyz_vis[idxs]

        extras: List[torch.Tensor] = [cluster_pts]
        for roi in self._rois:
            a = torch.tensor(roi.aabb_min, device=dev, dtype=dt)
            b = torch.tensor(roi.aabb_max, device=dev, dtype=dt)
            corners = torch.stack(
                [
                    torch.tensor([x, y, z], device=dev, dtype=dt)
                    for x in (a[0], b[0])
                    for y in (a[1], b[1])
                    for z in (a[2], b[2])
                ],
                dim=0,
            )
            center = torch.tensor(roi.center, device=dev, dtype=dt)
            face_centers = torch.stack(
                [
                    torch.tensor([center[0], center[1], a[2]], device=dev, dtype=dt),
                    torch.tensor([center[0], center[1], b[2]], device=dev, dtype=dt),
                    torch.tensor([center[0], a[1], center[2]], device=dev, dtype=dt),
                    torch.tensor([center[0], b[1], center[2]], device=dev, dtype=dt),
                    torch.tensor([a[0], center[1], center[2]], device=dev, dtype=dt),
                    torch.tensor([b[0], center[1], center[2]], device=dev, dtype=dt),
                ],
                dim=0,
            )
            extras.extend([corners, face_centers, center.view(1, 3)])

        all_targets = torch.cat(extras, dim=0)
        radius = float(self.TARGET_POISSON_RADIUS_FRAC * float(self._diag))
        sampled = self._poisson_sample_points(all_targets, radius)
        if len(sampled) == 0:
            return all_targets.unique(dim=0)
        stacked = torch.stack(sampled)
        return stacked

    # ----------------------- Visibility & coverage helpers ---------------------

    def _visibility_mask(
            self,
            eye: torch.Tensor,
            forward: torch.Tensor,
            fovx: float,
            fovy: float,
            near: float,
            far: float,
    ) -> Tuple[torch.Tensor, int]:
        fwd = forward / (torch.linalg.norm(forward) + 1e-9)
        V = self._xyz_vis - eye
        dist = torch.linalg.norm(V, dim=1)
        Vn = V / (dist.unsqueeze(-1) + 1e-9)
        max_half = max(fovx, fovy) * 0.5 + math.radians(self.VIS_FOV_MARGIN_DEG)
        min_cos = math.cos(max_half)
        cosang = (Vn * fwd).sum(dim=1)
        ok = (cosang >= min_cos) & (dist >= near * 1.5) & (dist <= far * 0.95)
        hits = int(ok.sum().item())
        return ok, hits

    def _free_solid_frac(self, eye_info: dict, inward_relax: float, probes: int = 12) -> float:
        """
        Estimate the fraction of free solid angle for an eye inside a CLOSED ROI,
        accounting for distance-aware inward gating with fade.
        Returns φ in [0,1]. For OPEN scenes or missing grad -> 1.0.
        """
        roi = eye_info["roi"]
        inward_grad = eye_info.get("inward_grad", None)
        if roi.kind != "closed" or inward_grad is None:
            return 1.0

        # Normalize inward gradient
        g = inward_grad / (torch.linalg.norm(inward_grad) + 1e-9)

        # Distance-aware inward threshold with fade
        t_fade = self._effective_inward_thresh(eye_info["eye"])

        # Respect any relaxation requested by top-up passes (can be negative)
        eff_thresh = min(t_fade, float(inward_relax))

        # Probe random directions on the sphere to estimate free fraction
        dirs = fibonacci_sphere(probes, self.device, self.dtype)
        ok = (dirs @ g.view(3, )) >= eff_thresh
        return float(ok.float().mean().item())

    # ----------------------- Class __init__ Constructor ---------------------

    def __init__(self, ply_path: str, out_json_path: str, seed: Optional[int] = None) -> None:
        self.ply_path = ply_path
        self.out_json_path = out_json_path
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % 2 ** 32)

        self.device = torch_ctx.device
        self.dtype = torch_ctx.dtype

        # Load scene
        self._model = load_gaussian_ply(self.ply_path, max_sh_degree=3, device=self.device)
        self._xyz = self._model.means.detach()  # (N,3)
        self._scales_log = self._model.scales.detach()  # (N,3)  (log)
        self._rots = self._model.rotations_wxyz.detach()  # (N,4)
        self._opacity_logits = self._model.opacity_logits.detach()

        # Subsample for visibility tests
        N_all = self._xyz.shape[0]
        k = min(N_all, self.VIS_SAMPLE_MAX)
        perm = torch.randperm(N_all, device=self.device)
        self._xyz_vis = self._xyz[perm[:k]].contiguous()
        alpha_all = torch.sigmoid(self._opacity_logits.view(-1))
        self._alpha_vis = alpha_all[perm[:k]].to(device=self.device, dtype=self.dtype)

        # Robust global AABB (for coarse diagnostics)
        aabb_min_t, aabb_max_t = robust_aabb(self._xyz, self.LOW_PCT, self.HIGH_PCT, mul_fact=self.MUL_FACT)
        self._aabb_min = aabb_min_t
        self._aabb_max = aabb_max_t
        self._center = 0.5 * (self._aabb_min + self._aabb_max)
        self._diag = torch.linalg.norm(self._aabb_max - self._aabb_min).clamp_min(1e-8)
        self._radius = 0.5 * self._diag
        self._sigma_med_world = float(torch.median(torch.exp(self._scales_log)).item())

        # --- Bounding box diagnostics ---
        print_debug("Scene AABB:")
        print_debug(
            f"  min: [{float(self._aabb_min[0]):.6f}, {float(self._aabb_min[1]):.6f}, {float(self._aabb_min[2]):.6f}]")
        print_debug(
            f"  max: [{float(self._aabb_max[0]):.6f}, {float(self._aabb_max[1]):.6f}, {float(self._aabb_max[2]):.6f}]")
        print_debug(
            f"  center: [{float(self._center[0]):.6f}, {float(self._center[1]):.6f}, {float(self._center[2]):.6f}]")
        print_debug(f"diag: {float(self._diag):.6f}  radius: {float(self._radius):.6f}")

        # Build occupancy grid, discover ROIs, distance field, labels
        self._rois = self._build_rois_and_fields()

        # Print ROI count, volumes, and scene scale
        scene_lengths = (self._aabb_max - self._aabb_min).detach().cpu().numpy()
        scene_vol = float(np.prod(scene_lengths))
        print(f"Scene AABB volume: {scene_vol:.6g} (units^3)")
        print(f"Discovered {len(self._rois)} region(s) of interest")

        # Per-ROI volumes + bbox coordinates (for visualization)
        vx_size = voxel_size_world(self._aabb_min, self._aabb_max, self._grid["dims"]).detach().cpu().numpy()
        voxel_vol = float(vx_size[0] * vx_size[1] * vx_size[2])
        for i, roi in enumerate(self._rois):
            if roi.kind == "closed":
                label_id = roi.label_id
                voxels_in_roi = int((self._grid["labels"] == label_id).sum())
                roi_vol = voxels_in_roi * voxel_vol
                bbox = np.array(roi.aabb_max) - np.array(roi.aabb_min)
                bbox_vol = float(np.prod(bbox))
                print(
                    f"  ROI #{i} [closed]  voxel-volume≈{roi_vol:.6g},  bbox-volume≈{bbox_vol:.6g},  ratio_to_scene≈{roi_vol / scene_vol:.4f}"
                )
                print_debug(
                    f"           ROI #{i} bbox min=({roi.aabb_min[0]:.6f}, {roi.aabb_min[1]:.6f}, {roi.aabb_min[2]:.6f})  max=({roi.aabb_max[0]:.6f}, {roi.aabb_max[1]:.6f}, {roi.aabb_max[2]:.6f})  center=({roi.center[0]:.6f}, {roi.center[1]:.6f}, {roi.center[2]:.6f})"
                )
            else:
                bbox = np.array(roi.aabb_max) - np.array(roi.aabb_min)
                bbox_vol = float(np.prod(bbox))
                print(f"  ROI #{i} [open]    bbox-volume≈{bbox_vol:.6g},  ratio_to_scene≈{bbox_vol / scene_vol:.4f}")
                print_debug(
                    f"           ROI #{i} bbox min=({roi.aabb_min[0]:.6f}, {roi.aabb_min[1]:.6f}, {roi.aabb_min[2]:.6f})  max=({roi.aabb_max[0]:.6f}, {roi.aabb_max[1]:.6f}, {roi.aabb_max[2]:.6f})  center=({roi.center[0]:.6f}, {roi.center[1]:.6f}, {roi.center[2]:.6f})"
                )

        # Estimate meaningful camera capacity [min, max]
        cap_min, cap_max = self._estimate_capacity(voxel_vol)
        self._capacity_est = (cap_min, cap_max)
        print_debug(f"Estimated meaningful camera capacity: [{cap_min}, {cap_max}]")

    # ------------------------------ Main generate ------------------------------

    def generate(self, num_cameras: int, target_coverage: float = 0.99, estimate_coverage: bool = True) -> List[Camera]:
        if num_cameras <= 0:
            raise ValueError("num_cameras must be positive")

        ## New run_once(...)
        # Implements:
        #   Distance-aware inward fade,
        #   Adaptive k_i dirs/eye via _free_solid_frac,
        #   Pre-filter seed directions against the inward gate,
        #   Global candidate budget (hard cap),
        #   ROI-scaled pose-distance penalty,
        def run_once(min_abs: float, inward_thresh: float) -> Tuple[List[Camera], List[Dict[str, Any]], Dict[str, int]]:
            self._reset_dbg()
            dev, dt = self.device, self.dtype
            W, H = int(self.DEFAULT_W), int(self.DEFAULT_H)
            fovx = math.radians(float(self.DEFAULT_FOV_DEG))
            fovy = math.radians(float(self.DEFAULT_FOV_DEG))

            # Relaxation scale
            _world_default = float(self._min_clearance_world())
            relax_scale = float(min_abs) / max(_world_default, 1e-12)
            relax_scale = max(0.05, min(relax_scale, 20.0))

            world_up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dt)
            alt_up = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=dt)

            # ROI diagonals (for ROI-scaled pose-distance & distance bins) — single source
            roi_diags: List[float] = [max(CameraGenerator.__roi_diag_len(r), 1e-8) for r in self._rois]

            # ROI weights (unchanged)
            num_rois = len(self._rois)
            if num_rois == 0:
                return [], [], dict(self._dbg)

            vx_size = voxel_size_world(self._aabb_min, self._aabb_max, self._grid["dims"]).detach().cpu().numpy()
            voxel_vol = float(vx_size[0] * vx_size[1] * vx_size[2])

            closed_indices = [i for i, r in enumerate(self._rois) if r.kind == "closed"]
            open_indices = [i for i, r in enumerate(self._rois) if r.kind == "open"]

            scene_lengths = (self._aabb_max - self._aabb_min).detach().cpu().numpy()
            scene_vol = float(np.prod(scene_lengths))

            roi_weights = [0.0 for _ in self._rois]
            if closed_indices:
                closed_vols: List[Tuple[int, float]] = []
                for idx in closed_indices:
                    roi = self._rois[idx]
                    voxels_in_roi = int((self._grid["labels"] == roi.label_id).sum())
                    closed_vols.append((idx, voxels_in_roi * voxel_vol))
                open_bonus = 0.1 * scene_vol if open_indices else 0.0
                total = sum(v for _, v in closed_vols) + open_bonus
                if total <= 0.0:
                    roi_weights = [1.0 / num_rois for _ in self._rois]
                else:
                    for idx, vol in closed_vols:
                        roi_weights[idx] = max(1e-9, vol / total)
                    if open_indices:
                        per_open = (open_bonus / total) / max(len(open_indices), 1)
                        for idx in open_indices:
                            roi_weights[idx] = max(1e-9, per_open)
            else:
                roi_weights = [1.0 / num_rois for _ in self._rois]

            # Allocate eyes per ROI
            total_eye_slots = max(num_cameras * self.EYE_OVERSAMPLE_FACTOR, num_cameras)
            eyes_per_roi = [max(1, int(round(total_eye_slots * w))) for w in roi_weights]
            total_assigned = sum(eyes_per_roi)
            while total_assigned > total_eye_slots and any(v > 1 for v in eyes_per_roi):
                i = int(np.argmax(np.array(eyes_per_roi)))
                if eyes_per_roi[i] > 1:
                    eyes_per_roi[i] -= 1
                    total_assigned -= 1
            while total_assigned < total_eye_slots and len(eyes_per_roi) > 0:
                i = int(np.argmin(np.array(eyes_per_roi)))
                eyes_per_roi[i] += 1
                total_assigned += 1

            # Sample eyes (tag with eye_id) + inward grads
            eye_infos: List[Dict[str, Any]] = []
            eye_id_counter = 0
            for r_idx, roi in enumerate(self._rois):
                target_eyes = eyes_per_roi[r_idx] if r_idx < len(eyes_per_roi) else 1
                min_abs_roi = self._min_clearance_for_roi(roi, relax_scale=relax_scale)
                sampled = self._sample_eye_positions(roi, target_eyes, min_abs_roi)
                print(f"[Eyes] ROI={r_idx} kind={roi.kind} requested={target_eyes} sampled={len(sampled)}")
                self._dbg["eyes_total"] += len(sampled)
                for eye in sampled:
                    if self._nearest_gaussian_distance(eye) < min_abs_roi:
                        self._dbg["rejected_too_close_gauss"] += 1
                        continue
                    if not self._inside_roi(eye, roi):
                        self._dbg["rejected_outside_roi"] += 1
                        continue
                    inward_grad = self._grad_dist_world(eye) if roi.kind == "closed" else None
                    eye_infos.append(
                        {
                            "eye": eye,
                            "eye_id": int(eye_id_counter),
                            "roi": roi,
                            "roi_index": r_idx,
                            "used_dirs": [],
                            "used_targets": set(),
                            "inward_grad": inward_grad,
                            "min_abs": float(min_abs_roi),
                        }
                    )
                    eye_id_counter += 1

            if len(eye_infos) == 0:
                return [], [], dict(self._dbg)

            # Targets
            targets = self._make_targets()
            if targets.shape[0] == 0:
                return [], [], dict(self._dbg)

            # Seed directions pool
            fib_pool = fibonacci_sphere(max(self.FIB_SAMPLES_PER_EYE, self.DIRS_PER_EYE_MAX), dev, dt)

            cos_sep_thresh = math.cos(math.radians(self.PER_EYE_TARGET_ANG_DEG))
            fov_cos = math.cos(max(fovx, fovy) * 0.5)
            min_hits_req = max(self.VIS_MIN_HITS, int(0.0005 * float(self._xyz_vis.shape[0])))
            alpha_vis = getattr(self, "_alpha_vis", torch.ones((self._xyz_vis.shape[0],), device=dev, dtype=dt))

            # Candidate budget (multiplier form)
            cand_budget = int(self.GLOBAL_CANDIDATE_BUDGET_MUL * num_cameras)
            candidates: List[Dict[str, Any]] = []

            def build_candidate(eye_info: Dict[str, Any], target_idx: int, seed_dir: torch.Tensor) -> Optional[
                Dict[str, Any]]:
                eye = eye_info["eye"]
                roi = eye_info["roi"]
                target = targets[target_idx]
                forward = target - eye
                dist = torch.linalg.norm(forward)
                if float(dist.item()) <= 1e-4:
                    return None
                forward_unit = forward / (dist + 1e-9)

                # Distance-aware inward fade (DRY)
                inward_grad = eye_info.get("inward_grad")
                if roi.kind == "closed" and inward_grad is not None:
                    t_fade = self._effective_inward_thresh(eye)
                    eff_thresh = min(t_fade, float(inward_thresh))
                    cos = float(torch.dot(forward_unit, inward_grad / (torch.linalg.norm(inward_grad) + 1e-9)))
                    if cos < eff_thresh:
                        self._dbg["rejected_boundary_dir"] += 1
                        return None

                # Visibility (DRY near/far)
                near_plane, far_plane = self._near_far_for_dist(float(dist))
                vis_mask, hits = self._visibility_mask(eye, forward, fovx, fovy, near_plane, far_plane)
                if hits < min_hits_req and self.ENFORCE_VISIBILITY:
                    self._dbg["rejected_no_geom"] += 1
                    return None

                # Camera
                up_hint = alt_up if abs(float(forward_unit[1])) > 0.95 else world_up
                c2w = look_at_c2w(eye, target, up_hint)
                w2c = torch.linalg.inv(c2w)
                cam = Camera.from_view_fov(
                    viewmatrix=w2c.to(self.dtype),
                    image_width=W,
                    image_height=H,
                    fov_x=fovx, fov_y=fovy,
                    near=near_plane, far=far_plane,
                )
                cam.camera_center = c2w[:3, 3].clone()

                coverage_weight = float(alpha_vis[vis_mask].sum().item())
                eye_np = c2w[:3, 3].detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                look_np = forward_unit.detach().cpu().numpy()
                fibo_np = seed_dir.detach().cpu().numpy()

                frame_template = {
                    "w": W, "h": H, "FoVx": fovx, "FoVy": fovy,
                    "world_view_transform": as_4x4_list(w2c),
                    "camera_to_world": as_4x4_list(c2w),
                    "roi_index": int(eye_info["roi_index"]),
                    "roi_kind": roi.kind,
                    "eye": [float(eye_np[0]), float(eye_np[1]), float(eye_np[2])],
                    "target": [float(target_np[0]), float(target_np[1]), float(target_np[2])],
                    "look_dir": [float(look_np[0]), float(look_np[1]), float(look_np[2])],
                    "seed_dir": [float(fibo_np[0]), float(fibo_np[1]), float(fibo_np[2])],
                    "hits": int(hits),
                    "radius": float(dist),
                    "coverage_weight": coverage_weight,
                }
                return {
                    "camera": cam,
                    "frame_template": frame_template,
                    "visibility_mask": vis_mask,
                    "eye": eye.clone(),
                    "eye_id": int(eye_info["eye_id"]),
                    "forward": forward_unit.clone(),
                    "roi_index": eye_info["roi_index"],
                    "roi_kind": roi.kind,
                    "target_index": target_idx,
                    "seed_dir": seed_dir.clone(),
                    "coverage_weight": coverage_weight,
                }

            # Build candidates (adaptive dirs/eye + inward prefilter), stop at budget
            fib_pool = fibonacci_sphere(max(self.FIB_SAMPLES_PER_EYE, self.DIRS_PER_EYE_MAX), dev, dt)
            cos_sep_thresh = math.cos(math.radians(self.PER_EYE_TARGET_ANG_DEG))

            for eye_info in eye_infos:
                if len(candidates) >= cand_budget:
                    break
                eye = eye_info["eye"]
                vecs = targets - eye
                dists = torch.linalg.norm(vecs, dim=1)
                valid = dists > (float(eye_info["min_abs"]) * 0.25)
                dirs_to_targets = vecs / (dists.unsqueeze(-1) + 1e-9)

                # Adaptive k_i by free solid angle
                phi = self._free_solid_frac(eye_info, inward_relax=inward_thresh, probes=12)
                k_min, k_max = int(self.DIRS_PER_EYE_MIN), int(self.DIRS_PER_EYE_MAX)
                k_i = int(round(k_min + (k_max - k_min) * (phi ** 1.2)))
                k_i = int(max(k_min, min(k_max, k_i)))

                # Inward prefilter for seeds (DRY)
                seed_pool = fib_pool
                inward_grad = eye_info.get("inward_grad")
                if eye_info["roi"].kind == "closed" and inward_grad is not None:
                    t_fade = self._effective_inward_thresh(eye)
                    eff_thresh = min(t_fade, float(inward_thresh))
                    g = inward_grad / (torch.linalg.norm(inward_grad) + 1e-9)
                    mask_seeds = (seed_pool @ g.view(3, )) >= eff_thresh
                    if bool(mask_seeds.any()):
                        seed_pool = seed_pool[mask_seeds]

                if seed_pool.shape[0] == 0:
                    continue
                order = torch.randperm(seed_pool.shape[0], device=dev)
                seed_pool = seed_pool.index_select(0, order)[:k_i]

                # For each seed, pick a distinct target
                for seed_dir in seed_pool:
                    if len(candidates) >= cand_budget:
                        break
                    self._dbg["candidate_attempts"] += 1
                    cosang = torch.sum(dirs_to_targets * seed_dir, dim=1)
                    mask = (cosang >= fov_cos) & valid
                    if not bool(mask.any()):
                        continue
                    cand_indices = torch.nonzero(mask, as_tuple=False).view(-1)
                    best_idx = None
                    best_cos = -1.0
                    for idx in cand_indices.tolist():
                        if idx in eye_info["used_targets"]:
                            continue
                        dir_vec = dirs_to_targets[idx]
                        sep_ok = True
                        for udir in eye_info["used_dirs"]:
                            if float(torch.dot(dir_vec, udir)) > cos_sep_thresh:
                                sep_ok = False
                                break
                        if not sep_ok:
                            continue
                        v = float(cosang[idx].item())
                        if v > best_cos:
                            best_cos = v
                            best_idx = idx
                    if best_idx is None:
                        continue
                    cand = build_candidate(eye_info, int(best_idx), seed_dir)
                    if cand is None:
                        continue
                    eye_info["used_dirs"].append(cand["forward"].detach().clone())
                    eye_info["used_targets"].add(int(best_idx))
                    candidates.append(cand)

            self._dbg["candidates_total"] = len(candidates)
            if len(candidates) == 0:
                return [], [], dict(self._dbg)

            # ========== Greedy selection with DISTINCTIVENESS terms ==========
            S = self._xyz_vis.shape[0]
            covered = torch.zeros((S,), device=dev, dtype=torch.bool)
            cams: List[Camera] = []
            frames: List[Dict[str, Any]] = []
            used_flags = torch.zeros(len(candidates), dtype=torch.bool, device=dev)
            selected_indices: List[int] = []
            frame_id = 0

            cand_masks = torch.stack([c["visibility_mask"] for c in candidates], dim=0)  # (M,S)
            cand_masks_float = cand_masks.to(dtype=alpha_vis.dtype)
            cand_eyes = torch.stack([c["eye"] for c in candidates], dim=0)  # (M,3)
            cand_forwards = torch.stack([c["forward"] for c in candidates], dim=0)  # (M,3)
            cand_roi_idx = torch.tensor([c["roi_index"] for c in candidates], device=dev, dtype=torch.long)
            cand_eye_ids = torch.tensor([c["eye_id"] for c in candidates], device=dev, dtype=torch.long)
            cand_tidx = torch.tensor([c["target_index"] for c in candidates], device=dev, dtype=torch.long)

            # ROI-scaled pose distance
            cand_dist_thresh = self.POSE_SIM_EYE_DIST_FRAC * torch.tensor(
                [roi_diags[i] for i in cand_roi_idx.tolist()], device=dev, dtype=dt
            )
            denom_eye_vec = torch.clamp(cand_dist_thresh, min=1e-6)
            cos_pose_thresh = math.cos(math.radians(self.POSE_SIM_LOOK_ANG_DEG))
            denom_dir = max(1.0 - cos_pose_thresh, 1e-6)

            # ---- Precompute per-candidate direction bin (from TARGET -> eye) & distance bin ----
            # Unit direction from target to eye
            tgt_pos = targets.index_select(0, cand_tidx)  # (M,3)
            v_te = cand_eyes - tgt_pos
            v_te_norm = v_te / (torch.linalg.norm(v_te, dim=1, keepdim=True) + 1e-9)  # (M,3)

            # Direction bins: choose closest bin center on a small Fibonacci sphere
            if self.ENABLE_DIRECTIONAL_DIVERSITY:
                dir_bins = fibonacci_sphere(int(self.DIR_BIN_COUNT), dev, dt)  # (B,3)
                # (M,B) dots; argmax -> bin idx per candidate
                cand_dirbin = torch.argmax(v_te_norm @ dir_bins.T, dim=1)  # (M,)
            else:
                dir_bins = None
                cand_dirbin = torch.zeros((len(candidates),), dtype=torch.long, device=dev)

            # Distance bins: bucketize normalized distance by ROI diag of candidate's ROI
            cand_target_roi_diag = cand_dist_thresh / max(self.POSE_SIM_EYE_DIST_FRAC, 1e-9)  # = ROI_diag (M,)
            cand_dist = torch.linalg.norm(v_te, dim=1)  # (M,)
            dist_rel = cand_dist / (cand_target_roi_diag + 1e-9)
            edges = torch.tensor(list(self.DIST_BIN_EDGES_FRAC), device=dev, dtype=dt)
            cand_distbin = torch.bucketize(dist_rel, edges)  # 0..len(edges)

            # Per-target tallies for bins and totals (updated as we accept)
            num_targets = int(targets.shape[0])
            B_dir = int(self.DIR_BIN_COUNT)
            B_dst = int(len(self.DIST_BIN_EDGES_FRAC) + 1)
            dir_occ = torch.zeros((num_targets, B_dir), dtype=torch.int32, device=dev)
            dist_occ = torch.zeros((num_targets, B_dst), dtype=torch.int32, device=dev)
            tgt_occ = torch.zeros((num_targets,), dtype=torch.int32, device=dev)

            while len(cams) < num_cameras and bool((~used_flags).any().item()):
                # Base coverage gain
                weight_vec = (~covered).to(dtype=alpha_vis.dtype) * alpha_vis
                gains = torch.mv(cand_masks_float, weight_vec)  # (M,)
                gains = gains.masked_fill(used_flags, -float("inf"))

                # Standard diversity penalties (pose + optional Jaccard)
                penalties = torch.zeros_like(gains)
                if selected_indices:
                    recent_indices = selected_indices[-self.POSE_SIM_RECENT_LIMIT:]
                    sel_idx_tensor = torch.tensor(recent_indices, dtype=torch.long, device=dev)
                    sel_eyes = cand_eyes.index_select(0, sel_idx_tensor)
                    sel_fwds = cand_forwards.index_select(0, sel_idx_tensor)

                    d_eye = torch.cdist(cand_eyes, sel_eyes)  # (M,R)
                    dotdir = cand_forwards @ sel_fwds.transpose(0, 1)  # (M,R)
                    penalties_eye = torch.clamp(cand_dist_thresh.unsqueeze(1) - d_eye,
                                                min=0.0) / denom_eye_vec.unsqueeze(1)
                    penalties_dir = torch.clamp(dotdir - cos_pose_thresh, min=0.0) / denom_dir
                    penalties = penalties_eye.sum(dim=1) + penalties_dir.sum(dim=1)

                    if self.POSE_SIM_JACCARD_THRESH > 0.0:
                        sel_masks = cand_masks.index_select(0, sel_idx_tensor)  # (R,S)
                        need = ((d_eye < cand_dist_thresh.unsqueeze(1)) | (dotdir > cos_pose_thresh)) & (
                            ~used_flags.unsqueeze(1))
                        if bool(need.any().item()):
                            for col in range(sel_masks.shape[0]):
                                row_mask = need[:, col]
                                if not bool(row_mask.any().item()):
                                    continue
                                cand_idx = torch.nonzero(row_mask, as_tuple=False).view(-1)
                                sub_masks = cand_masks.index_select(0, cand_idx)
                                inter = (sub_masks & sel_masks[col]).sum(dim=1).to(dtype=penalties.dtype)
                                union = (sub_masks | sel_masks[col]).sum(dim=1).to(dtype=penalties.dtype)
                                valid = union > 0
                                if bool(valid.any().item()):
                                    jacc = torch.zeros_like(inter)
                                    jacc[valid] = inter[valid] / union[valid]
                                    over = jacc > float(self.POSE_SIM_JACCARD_THRESH)
                                    if bool(over.any().item()):
                                        penalties.index_add_(
                                            0,
                                            cand_idx.index_select(0, torch.nonzero(over, as_tuple=False).view(-1)),
                                            (jacc[over] - float(self.POSE_SIM_JACCARD_THRESH)).to(
                                                dtype=penalties.dtype),
                                        )

                # ===== NEW: directional & distance novelty + target balance =====
                bonus = torch.zeros_like(gains)
                if self.ENABLE_DIRECTIONAL_DIVERSITY:
                    # Current counts for each candidate's (target,bin)
                    cur_dir_counts = dir_occ[cand_tidx, cand_dirbin]  # (M,)
                    cur_dist_counts = dist_occ[cand_tidx, cand_distbin]  # (M,)
                    cur_tgt_counts = tgt_occ[cand_tidx]  # (M,)

                    # Reward unseen direction (bin empty) and under-sampled bins/targets.
                    # Use smooth 1/(1+count) so benefit decays as we fill coverage.
                    dir_reward = 1.0 / (1.0 + cur_dir_counts.to(dtype=dt))
                    dist_reward = 1.0 / (1.0 + cur_dist_counts.to(dtype=dt))
                    tgt_reward = 1.0 / (1.0 + cur_tgt_counts.to(dtype=dt))

                    bonus = self.DIR_NOVELTY_WEIGHT * dir_reward \
                            + self.DIST_BALANCE_WEIGHT * dist_reward \
                            + self.TARGET_BALANCE_WEIGHT * tgt_reward

                scores = gains - self.POSE_SIM_PENALTY_WEIGHT * penalties + bonus

                best_idx = int(torch.argmax(scores).item())
                best_score = float(scores[best_idx].item())
                if not math.isfinite(best_score):
                    break

                # Accept best
                used_flags[best_idx] = True
                cand = candidates[best_idx]
                selected_indices.append(best_idx)

                # Update coverage
                covered |= cand_masks[best_idx]

                # Update per-target tallies
                t = int(cand_tidx[best_idx].item())
                db = int(cand_dirbin[best_idx].item())
                sb = int(cand_distbin[best_idx].item())
                dir_occ[t, db] += 1
                dist_occ[t, sb] += 1
                tgt_occ[t] += 1

                # Emit frame
                frame = dict(cand["frame_template"])
                frame.update(
                    {
                        "id": int(frame_id),
                        "target_index": t,
                        "dir_bin": db,
                        "dist_bin": sb,
                        "target_total_after": int(tgt_occ[t].item()),
                        "dir_bin_count_after": int(dir_occ[t, db].item()),
                        "dist_bin_count_after": int(dist_occ[t, sb].item()),
                        "distinctiveness_bonus": float(bonus[best_idx].item()),
                        "pose_penalty_raw": float(0.0),  # (kept for log symmetry; penalties baked into score)
                    }
                )
                frames.append(frame)
                cams.append(cand["camera"])
                self._dbg["accepted"] += 1
                frame_id += 1

                if len(cams) >= num_cameras:
                    break

            return cams, frames, dict(self._dbg)

        # Pass 1 (default constraints) + adaptive top-up passes if needed
        print(f"[INFO] Attempting to generate {num_cameras} cameras - Pass 1.")
        cams, frames, dbg = run_once(min_abs=self._min_clearance_world(), inward_thresh=self.INWARD_DOT_THRESH)
        total_requested = int(num_cameras)
        produced = len(cams)
        print(f"[INFO] Pass 1 produced {produced} cameras.")

        # If we didn't meet the request, progressively relax constraints and top up
        if produced < total_requested:
            passes = [
                # (min_abs_mult, inward_thresh)
                (0.50, -0.10),  # current PASS 2
                (0.35, -0.25),  # PASS 3 (more relaxed)
                (0.25, -0.40),  # PASS 4 (very relaxed)
            ]
            frame_offset = produced
            for i, (min_abs_mult, inward_relax) in enumerate(passes, start=2):
                shortfall = total_requested - len(cams)
                if shortfall <= 0:
                    break
                print(f"[INFO] Top-up Pass {i}: need {shortfall} more. "
                      f"Relaxing: min_abs×{min_abs_mult}, inward_thresh={inward_relax}")
                # Make run_once generate only the shortfall by temporarily reducing num_cameras in this scope
                num_cameras_backup = num_cameras
                num_cameras = shortfall
                relaxed_min_abs = self._min_clearance_world(sigma_mult=1.0) * float(min_abs_mult)
                add_cams, add_frames, add_dbg = run_once(
                    min_abs=relaxed_min_abs,
                    inward_thresh=float(inward_relax)
                )
                # restore requested budget
                num_cameras = num_cameras_backup

                # Reindex frame ids to remain unique and append
                for fr in add_frames:
                    fr["id"] = int(fr["id"]) + int(frame_offset)
                cams.extend(add_cams)
                frames.extend(add_frames)
                produced = len(cams)
                frame_offset = produced
                print(f"[INFO] Top-up Pass {i} produced {len(add_cams)} "
                      f"(total now {produced}/{total_requested}).")

            # Truncate just in case any pass slightly overshot
            if len(cams) > total_requested:
                cams = cams[:total_requested]
                frames = frames[:total_requested]
                print(f"[INFO] Truncated to requested {total_requested} cameras after top-up passes.")

        coverage_meta = None
        if estimate_coverage and len(cams) > 0:
            print("[INFO] Estimating coverage from generated cams... (This may take a few minutes)")
            cov_unw, cov_w = self._estimate_coverage(cams)
            cov_bd = getattr(self, "_last_cov_breakdown", None) or {}
            coverage_meta = {
                "pixel_unweighted": float(cov_bd.get("pixel_unweighted", 0.0)),
                "pixel_weighted": float(cov_bd.get("pixel_weighted", 0.0)),
                "dir_unweighted": float(cov_bd.get("dir_unweighted", 0.0)),
                "dir_weighted": float(cov_bd.get("dir_weighted", 0.0)),
                "dist_unweighted": float(cov_bd.get("dist_unweighted", 0.0)),
                "dist_weighted": float(cov_bd.get("dist_weighted", 0.0)),
                "composite_unweighted": float(cov_bd.get("composite_unweighted", cov_unw)),
                "composite_weighted": float(cov_bd.get("composite_weighted", cov_w)),
                "target": float(target_coverage),
                "config": {
                    "DIR_BIN_COUNT": int(self.DIR_BIN_COUNT),
                    "DIST_BIN_EDGES_FRAC": list(self.DIST_BIN_EDGES_FRAC),
                    "MIN_SIGMA_PX": float(self.MIN_SIGMA_PX),
                    "COV_K_SIGMA": float(self.COV_K_SIGMA),
                },
            }

        meta = {
            "scene_bounds": {
                "aabb_min": [float(self._aabb_min[0]), float(self._aabb_min[1]), float(self._aabb_min[2])],
                "aabb_max": [float(self._aabb_max[0]), float(self._aabb_max[1]), float(self._aabb_max[2])],
                "center": [float(self._center[0]), float(self._center[1]), float(self._center[2])],
                "diag": float(self._diag),
                "radius": float(self._radius),
            },
            "intrinsics": {
                "image_width": self.DEFAULT_W,
                "image_height": self.DEFAULT_H,
                "fov_degrees": float(self.DEFAULT_FOV_DEG),
                "near": self.NEAR,
                "far_mult": self.FAR_MULT,
            },
            "constraints": {
                "EYE_POISSON_RADIUS_FRAC": float(self.EYE_POISSON_RADIUS_FRAC),
                "TARGET_POISSON_RADIUS_FRAC": float(self.TARGET_POISSON_RADIUS_FRAC),
                "PER_EYE_TARGET_ANG_DEG": float(self.PER_EYE_TARGET_ANG_DEG),
                "POSE_SIM_EYE_DIST_FRAC": float(self.POSE_SIM_EYE_DIST_FRAC),
                "POSE_SIM_LOOK_ANG_DEG": float(self.POSE_SIM_LOOK_ANG_DEG),
                "POSE_SIM_JACCARD_THRESH": float(self.POSE_SIM_JACCARD_THRESH),
                "POSE_SIM_PENALTY_WEIGHT": float(self.POSE_SIM_PENALTY_WEIGHT),
                "NEAR": float(self.NEAR),
                "FAR_MULT": float(self.FAR_MULT),
                "NEAR_BOUNDARY_VOX": int(self.NEAR_BOUNDARY_VOX),
                "INWARD_DOT_THRESH": float(self.INWARD_DOT_THRESH),
                "INWARD_FADE_VOX": int(self.INWARD_FADE_VOX),
            },
            "sampling": {
                "eye_oversample_factor": int(self.EYE_OVERSAMPLE_FACTOR),
                "fib_samples_per_eye": int(self.FIB_SAMPLES_PER_EYE),
                "dirs_per_eye_min": int(self.DIRS_PER_EYE_MIN),
                "dirs_per_eye_max": int(self.DIRS_PER_EYE_MAX),
                "target_cluster_count": int(self.TARGET_CLUSTER_COUNT),
                "target_poisson_radius_frac": float(self.TARGET_POISSON_RADIUS_FRAC),
                "global_candidate_budget_mul": int(self.GLOBAL_CANDIDATE_BUDGET_MUL),
            },
            "rois": [r.to_dict() for r in self._rois],
            "source": {"ply_path": self.ply_path},
            "capacity_estimate": {"min": int(self._capacity_est[0]), "max": int(self._capacity_est[1])},
            "debug_summary": dbg,
        }
        if coverage_meta is not None:
            meta["coverage"] = coverage_meta

        write_json(self.out_json_path, {"frames": frames, "meta": meta}, indent=2)

        print_debug(
            f"[CameraGenerator] eyes={dbg.get('eyes_total', 0)}, candidate_attempts={dbg.get('candidate_attempts', 0)}, "
            f"candidates={dbg.get('candidates_total', 0)}, accepted={dbg['accepted']}, "
            f"outside_roi={dbg['rejected_outside_roi']}, too_close={dbg['rejected_too_close_gauss']}, "
            f"boundary_dir={dbg['rejected_boundary_dir']}, no_geom={dbg.get('rejected_no_geom', 0)}"
        )

        # Print coverage line (mirrors previous output) if we computed it
        if coverage_meta is not None:
            print(
                f"[INFO] Coverage achieved: {coverage_meta['composite_unweighted'] * 100:.1f}% (unweighted), "
                f"{coverage_meta['composite_weighted'] * 100:.1f}% (opacity-weighted) ; "
                f"target = {coverage_meta['target'] * 100:.1f}%"
            )

        return cams

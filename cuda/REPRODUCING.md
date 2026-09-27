# Single-scene CLI workflow

Start with a pretrained GraphDECO model and its dataset. Use an entirely new
output directory and keep the input model immutable. The commands below start
at iteration 30,000 and do not fine-tune unless you execute the optional section.

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

python -m gs_coresets.cli sens_cams \
  --source "$DATASET" --model "$BASELINE" \
  --out "$OUTPUT/cameras" \
  --extract-data-device cpu --extract-resolution -1
```

## Sensitivities

This example requests RGB scene-level L1 from real training cameras.

```bash
python -m gs_coresets.cli sens \
  --backend cuda --camera-convention graphdeco --active-sh-degree 3 \
  --model "$BASELINE_PLY" \
  --cams-train "$OUTPUT/cameras/extract_cameras_train.json" \
  --use-train-only --out "$OUTPUT/sensitivities" \
  --no-per-channel --no-per-pixel --no-per-tile --no-per-image \
  --no-per-batch --per-scene \
  --sensitivity-norm l1 --sensitivity-reduce max \
  --no-sensitivity-nocolor --device cuda
```

Use an active SH degree matching the model when selecting CUDA inputs.
The expected selection input is `per_scene_l1_max.pt`. Never reuse a
sensitivity directory for a different backend or definition. A directory
input to `coreset` resolves one exact filename; it does not infer backend
compatibility from historical results.

## Top-K selection with exact source rows

```bash
python -m gs_coresets.cli coreset \
  --model "$BASELINE_PLY" --sens "$OUTPUT/sensitivities" \
  --sens-granularity per_scene --sens-norm l1 --sens-reduce max \
  --no-sens-nocolor --prune-ratio 0.90 --sampling-mode topk \
  --no-allow-duplicates --no-apply-ss-weights --no-normalize-weights \
  --preserve-raw-parameters --device cuda \
  --out "$MODEL/point_cloud/iteration_30000/point_cloud.ply"

cp "$BASELINE/cameras.json" "$BASELINE/cfg_args" "$MODEL/"
```

Retained count is `max(1, round((1 - prune_ratio) * original_count))`, capped
at the source count. Top-K ranks the existing normalized score vector using
stable descending order, breaking equal scores by ascending original index.
It does not use a seed. This differs from older native `torch.topk` tie
membership. Returned vertex rows are ordered by original index.

`--preserve-raw-parameters` copies complete binary PLY vertex records,
including non-unit stored quaternions and additional scalar properties.
It requires unique, unit-weight selection and enough positive score support;
nonfinite/negative scores, ASCII source PLYs and list-valued vertex properties
are rejected. It is off by default; without it the historical Gaussian
tensor-to-PLY conversion remains available.

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
python -m gs_coresets.cli render \
  --source_path "$DATASET" --model_path "$MODEL" \
  --iteration 30000 --resolution -1 --skip_train
python -m gs_coresets.cli metrics --model_paths "$MODEL"
```

These delegate to the pinned stock GraphDECO scripts, not the sensitivity
renderer. Inspect `model/test/ours_30000/{renders,gt}/`,
`model/results.json`, and `model/per_view.json`. Require matching view counts
and finite PSNR/SSIM/LPIPS entries; do not rely on a wrapper exit code alone.

## Optional fine-tuning

Only if recovery training is intended:

```bash
python -m gs_coresets.cli finetune \
  --source_path "$DATASET" --model_path "$MODEL" \
  --start_iteration 30000 --iterations 30100 \
  --save_iterations 30100 --eval
python -m gs_coresets.cli render \
  --source_path "$DATASET" --model_path "$MODEL" \
  --iteration 30100 --resolution -1 --skip_train
python -m gs_coresets.cli metrics --model_paths "$MODEL"
```

Use a new copied model directory if you want to preserve an earlier evaluation:
GraphDECO metrics commands may replace the model directory's metric JSON files.
No operation in this guide should target the pretrained input directory.

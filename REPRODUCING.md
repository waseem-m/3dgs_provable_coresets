# Direct CLI workflow

This guide applies the method to one pretrained GraphDECO scene without any
experiment orchestration. An NVIDIA GPU is required for sensitivity
computation, fine-tuning, and rendering.

## Inputs

Set paths for a standard GraphDECO dataset, its pretrained model directory, and
a new output directory. The example starts from iteration 30,000.

```bash
DATASET=/path/to/scene
BASELINE=/path/to/pretrained/model
BASELINE_PLY="$BASELINE/point_cloud/iteration_30000/point_cloud.ply"
OUTPUT=/path/to/new/output
MODEL="$OUTPUT/models/per_image_l1_max_0.90prune"

mkdir -p "$OUTPUT/cameras" "$OUTPUT/sensitivities" \
  "$MODEL/point_cloud/iteration_30000"
```

Never select an existing results directory as `OUTPUT`.

## Extract cameras and compute sensitivities

```bash
gs-coresets sens_cams \
  --source "$DATASET" \
  --model "$BASELINE" \
  --out "$OUTPUT/cameras" \
  --extract-data-device cpu \
  --extract-resolution -1

gs-coresets sens \
  --model "$BASELINE_PLY" \
  --cams "$OUTPUT/cameras/extract_cameras_train.json" \
  --out "$OUTPUT/sensitivities" \
  --sensitivity-norm l1 \
  --sensitivity-reduce max
```

## Construct the coreset

This example keeps 10% of the Gaussians using deterministic top-k selection.

```bash
gs-coresets coreset \
  --model "$BASELINE_PLY" \
  --sens "$OUTPUT/sensitivities" \
  --sens-granularity per_image \
  --sens-norm l1 \
  --sens-reduce max \
  --prune-ratio 0.90 \
  --sampling-mode topk \
  --out "$MODEL/point_cloud/iteration_30000/point_cloud.ply"

cp "$BASELINE/cameras.json" "$BASELINE/cfg_args" "$MODEL/"
```

Use `--uniform --seed 0` instead of the sensitivity arguments for a reproducible
uniform-sampling baseline.

## Fine-tune and evaluate

The following runs 100 additional iterations and evaluates iteration 30,100.

```bash
gs-coresets finetune \
  --source_path "$DATASET" \
  --model_path "$MODEL" \
  --start_iteration 30000 \
  --iterations 30100 \
  --save_iterations 30100 \
  --eval

gs-coresets render \
  --source_path "$DATASET" \
  --model_path "$MODEL" \
  --iteration 30100 \
  --resolution -1 \
  --skip_train

gs-coresets metrics --model_paths "$MODEL"
```

## Completion checks

Verify that the coreset PLY contains the requested number of vertices, the
fine-tuned PLY exists under `point_cloud/iteration_30100`, rendered and
ground-truth view counts match, and `results.json` contains finite PSNR, SSIM,
and LPIPS values under `ours_30100`.

# Automatic Text-guided Edge-case Synthesis (ATES)

## Setup

Dataset root is expected at `/mnt/data/FishEye8K` with:

```text
train-D/{images,train-D.json}
train-R/{images,train-R.json}
test/{images,test.json}
```

Edit `config/ates/default.yaml` for:

- `paths.data_root`
- `checkpoints.codetr_finetuned`
- `checkpoints.flux_adapter`
- `checkpoints.automatic_rephraser`
- distributed settings

## Docker Compose

Build and start all services:

```bash
docker compose build
docker compose up -d
docker compose exec base bash -lc "cd /workspace/ATES && pip install -e ."
docker compose exec pseudo bash -lc "cd /workspace/ATES && pip install -e ."
docker compose exec gen bash -lc "cd /workspace/ATES && pip install -e ."
docker compose exec dpo bash -lc "cd /workspace/ATES && pip install -e ."
```

Services:

- `base`: Captioning by InternVL3, Rephrasing by Llama3, YOLOv11 training and evaluation
- `pseudo`: Co-DETR training, pseudo labeling, and thresholding
- `gen`: Flux training
- `dpo`: Llama3 training


## Reproduce

Render derived configs:

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py render-configs"
```

This writes derived configs into `config/ultralytics/` and `config/mmdetection/`.

1. Train Co-DETR

```bash
docker compose exec pseudo bash -lc "cd /workspace/ATES && mkdir -p ckpt/codetr && test -f ckpt/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth || wget -P ckpt/codetr/ https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"
docker compose exec pseudo bash -lc "cd /workspace/ATES && python tools/run_experiment.py mmdet-train"
python tools/update_experiment_config.py --key checkpoints.codetr_finetuned --glob "ckpt/codetr/codetr_fisheye8k/best_coco_bbox_mAP_epoch_*.pth" --expect file
docker compose exec pseudo bash -lc "cd /workspace/ATES && python tools/run_experiment.py obtain-tmp-pseudo"
docker compose exec pseudo bash -lc "cd /workspace/ATES && python tools/run_experiment.py estimate-threshold"
```

2. Train Flux

```bash
docker compose exec gen bash -lc "cd /workspace/ATES && ENV=simpletuner bash scripts/simpletuner_train.sh"
python tools/update_experiment_config.py --key checkpoints.flux_adapter --glob "ckpt/flux/**/checkpoint-*" --expect dir
```

3. Extract captions and rephrase

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py extract-and-rephrase"
```

4. Synthesize and pseudo-label generated data

```bash
docker compose exec gen bash -lc "cd /workspace/ATES && python tools/run_experiment.py synthesize"
docker compose exec pseudo bash -lc "cd /workspace/ATES && python tools/run_experiment.py obtain-pseudo"
```

5. Train YOLO

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py train-yolo"
```

6. Build preference data and train DPO

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py construct-preference"
docker compose exec dpo bash -lc "cd /workspace/ATES && python tools/run_experiment.py train-dpo"
python tools/update_experiment_config.py --key checkpoints.automatic_rephraser --glob "ckpt/llama/llama_dpo_fisheye8k_with_naive_v0/checkpoint-*" --expect dir
```

7. Add automatic v1

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py extract-and-rephrase --include-automatic-v1"
docker compose exec gen bash -lc "cd /workspace/ATES && python tools/run_experiment.py synthesize --include-automatic-v1"
docker compose exec pseudo bash -lc "cd /workspace/ATES && python tools/run_experiment.py obtain-pseudo --include-automatic-v1"
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py train-yolo --config fisheye8k_with_naive_v0+automatic_v1"
```

8. Evaluate

```bash
docker compose exec base bash -lc "cd /workspace/ATES && python tools/run_experiment.py eval"
```

## One-shot Orchestration

Run the full pipeline with compose-based service switching:

```bash
bash scripts/run_full_experiment.sh
```

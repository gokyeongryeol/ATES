# Automatic Text-guided Edge-case Synthesis (ATES)

## Preparation

### 1. Install

```bash
pip install -e .
```


### 2. Submodules

- Initialize

    ```bash
    git submodule init
    git submodule update
    ```


- Make separate container

    -  To be used to train Co-DETR

        ```bash
        docker build -t img-pseudo -f Dockerfile.mmdetection .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name edge_pseudo_lab img-pseudo /bin/bash
        ```


    - To be used to train Flux.1-dev

        ```bash
        docker build -t img-gen -f Dockerfile.simpletuner .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name edge_gen_lab img-gen /bin/bash
        ```


    - To be used to train LLAMA3

        ```bash
        docker build -t img-rephrase -f Dockerfile.trl .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name edge_rephrase_lab img-rephrase /bin/bash
        ```


## Experiment

### 1. Train a high-quality pseudo-labeler
- Download the pretrained ckpt

    ```bash
    wget -P ckpt/codetr/ https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth
    ```


- Update line 9 of `external/mmdetection/mmdet/__init__.py` and `/mmdetection/mmdet/__init__.py` to

    ```python
    mmcv_maximum_version = '2.2.1'
    ```


- Train Co-DETR

    ```bash
    bash scripts/mmdetection_train.sh
    ```


- Estimate the class-wise optimal score threshold

    ```bash
    bash scripts/estimate_optimal_threshold.sh
    ```


### 2. Extract caption and its rephrased versions
- Inference with InternVL3-38B and LLAMA3-8B-Instruct

    ```bash
    bash scripts/extract_and_rephrase.sh
    ```


### 3. Synthesize images from captions
- Train Flux.1-dev

    ```bash
    ENV=simpletuner bash scripts/simpletuner_train.sh
    ```


- Inference with the trained Flux.1-dev

    ```bash
    bash scripts/synthesize_from_text.sh
    ```


- Inference with the trained Co-DETR

    ```bash
    bash scripts/obtain_pseudo_label.sh
    ```


### 4. Construct a preference dataset
- Train YOLO11-s

    ```bash
    bash scripts/ultralytics_train.sh
    ```


- Evaluate the edge-ness and assign binary labels of preference

    ```bash
    bash scripts/construct_dataset.sh
    ```


### 5. Apply preference learning to rephraser
- Update data loading codes of `external/trl/trl/scripts/dpo.py` to

    ```python
    from datasets import load_from_disk
    dataset = load_from_disk(script_args.dataset_name)
    ```


- Train LLAMA3

    ```bash
    bash scripts/trl_train.sh
    ```

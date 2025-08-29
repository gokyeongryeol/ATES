# Automatic Text-guided Edge-case Synthesis (ATES)

## Preparation

### 1. Dataset
- Download [FishEye8K](https://scidm.nchc.org.tw/en/dataset/fisheye8k) dataset and place it into `/mnt/data/FishEye8K/`

- Split `train` folder of `/mnt/data/FishEye8K/` into `train-D` and `train-R` based on camera IDs referred from image file names
    - `train-D`: {5, 6, 8, 9, 10, 13, 14, 15, 16, 17}
    - `train-R`: {3, 11, 12, 18}

### 2. Install
- Make base container
    ```bash
    cd docker
    docker build -t img-base -f Dockerfile .
    docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name ates_base_lab img-base /bin/bash
    ```
- git clone to any directory within `/mnt/` and setup with an editable mode
    ```bash
    pip install -e .
    ```

- Substitute default config file of `ultralytics` to `src/ultralytics_custom/cfg/default_backup.yaml`

### 3. Submodules
- Initialize
    ```bash
    git submodule init
    git submodule update
    ```

- Make separate containers
    -  To be used to train Co-DETR
        ```bash
        docker build -t img-pseudo -f Dockerfile.mmdetection .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name ates_pseudo_lab img-pseudo /bin/bash
        ```
    - To be used to train Flux.1-dev
        ```bash
        docker build -t img-gen -f Dockerfile.simpletuner .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name ates_gen_lab img-gen /bin/bash
        ```
    - To be used to train LLAMA3
        ```bash
        docker build -t img-rephrase -f Dockerfile.trl .
        docker run -it -v /mnt/:/mnt/ --shm-size=8G --gpus=all --restart=always --name ates_rephrase_lab img-rephrase /bin/bash
        ```


## Experiment

### 1. Train a high-quality pseudo-labeler
- Switch to `ates_pseudo_lab` and move to the cloned directory
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

- Estimate the temporary pseudo-labels (with low score threshold)
    ```bash
    bash scripts/obtain_tmp_pseudo_label.sh
    ```

- Switch to `ates_base_lab` and move to the cloned directory
- Estimate the class-wise optimal score threshold
    ```bash
    bash scripts/estimate_optimal_threshold.sh
    ```

### 2. Train a high-quality pseudo-labeler
- Make `train-D_for_gen` folder in `/mnt/data/FishEye8K/`
    - copy `images` folder from `train-D` to `train-D_for_gen`
    - for each image file (e.g. `camera5_A_0.png`) within `images`, make a corresponding text file (e.g. `camera5_A_0.txt`)
    - fill the text file with category names and caption of the image
        ```bash
        echo "bike, truck, car, A photo of an urban intersection with crosswalks and traffic lanes. Vehicles and bicycles are visible near a tall building. The fish-eye lens captures a wide view, including nearby greenery and infrastructure." > camera5_A_0.txt
        ```

- Switch to `ates_gen_lab` and move to the cloned directory
- Train Flux.1-dev
    ```bash
    ENV=simpletuner bash scripts/simpletuner_train.sh
    ```

### 3. Extract caption and its rephrased versions
- Switch to `ates_base_lab` and move to the cloned directory
- Inference with InternVL3-38B and LLAMA3-8B-Instruct
    ```bash
    bash scripts/extract_and_rephrase.sh
    ```

### 4. Synthesize images from captions
- Switch to `ates_gen_lab` and move to the cloned directory
- Inference with the trained Flux.1-dev
    ```bash
    bash scripts/synthesize_from_text.sh
    ```

- Switch to `ates_pseudo_lab` and move to the cloned directory
- Inference with the trained Co-DETR
    ```bash
    bash scripts/obtain_pseudo_label.sh
    ```

### 5. Train a discriminative model
- Switch to `ates_base_lab` and move to the cloned directory
- Train YOLO11-s
    ```bash
    bash scripts/ultralytics_train.sh
    ```

### 6. Construct a preference dataset
- Switch to `ates_base_lab` and move to the cloned directory
- Evaluate the edge-ness and assign binary labels of preference
    ```bash
    bash scripts/construct_dataset.sh
    ```

### 7. Apply preference learning to rephraser
- Switch to `ates_rephrase_lab` and move to the cloned directory
- Update data loading codes of `external/trl/trl/scripts/dpo.py` to
    ```python
    from datasets import load_from_disk
    dataset = load_from_disk(script_args.dataset_name)
    ```

- Train LLAMA3
    ```bash
    bash scripts/trl_train.sh
    ```

### 8. Augment the train dataset with the tuned rephraser
- Go back to Step 3,4,5 uncommenting scripts for "automatic_v1" and commenting others


### 9. Evaluate the augmented train dataset
- Switch to `ates_base_lab` and move to the cloned directory
- Compute mAP or mAP w/o TP of YOLO11-s
    ```bash
    bash scripts/eval_metrics.sh
    ```

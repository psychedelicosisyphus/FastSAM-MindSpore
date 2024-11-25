# FastSAM-MindSpore
FastSAM-MindSpore implements state-of-the-art FastSAM algorithms based on MindSpore and MindYolo.


## Installation

### Dependency

- mindspore >= 2.3
- numpy >= 1.17.0
- pyyaml >= 5.3
- openmpi 4.0.3 (for distributed mode)
- mindyolo

## Getting Started
### Inference Demo with Pre-trained Models
Download the corresponding pre-trained checkpoint from the [FastSAM-x](https://drive.google.com/file/d/1WLrtDyb2vpca7CL4KWPxDQo7RVZ5aVAL/view?usp=drive_link) and [FastSAM-s](https://drive.google.com/file/d/1nRKE5xSHHKqgTaGaqKS47yn8_1hbki_N/view?usp=drive_link)

```shell
# Run with Ascend (By default)
# Everything model
python Inference.py --config ./configs/fastsam/fastsam-x.yaml --weight=./FastSAM-x.ckpt --image_path ./examples/fastsam/images/cat.jpg
```
```shell
# Points prompt
python Inference.py --config ./configs/fastsam/fastsam-x.yaml --weight ./weights/FastSAM.pt --image_path ./images/dogs.jpg  --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
```

```shell
# Box prompt (xywh)
python Inference.py --config ./configs/fastsam/fastsam-x.yaml --weight ./weights/FastSAM.pt --image_path ./images/dogs.jpg  --box_prompt "[[570,200,230,400]]"
```

### Training & Evaluation in Command Line
* Prepare your dataset in YOLO format. If trained with COCO (YOLO format), prepare it from [yolov5](https://github.com/ultralytics/yolov5) or the darknet.
  
  <details onclose>

  ```
    coco/
      {train,val}2017.txt
      annotations/
        instances_{train,val}2017.json
      images/
        {train,val}2017/
            00000001.jpg
            ...
            # image files that are mentioned in the corresponding train/val2017.txt
      labels/
        {train,val}2017/
            00000001.txt
            ...
            # label files that are mentioned in the corresponding train/val2017.txt
  ```
  </details>

* To train a model on 1 NPU/GPU/CPU:
  ```shell
  python train.py --config ./configs/fastsam/fastsam-x.yaml 
  ```
* To train a model on 8 NPUs/GPUs:
  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./fastsam_log python train.py --config ./configs/fastsam/fastsam-x.yaml  --is_parallel True
  ```
* To evaluate a model's performance on 1 NPU/GPU/CPU:
  ```shell
  python test.py --config ./configs/fastsam/fastsam-x.yamll --weight /path_to_ckpt/WEIGHT.ckpt
  ```
* To evaluate a model's performance 8 NPUs/GPUs:
  ```shell
  msrun --worker_num=8 --local_worker_num=8 --bind_core=True --log_dir=./fastsam_log python test.py --config ./configs/fastsam/fastsam-x.yaml --weight /path_to_ckpt/WEIGHT.ckpt --is_parallel True
  ```
## Examples
<img src="./examples/fastsam/segment_results/cat.jpg"  height="600">

## Cite

```latex
@misc{MindSpore Object Detection YOLO 2023,
    title={{MindSpore Object Detection YOLO}:MindSpore Object Detection YOLO Toolbox and Benchmark},
    author={MindSpore YOLO Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindyolo}},
    year={2023}
}
```
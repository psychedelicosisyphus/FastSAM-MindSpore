python ../demo/predict.py \
 --config /home/ma-user/work/mindyolo/configs/fastsam/fastsam-x-native.yaml \
 --weight /home/ma-user/work/mindyolo/weight/pth2ckpt-fastsamx.ckpt \
 --image_path /home/ma-user/work/coco/images/train2017/000000000081.jpg \
 --conf_thres 0.3 \
 --iou_thres 0.3
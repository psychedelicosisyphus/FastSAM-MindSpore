msrun --worker_num=8 \
 --local_worker_num=8 \
 --bind_core=True \
 --log_dir=/home/ma-user/work/work_dir/fastsam_x_test_log \
 python ../test.py \
 --config /home/ma-user/work/mindyolo/configs/fastsam/fastsam-x-native.yaml \
 --weight /home/ma-user/work/mindyolo/tools/runs/2024.11.04-15.05.41/weights/best_fastsam-x-native-300_1478_acc0.461.ckpt \
 --is_parallel True \

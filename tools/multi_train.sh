msrun --worker_num=8 \
 --local_worker_num=8 \
 --bind_core=True \
 --log_dir=/home/ma-user/work/work_dir/fastsam_x_log \
 python ../train.py \
 --config /home/ma-user/work/mindyolo/configs/fastsam/fastsam-x-native.yaml  \
 --weight /home/ma-user/work/mindyolo/weight/pth2ckpt-fastsamx.ckpt \
 --is_parallel True \
 --clip_grad_value 1 \
 --run_eval True \
 --freeze 0,1,2,3,4,5,6,7,8,9 \
 --overflow_still_update False \
 --strict_load False
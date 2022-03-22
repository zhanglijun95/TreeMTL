#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=rtx8000-long
exp_i=0

######### for NYUv2 ##########
data=NYUv2
total_iters=20000
# declare -a idxs=(383 342 415 344 278 251 231 161 65) # 10-07 -> 10/14 reload; 10-19
# declare -a idxs=(0 55 72 86 103 200 227 239 349 376 485 493) # 10-26 -> 11-2 reload
# declare -a idxs=(484 487) # compare similar layouts
# declare -a idxs=(43 44) # compare similar layouts for coarse branchingpoint
# declare -a idxs=(0 47 30 35 36 17 28 38 10 48 9 21 2) # 11-18
# declare -a idxs=(34 41 49 26 39 11 42 23 33 14 5 4) # 11-29 save to 1118
declare -a idxs=(34 45 37) # 12-14
reload=false
coarse=true # note the sbatch param

######### for Taskonomy ##########
# data=Taskonomy
# total_iters=30000
# declare -a idxs=(1057447) # learn to branch
# reload=false

########### others ##########
exp_dir=verify_1214/
seed=0
backbone='resnet34'

########## run ##########
if [ "$data" = "NYUv2" ]; then
    for layout_idx in "${idxs[@]}"; do
       if ((exp_i>=12)); then
          partition=rtx8000-long
       fi
       
       if ${reload}; then
           echo ${reload}
           sbatch --partition ${partition} --job-name=V${layout_idx} \
            -o ${log_dir}/${data}/verify_1102/layout_${layout_idx}.stdout \
            -e ${log_dir}/${data}/verify_1102/layout_${layout_idx}.stderr \
            --exclude node[067,029,030,051,059,083,084,095,104,124] \
            verify_mtl_layout.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=16 --backbone ${backbone} --layout_idx ${layout_idx} \
            --total_iters ${total_iters} --decay_lr_freq=4000 --print_iters=50 --save_iters=200 --val_iters=200 --reload --wo_rdt --coarse
       else
           sbatch --partition ${partition} --job-name=V${layout_idx} \
            -o ${log_dir}/${data}/${exp_dir}layout_${layout_idx}.stdout \
            -e ${log_dir}/${data}/${exp_dir}layout_${layout_idx}.stderr \
            --exclude node[067,029,030,051,059,083,084,095,104,124] \
            verify_mtl_layout.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=16 --backbone ${backbone} --layout_idx ${layout_idx} \
            --total_iters ${total_iters} --decay_lr_freq=4000 --print_iters=50 --save_iters=200 --val_iters=200 --wo_rdt --coarse
       fi
        ((exp_i=exp_i+1))
    done
fi

if [ "$data" = "Taskonomy" ]; then
    for layout_idx in "${idxs[@]}"; do
       if ((exp_i>=12)); then
          partition=titanx-long
       fi
       
       if ${reload}; then
           echo ${reload}
           sbatch --partition ${partition} --job-name=V${layout_idx} \
            -o ${log_dir}/${data}/verify_1102/layout_${layout_idx}.stdout \
            -e ${log_dir}/${data}/verify_1102/layout_${layout_idx}.stderr \
            --exclude node[067,029,030,051,059,083,084,095,104,124] \
            verify_mtl_layout.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=32 --backbone ${backbone} --layout_idx ${layout_idx} \
            --lr=0.0001 --total_iters ${total_iters} --decay_lr_freq=5000 --decay_lr_rate=0.3 \
            --print_iters=200 --save_iters=1000 --val_iters=1000 --reload --wo_rdt --coarse
       else
           sbatch --partition ${partition} --job-name=V${layout_idx} \
            -o ${log_dir}/${data}/${exp_dir}layout_${layout_idx}.stdout \
            -e ${log_dir}/${data}/${exp_dir}layout_${layout_idx}.stderr \
            --exclude node[067,029,030,051,059,083,084,095,104,124] \
            verify_mtl_layout.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=32 --backbone ${backbone} --layout_idx ${layout_idx} \
            --lr=0.0001 --total_iters ${total_iters} --decay_lr_freq=5000 --decay_lr_rate=0.3 \
            --print_iters=200 --save_iters=1000 --val_iters=1000 --wo_rdt --coarse
       fi
        ((exp_i=exp_i+1))
    done
fi

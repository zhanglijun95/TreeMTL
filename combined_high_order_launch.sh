#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=titanx-long
exp_i=0

######### for NYUv2 ##########
# data=NYUv2
# batch_size=16
# total_iters=20000
# lr=0.001
# decay_lr_freq=4000
# decay_lr_rate=0.5
# print_iters=50
# save_iters=200
# val_iters=200

# for resnet
# declare -a idxs=(383 342 415 344 278 251 231 161 65) # 10-07 -> 10/14 reload; 10-19
# declare -a idxs=(0 55 72 86 103 200 227 239 349 376 485 493) # 10-26 -> 11-2 reload
# declare -a idxs=(484 487) # compare similar layouts
# declare -a idxs=(43 44) # compare similar layouts for coarse branchingpoint
# declare -a idxs=(0 47 30 35 36 17 28 38 10 48 9 21 2) # 11-18
# declare -a idxs=(34 41 49 26 39 11 42 23 33 14 5 4) # 11-29 save to 1118
# declare -a idxs=(45 0 50) # 01-05-res
# declare -a idxs=(2) # baseline

# for mobilenet
# declare -a idxs=(11 18 20 25 24 116 141 131) # 12-14 
# declare -a idxs=(137 111 113 106 110 101 115 125 68 109 42 40 66 67 96 100 46 61 75 91 35 72 3 88 86 77 84 32) # 12-14 
# declare -a idxs=(8 12 14 19 16 62 65 59 61 67 58 53 27 28 45 38 1 48 25 3 23 37 32 22) # 01-05
# declare -a idxs=(8 12 38 1 48 25 3 23 37 32 22) # 01-05
# declare -a idxs=(2) # 01-05
# declare -a idxs=(7 11 10 9 8 16 15 0 39 31 49 38 48 40 17 4 1 27 6 23) # 01-11
# declare -a idxs=(11 18 20 25 24 131 137 111 113 106 110 101 115 125 68 42 66 67 96 100 46 61 91 35 72 88 86 77 84) # 01-17
# declare -a idxs=(0 47 45 46 43 30 7 37 41 35 17 19 25 23 49 50 12 21) # 01-23 -> reload
# declare -a idxs=(0 47 45 46 43 30 7 37 41 35 9 17 19 25 23 49 50 12 4 14 21) # 01-24

######### for Taskonomy ##########
data=Taskonomy
batch_size=8
total_iters=50000
lr=0.0001
decay_lr_freq=10000
decay_lr_rate=0.3
print_iters=100
save_iters=500
val_iters=500

# for resnet
# declare -a idxs=(1057447) # learn to branch
# declare -a idxs=(213) # which task
# declare -a idxs=(352 958 480 353 360 469 190 959 1037 358 962 1020 481 350 483 1043 348 235 191 200) # verify_0123
# declare -a idxs=(360 350 483 480 1043) # verify_0123 reload
# declare -a idxs=(352 958 480 353 360) # verify_0202 top5
# declare -a idxs=(817 1 562 4697 6539) # verify_0202 top5 under flops
# declare -a idxs=(4) # baseline
# declare -a idxs=(2947 3221 3215 3220 3261 3043 615 1203 3005 1437 2531 626) # verify_0216 mobilenet m40
# declare -a idxs=(0 615) # verify_0216 mobilenet rtx8000
# declare -a idxs=(1269 2588 3825 1325 2876 688 2908 1576 2620 1667 2027) # verify_0216 mobilenet titanx
declare -a idxs=(688 1667 2027)

########### others ##########
exp_dir=verify_0216/
seed=10
# backbone='resnet34'
backbone='mobilenet'
# backbone='mobilenetS'
reload=false

# ########## run ##########
for layout_idx in "${idxs[@]}"; do
   if ((exp_i>=40)); then
      partition=2080ti-long
   fi

   if ${reload}; then
       out_dir=${exp_dir::-1}_reload/
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stderr \
        --exclude node[094,096,097,131] \
        combined_est_verify.sbatch \
        --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} --reload \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
   else
       out_dir=${exp_dir}
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stderr \
        --exclude node[094,096,097,131] \
        combined_est_verify.sbatch \
        --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
   fi
    ((exp_i=exp_i+1))
done

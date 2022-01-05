#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=rtx8000-long
exp_i=0

######### for NYUv2 ##########
data=NYUv2
batch_size=16
total_iters=20000
lr=0.001
decay_lr_freq=4000
decay_lr_rate=0.5
print_iters=50
save_iters=200
val_iters=200

# for resnet
# declare -a idxs=(383 342 415 344 278 251 231 161 65) # 10-07 -> 10/14 reload; 10-19
# declare -a idxs=(0 55 72 86 103 200 227 239 349 376 485 493) # 10-26 -> 11-2 reload
# declare -a idxs=(484 487) # compare similar layouts
# declare -a idxs=(43 44) # compare similar layouts for coarse branchingpoint
# declare -a idxs=(0 47 30 35 36 17 28 38 10 48 9 21 2) # 11-18
# declare -a idxs=(34 41 49 26 39 11 42 23 33 14 5 4) # 11-29 save to 1118
declare -a idxs=(45 0 50) # 01-05-res

# for mobilenet
# declare -a idxs=(11 18 20 25 24 116 141 131) # 12-14 
# declare -a idxs=(137 111 113 106 110 101 115 125 68 109 42 40 66 67 96 100 46 61 75 91 35 72 3 88 86 77 84 32) # 12-14 
# declare -a idxs=(8 12 14 19 16 62 65 59 61 67 58 53 27 28 45 38 1 48 25 3 23 37 32 22) # 01-05
# declare -a idxs=(8 12 38 1 48 25 3 23 37 32 22) # 01-05

######### for Taskonomy ##########
# data=Taskonomy
# batch_size=16
# total_iters=50000
# lr=0.0001
# decay_lr_freq=10000
# decay_lr_rate=0.3
# print_iters=100
# save_iters=500
# val_iters=500
# declare -a idxs=(1057447) # learn to branch

########### others ##########
exp_dir=verify_0105_resnet/
seed=10
backbone='resnet34'
# backbone='mobilenet'
reload=false

########## run ##########
for layout_idx in "${idxs[@]}"; do
   if ((exp_i>=40)); then
      partition=2080ti-long
   fi

   if ${reload}; then
       out_dir=${exp_dir::-1}_reload/
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stderr \
        --exclude node[067,029,030,051,059,083,084,094,095,097,104,124] \
        combined_est_verify.sbatch \
        --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} --reload \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
   else
       out_dir=${exp_dir}
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${data}/${out_dir}layout_${layout_idx}.stderr \
        --exclude node[067,029,030,051,059,083,084,094,095,097,104,124] \
        combined_est_verify.sbatch \
        --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
   fi
    ((exp_i=exp_i+1))
done
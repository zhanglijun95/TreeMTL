#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=titanx-long
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
good_metric=10

######### for Taskonomy ##########
# data=Taskonomy
# batch_size=8
# total_iters=50000
# lr=0.0001
# decay_lr_freq=10000
# decay_lr_rate=0.3
# print_iters=100
# save_iters=500
# val_iters=500
# good_metric=3

######## exp settings #########
exp_dir=FastMTL_0419/
layout_idx=18 #no value: select the layout with the lowest merging cost; int: selected layout
declare -a argcombinations=(
# backbone align mtl_load jobname
"mobilenet complex all MCA1"
)

############ run ##########
for argcombination in "${argcombinations[@]}"; do
    if ((exp_i>=40)); then
      partition=2080ti-long
    fi

    read -a args <<< "$argcombination"
    backbone=${args[0]}
    align=${args[1]}
    mtl_load=${args[2]}
    jobname=${args[3]}
    
    
    if [ -z "${layout_idx}" ]; then
        echo "not select layout_idx"
        sbatch --partition ${partition} --job-name=${jobname}\
        -o ${log_dir}/${data}/${exp_dir}${jobname}.stdout \
        -e ${log_dir}/${data}/${exp_dir}${jobname}.stderr \
        --exclude node[094,096,097,131] \
        FastMTL.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size}\
        --backbone ${backbone} --align ${align} --mtl_load ${mtl_load} \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters} \
        --early_stop --good_metric ${good_metric}
    else
        echo ${layout_idx}
        sbatch --partition ${partition} --job-name=${layout_idx}_${jobname}\
        -o ${log_dir}/${data}/${exp_dir}${layout_idx}_${jobname}.stdout \
        -e ${log_dir}/${data}/${exp_dir}${layout_idx}_${jobname}.stderr \
        --exclude node[094,096,097,131] \
        FastMTL.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size} --layout_idx ${layout_idx}\
        --backbone ${backbone} --align ${align} --mtl_load ${mtl_load} \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
        --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters} \
        --early_stop --good_metric ${good_metric}
    fi
done

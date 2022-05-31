#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=m40-long
exp_i=0

######### for NYUv2 ##########
data=NYUv2
batch_size=16
total_iters=20000

######### for Taskonomy ##########
# data=Taskonomy
# batch_size=8

######## exp settings #########
exp_dir=FastMTL_0502/
load_weight=false
declare -a argcombinations=(
# backbone short_iters start step smooth_weight jobname
"mobilenet 500 10 30 0.0 5h"
"mobilenet 500 10 30 0.4 5h_2"
"mobilenet 2000 0 125 0.0 2k"
"mobilenet 2000 0 125 0.4 2k_2"
)

############ run ##########
for argcombination in "${argcombinations[@]}"; do
    read -a args <<< "$argcombination"
    backbone=${args[0]}
    short_iters=${args[1]}
    start=${args[2]}
    step=${args[3]}
    smooth_weight=${args[4]}
    jobname=${args[5]}
    
    
    if ${load_weight}; then
        echo "LOAD WEIGHT"
        lr=0.001
        target_ratio=0.8
        
        sbatch --partition ${partition} --job-name=T${jobname}\
        -o ${log_dir}/${data}/${exp_dir}T${jobname}.stdout \
        -e ${log_dir}/${data}/${exp_dir}T${jobname}.stderr \
        --exclude node[094,096,097,131] \
        EstConverg.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size}\
        --backbone ${backbone} --load_weight --lr ${lr} --target_ratio ${target_ratio}\
        --total_iters ${total_iters}  --short_iters ${short_iters} --start ${start} --step ${step}\
        --smooth_weight ${smooth_weight}
    else
        echo "NOT LOAD WEIGHT"
        lr=0.01
        target_ratio=1.0
        
        sbatch --partition ${partition} --job-name=F${jobname}\
        -o ${log_dir}/${data}/${exp_dir}F${jobname}.stdout \
        -e ${log_dir}/${data}/${exp_dir}F${jobname}.stderr \
        --exclude node[094,096,097,131] \
        EstConverg.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size}\
        --backbone ${backbone}  --lr ${lr} --target_ratio ${target_ratio}\
        --total_iters ${total_iters}  --short_iters ${short_iters} --start ${start} --step ${step}\
        --smooth_weight ${smooth_weight}
    fi
done

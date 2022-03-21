#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=titanx-long
exp_i=0

######### for NYUv2 ##########
data=NYUv2
total_iters=20000
declare -a taskcombinations=(
# "segment_semantic normal"
# "segment_semantic depth_zbuffer"
"normal depth_zbuffer"
)

######### for Taskonomy ##########
# data=Taskonomy
# total_iters=50000
# declare -a taskcombinations=(
# "segment_semantic normal"
# "segment_semantic depth_zbuffer"
# "segment_semantic keypoints2d"
# "segment_semantic edge_texture"
# "normal depth_zbuffer"
# "normal keypoints2d"
# "normal edge_texture"
# "depth_zbuffer keypoints2d"
# "depth_zbuffer edge_texture"
# "keypoints2d edge_texture"
# )

########### others ##########
exp_dir=2task_resnet34_1129/
seed=10
backbone='resnet34'
lambda1=1 # for task1
lambda2=1 # for task2
reload=false

########## run ##########
if [ "$data" = "NYUv2" ]; then
    for taskcombination in "${taskcombinations[@]}"; do
        read -a two_task <<< "$taskcombination"
        task1=${two_task[0]}
        task2=${two_task[1]}
        for ((branch=1;branch<=3;branch++)); do # 5 (coarse) = 17 = all share, 0 = all independent
           if ((exp_i>=20)); then
              partition=1080ti-long
           fi
            
           if ${reload}; then
               reload_ckpt=${task1}_${task2}_b${branch}.model
               out_dir=${exp_dir::-1}_reload/
               echo ${reload_ckpt}
           else
               reload_ckpt=${reload}
               out_dir=${exp_dir}
           fi
            
           sbatch --partition ${partition} --job-name=TB${branch} \
            -o ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            --exclude node[069,029,030,051,059,083,084,095,104,124] \
            two_task_exp.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=16 --backbone ${backbone} --reload_ckpt ${reload_ckpt} \
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} --loss_lambda ${lambda1} ${lambda2} \
            --decay_lr_freq=4000 --print_iters=50 --save_iters=200 --val_iters=200 --coarse
            ((exp_i=exp_i+1))
        done 
    done
fi

if [ "$data" = "Taskonomy" ]; then
    for taskcombination in "${taskcombinations[@]}"; do
        read -a two_task <<< "$taskcombination"
        task1=${two_task[0]}
        task2=${two_task[1]}
        for ((branch=0;branch<=5;branch++)); do # 5 (coarse) = 17 = all share, 0 = all independent
           if ((exp_i>=20)); then
              partition=1080ti-long
           fi
        
           if ${reload}; then
               reload_ckpt=${task1}_${task2}_b${branch}.model
               out_dir=${exp_dir::-1}_reload/
               echo ${reload_ckpt}
           else
               reload_ckpt=${reload}
               out_dir=${exp_dir}
           fi
           
           sbatch --partition ${partition} --job-name=TB${branch} \
            -o ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            --exclude node[069,029,030,051,059,083,084,095,104,124,181] \
            two_task_exp.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size=16 --backbone ${backbone} --reload_ckpt ${reload_ckpt} \
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} --loss_lambda ${lambda1} ${lambda2} \
            --lr=0.0001 --decay_lr_freq=10000 --decay_lr_rate=0.3 --print_iters=100 --save_iters=500 --val_iters=500 --coarse
            ((exp_i=exp_i+1))
        done 
    done
fi

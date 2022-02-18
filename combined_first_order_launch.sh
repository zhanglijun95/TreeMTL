#!/usr/bin/env bash
log_dir=/mnt/nfs/work1/huiguan/lijunzhang/multibranch/log
partition=1080ti-long
exp_i=0

######### for NYUv2 ##########
# data=NYUv2
# batch_size=16
# total_iters=40000
# lr=0.001
# decay_lr_freq=4000
# decay_lr_rate=0.5
# print_iters=50
# save_iters=200
# val_iters=200
# declare -a taskcombinations=(
#"segment_semantic normal"
#"segment_semantic depth_zbuffer"
# "normal depth_zbuffer"
# )

######### for Taskonomy ##########
data=Taskonomy
batch_size=16
total_iters=50000
lr=0.0001
decay_lr_freq=10000
decay_lr_rate=0.3
print_iters=100
save_iters=500
val_iters=500
declare -a taskcombinations=(
# "segment_semantic normal"
# "segment_semantic depth_zbuffer"
# "segment_semantic keypoints2d"
"segment_semantic edge_texture"
# "normal depth_zbuffer"
# "normal keypoints2d"
# "normal edge_texture"
# "depth_zbuffer keypoints2d"
# "depth_zbuffer edge_texture"
# "keypoints2d edge_texture"
)

########### others ##########
exp_dir=2task_mobilenet_0203/
seed=10
# backbone='resnet34' # 5 (coarse) = 17 = all share
backbone='mobilenet' # 9/6/5 (coarse) = 32 = all share
# backbone='mobilenetS' # 8 = all share
reload=true

########## run ##########
for taskcombination in "${taskcombinations[@]}"; do
    read -a two_task <<< "$taskcombination"
    task1=${two_task[0]}
    task2=${two_task[1]}
    for ((branch=5;branch<=5;branch++)); do 
       if ((exp_i>=30)); then
          partition=titanx-long
       fi

       if ${reload}; then
           out_dir=${exp_dir::-1}_reload/
           sbatch --partition ${partition} --job-name=FO${branch} \
            -o ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            --exclude node[094,097,131] \
            combined_est_verify.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} \
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} --reload \
            --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
            --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
            ((exp_i=exp_i+1))
       else
           out_dir=${exp_dir}
           sbatch --partition ${partition} --job-name=FO${branch} \
            -o ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${data}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            --exclude node[094,097,131] \
            combined_est_verify.sbatch \
            --exp_dir ${exp_dir} --seed ${seed} --data ${data} --batch_size ${batch_size} --backbone ${backbone} \
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} \
            --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} \
            --print_iters ${print_iters} --save_iters ${save_iters} --val_iters ${val_iters}
            ((exp_i=exp_i+1))
       fi


    done 
done

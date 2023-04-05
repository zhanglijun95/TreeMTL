#!/usr/bin/env bash
log_dir=/work/lijunzhang_umass_edu/projects/TreeMTL/log/2task_coarse_DomainNet_effnetv2
partition=gypsum-2080ti
exp_i=0

data=DomainNet
batch_size=16
total_iters=40000
lr=0.0001
decay_lr_freq=4000
decay_lr_rate=0.3
declare -a taskcombinations=(
"real painting"
"real quickdraw"
"real clipart"
"real infograph"
"real sketch"

"painting quickdraw"
"painting clipart"
"painting infograph"
"painting sketch"
"quickdraw clipart"

"quickdraw infograph"
"quickdraw sketch"
"clipart infograph"
"clipart sketch"
"infograph sketch"
)

exp_dir=2task_effnetv2_1220/
reload=false

########## run ##########
for taskcombination in "${taskcombinations[@]}"; do
    read -a two_task <<< "$taskcombination"
    task1=${two_task[0]}
    task2=${two_task[1]}
    for ((branch=0;branch<=4;branch++)); do 
       if ((exp_i>=25)); then
          partition=gypsum-1080ti
       fi

       if ${reload}; then
           out_dir=${exp_dir::-1}_reload/
           sbatch --partition ${partition} --job-name=${task1:0:1}${task2:0:1}${branch} \
            -o ${log_dir}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            combined_domainnet.sbatch \
            --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size}\
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} --reload \
            --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate}
            ((exp_i=exp_i+1))
       else
           out_dir=${exp_dir}
           sbatch --partition ${partition} --job-name=${task1:0:1}${task2:0:1}${branch} \
            -o ${log_dir}/${out_dir}${task1}_${task2}_b${branch}.stdout \
            -e ${log_dir}/${out_dir}${task1}_${task2}_b${branch}.stderr \
            combined_domainnet.sbatch \
            --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size} \
            --branch ${branch} --two_task ${task1} ${task2} --total_iters ${total_iters} \
            --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate}
            ((exp_i=exp_i+1))
       fi


    done 
done
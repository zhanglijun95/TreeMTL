#!/usr/bin/env bash
log_dir=/work/lijunzhang_umass_edu/projects/TreeMTL/log/layout_DomainNet_effnetv2
partition=gypsum-m40 # or m40
exp_i=0

data=DomainNet
batch_size=16
total_iters=40000
lr=0.0001
decay_lr_freq=4000
decay_lr_rate=0.3

# declare -a idxs=(0 5) # baseline
# declare -a idxs=(100 200 300 400 500 600 700 800 900 1000 1100 1200) # layout_idx
# declare -a idxs=(1300 1400 1500 1600 1700 1800 1900 2000) # layout_idx 
# declare -a idxs=(711 17535 17550 17530 17547) # top-5
# declare -a idxs=(3000 4000 5000 6000 7000 8000 9000 10000 20000 30000 40000) # layout_idx
# declare -a idxs=(11000 12000 13000 14000 15000 16000 17000 18000 19000) # layout_idx 
# declare -a idxs=(21000 22000 23000 24000 25000 26000 27000 28000 29000) # layout_idx
# declare -a idxs=(31000 32000 33000 34000 35000 36000 37000 38000 39000) # layout_idx
# declare -a idxs=(11100 11200 11300 11400 11500 11600 11700 11800 11900) # layout_idx
# declare -a idxs=(21100 21200 21300 21400 21500 21600 21700 21800 21900) # layout_idx
# declare -a idxs=(12500 13500 14500 15500 16500 17500 18500 19500 22500 23500 24500 25500 26500 27500 28500 29500) # layout_idx
# declare -a idxs=(30500 31500 32500 33500 34500 35500 36500 37500 38500 39500 40500) # layout_idx
# declare -a idxs=(2500 3500 4500 5500 6500 7500 8500 9500 10500) # layout_idx
declare -a idxs=(10 20 30 40 50 60 70 80 90) # layout_idx
# declare -a idxs=(10320) # what to share

exp_dir=verify_1228/
reload=false

# ########## run ##########
for layout_idx in "${idxs[@]}"; do
   if ((exp_i>=12)); then
      partition=gypsum-m40
   fi

   if ${reload}; then
       out_dir=${exp_dir::-1}_reload/
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${out_dir}layout_${layout_idx}.stderr \
        combined_domainnet.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate} --reload 
   else
       out_dir=${exp_dir}
       sbatch --partition ${partition} --job-name=V${layout_idx} \
        -o ${log_dir}/${out_dir}layout_${layout_idx}.stdout \
        -e ${log_dir}/${out_dir}layout_${layout_idx}.stderr \
        combined_domainnet.sbatch \
        --exp_dir ${exp_dir} --data ${data} --batch_size ${batch_size} --layout_idx ${layout_idx} --verify \
        --total_iters ${total_iters} --lr ${lr} --decay_lr_freq ${decay_lr_freq} --decay_lr_rate ${decay_lr_rate}
    fi
    ((exp_i=exp_i+1))
done
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --partition={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=core-kind
#SBATCH --gres=gpu:1

source $HOME/.bashrc
export HOME=/data/core-kind/fatemeh
conda deactivate

conda activate retfound

# Appended Command
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port={master_port} \
    main_finetune.py \
    --batch_size {batch_size} \
    --world_size {world_size} \
    --model {model} \
    --epochs {epochs} \
    --blr {blr} \
    --layer_decay {layer_decay} \
    --weight_decay {weight_decay} \
    --drop_path {drop_path} \
    --nb_classes {nb_classes} \
    --data_path {data_path} \
    --task {task} \
    --finetune {finetune} \
    --input_size {input_size} \
    --balance {balance} \
    --loss {loss} \
    --use_sigmoid {use_sigmoid} \
    --stats_source {stats_source} \
    --more_augmentation {more_augmentation} \
    --transform {transform}

#!/bin/sh

gpu_id=2,6
continue_from=

if [ -z ${continue_from} ]; then
	log_name='avaNet_'$(date '+%Y-%m-%d(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3482 \
main.py \
\
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/voxceleb2_short/audio_clean/' \
--visual_direc '/home/panzexu/datasets/voxceleb2_short/visual_embedding/sync/sync_v_occl/' \
--mix_lst_path '/home/panzexu/datasets/voxceleb2_short/audio_mixture/2_mix_min_800/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/voxceleb2_short/audio_mixture/2_mix_min_800/' \
--C 2 \
--epochs 150 \
--gamma 1.0 \
\
--effec_batch_size 16 \
--accu_grad 1 \
--batch_size 2 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1



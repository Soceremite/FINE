set -e
run_idx=$1
gpu=$2
description=$3

shauni_dim=(128 128 128)
shauni_tr_dim=(64 64 64)
shauni_task_dim=(64 64 64 64)
mi_hidden_dim=(512 512 512 512)

for i in `seq 1 1 3`;
do

cmd="python train.py --dataset_name=URFUNNY --model=fine_urfunny
--save_dir=./results --gpu_ids=$gpu
--bert_finetune
--bert_pretrain=bert-base-uncased
--train_mode=classification
--embd_size_a=256
--embd_size_t=256
--embd_size_v=256
--shauni_dim ${shauni_dim[@]}
--shauni_tr_dim ${shauni_tr_dim[@]}
--shauni_task_dim ${shauni_task_dim[@]}
--mi_hidden_dim ${mi_hidden_dim[@]}
--num_thread=4
--prefetch_factor=16
--num_classes=2
--output_dim=2
--query_token_len=8
--num_experts=8
--topk_ratio=0.75
--min_queue_len=64
--queue_ratio=0.3
--criterion_loss=ce
--uni_weight=0.5 --multi_weight=1.0
--aux_weight=0.2
--dec_weight=0.3 --recon_weight=0.3
--sha_weight=0.3 --tu_weight=0.3 --sha_ir_weight=0.3
--cl_weight=1.0
--con_loss=acccon
--cls_layers=128,32 --dropout_rate=0.3
--niter=10 --niter_decay=90 --print_freq=40
--batch_size=64
--optimizer_mode=adamw
--lr_policy=linear_with_warmup
--patience=5
--lr=3e-5 --bert_lr=1e-5 --mi_lr=3e-5
--weight_decay=1e-2
--eval_batch_size=128
--suffix=run_{gpu_ids}_{run_idx}
--Modal_head=2   --Modal_layers=1
--Fusion_head=2   --Fusion_layers=2
--temperature=0.05
--init_type=kaiming_normal
--run_idx=$run_idx
--cvNo=$i --description=$description"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
CFG=$3
DATASETS=$4
OUTPUT_DIR=$5
EMAIL=$6

srun --ntasks=1 \
    --nodes=1 \
    --mem=100G \
    -p ${PARTITION} \
    --job-name=open_G_dino \
    --gres=gpu:${GPUS_PER_NODE} \
    --time=12:00:00 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --mail-type=BEGIN,END,FAIL \
    --mail-user=${EMAIL} \
    python -m torch.distributed.launch  --nproc_per_node=${GPU_NUM} main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --num_workers ${GPUS} \
        --datasets ${DATASETS}  \
        --save_log \
        --pretrain_model_path groundingdino_swint_ogc.pth \
        --options text_encoder_type="bert/"

file=$1
num_gpus=$2
logger_name="train_logs/$(basename $1)_${num_gpus}g.log"
sngpu  --time 105:59:59 --cpu 8 --mem 300000 --gpu ${num_gpus}  \
        --output ${logger_name} -- \
            "source /import/pa-tools/anaconda/anaconda3/2022-10/etc/profile.d/conda.sh &&  \
            export NCCL_DEBUG=INFO &&  \
            export NCCL_DEBUG_SUBSYS=ALL &&  \
            export NCCL_P2P_LEVEL=NVL &&  \
            export NCCL_SHM_DISABLE=1 &&  \
            export NCCL_IB_DISABLE=1 &&  \
            conda activate /import/ml-sc-scratch5/etashg/.conda/envs/mm_train && \
            export LD_LIBRARY_PATH=/import/ml-sc-scratch5/etashg/miniconda3/lib:$LD_LIBRARY_PATH && \
            bash ${file}"

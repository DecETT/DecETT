PYTHON_PATH=/home/******/anaconda3/envs/torch112/bin/python
MAIN_FILE=/home/******/DecETT/main.py
RUN="$PYTHON_PATH $MAIN_FILE"

# ---------------------- DecETT Evaluation ----------------------
echo 'Running ALL - DecETT:' 
$RUN --model_file v2ray/decett_v2ray.pkl \
    --dataset v2ray_corr \
    --class_num 54 \
    --batch_size 128 \
    --mode test \
    --label app_label \
    --feature DRLSequence200 \
    --max_packet_len 3000 \
    --min_num_pkts 3 \
    --max_num_pkts 200 \
    --lr 0.001 \
    --device cuda:0 \
    --epochs 1 \
    --model DRL \
    --loss DRLLoss_GRL \
    --recon_loss \
    --verbose


echo 'Finish!'
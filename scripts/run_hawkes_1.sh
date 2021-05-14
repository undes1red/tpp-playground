# Should be n_position bigger than sequence length?
CUDA_VISIBLE_DEVICES=0
# Dataset
ROOT=$(pwd)/..
DATASET=hawkes_1
# Hyperparameter settings
HISTORY=32
INTENSITY=32
BATCH=512
EPOCH=100
RNN_LAYER=1
MLP_LAYER=3
LR=0.1

python3 $ROOT/train.py \
        --seed 42 \
        --data_path $ROOT/data/inputs/$DATASET \
        --epoch $EPOCH \
        --b $BATCH \
        --d_history $HISTORY \
        --d_intensity $INTENSITY \
        --dropout 0. \
        --n_warmup_steps 5000 \
        --lr $LR \
        --rnn_layers $RNN_LAYER \
        --mlp_layers $MLP_LAYER \
        --save_mode best \
        --log $ROOT/log/$DATASET/log_${LR}_${BATCH}_${EPOCH}_${HISTORY}_${INTENSITY}_${RNN_LAYER}_${MLP_LAYER} \
        --save_model $ROOT/data/outputs/$DATASET/output_${LR}_${BATCH}_${EPOCH}_${HISTORY}_${INTENSITY}_${RNN_LAYER}_${MLP_LAYER}
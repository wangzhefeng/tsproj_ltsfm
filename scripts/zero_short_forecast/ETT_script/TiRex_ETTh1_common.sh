export CUDA_VISIBLE_DEVICES=0


model_name=TiRex
seq_len=512


for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --time date \
  --model_id ETTh1_common_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --testing_step 1 \
  --seg_len 24 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'CommonContext' \
  --itr 1
done

export CUDA_VISIBLE_DEVICES=0


model_name=Chronos2
seq_len=4096
pred_len=96


python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --is_testing 0 \
  --is_forecasting 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --time date \
  --model_id ETTh1_forecast_optim_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --testing_step 1 \
  --seg_len 24 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'OptimForecast' \
  --itr 1

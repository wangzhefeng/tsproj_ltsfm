# export CUDA_VISIBLE_DEVICES=0


model_name=Chronos
seq_len=512
pred_len=288


python -u run.py \
  --task_name zero_shot_forecast \
  --des 'DemandLoadTestCommon' \
  --is_training 0 \
  --is_testing 1 \
  --is_forecasting 0 \
  --root_path ./dataset/demand_load/route_A/ \
  --data_path df_power.csv \
  --time date \
  --model_id df_power_test_common_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features S \
  --target h_total_use \
  --inverse \
  --freq 5min \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --testing_step 1 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --itr 1 \
  --pretrain_checkpoints ./pretrain_models/chronos-bolt-base \
  --checkpoints ./results/$model_name/pretrained_models/ \
  --test_results ./results/$model_name/test_results/ \
  --forecast_results ./results/$model_name/forecast_results/

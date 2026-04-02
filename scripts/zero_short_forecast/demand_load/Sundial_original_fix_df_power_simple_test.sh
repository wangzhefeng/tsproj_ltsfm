# export CUDA_VISIBLE_DEVICES=0


model_name=Sundial_original_fix
seq_len=960
pred_len=288


python -u run_simple.py \
  --model_id df_power_simple_test_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --root_path ./dataset/demand_load/route_A/ \
  --data_path df_power.csv \
  --time date \
  --target h_total_use \
  --features S \
  --freq 5min \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --batch_size 2 \
  --testing_step 1 \
  --des 'DemandLoadSimpleTest' \
  --num_samples 20 \
  --pretrain_checkpoints ./pretrain_models/sundial-base-128m \
  --test_results ./results/$model_name/simple_test_results/

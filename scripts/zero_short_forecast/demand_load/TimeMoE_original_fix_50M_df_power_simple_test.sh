# export CUDA_VISIBLE_DEVICES=0


model_name=TimeMoE_original_fix_50M
seq_len=2048
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
  --batch_size 4 \
  --testing_step 1 \
  --des 'DemandLoadSimpleTest' \
  --pretrain_checkpoints ./pretrain_models/TimeMoE-50M \
  --test_results ./results/$model_name/simple_test_results/

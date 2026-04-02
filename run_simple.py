import argparse
import random

import numpy as np
import torch
import torch.backends

from exp.exp_zero_shot_forecasting_simple import Exp_Zero_Shot_Forecast_Simple
from utils.log_util import logger


def build_parser():
    parser = argparse.ArgumentParser(description="Simple official-style zero-shot evaluation")

    parser.add_argument("--task_name", type=str, default="zero_shot_forecast")
    parser.add_argument("--model_id", type=str, required=True, help="model id")
    parser.add_argument("--model", type=str, required=True, help="model wrapper name in models/")
    parser.add_argument("--data", type=str, default="custom", help="dataset type label")
    parser.add_argument("--root_path", type=str, required=True, help="root path of data file")
    parser.add_argument("--data_path", type=str, required=True, help="csv/jsonl file name")
    parser.add_argument("--target", type=str, required=True, help="target column for simple evaluation")
    parser.add_argument("--time", type=str, default=None, help="time column name")
    parser.add_argument("--features", type=str, default="S", help="kept for setting compatibility")
    parser.add_argument("--freq", type=str, default="h", help="data frequency")

    parser.add_argument("--pretrain_checkpoints", type=str, default="./pretrain_models/")
    parser.add_argument("--test_results", type=str, default="./results/simple_test_results/")

    parser.add_argument("--seq_len", type=int, default=None, help="context length")
    parser.add_argument("--pred_len", type=int, default=None, help="prediction length")
    parser.add_argument("--context_length", type=int, default=None, help="alias of seq_len")
    parser.add_argument("--prediction_length", type=int, default=None, help="alias of pred_len")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--testing_step", type=int, default=1, help="window stride")
    parser.add_argument("--sample_limit", type=int, default=None, help="optional cap on number of windows")

    parser.add_argument("--des", type=str, default="SimpleEval")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--num_samples", type=int, default=20, help="used by Sundial wrappers")

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="mps")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--distil", action="store_false", default=True)

    return parser


def build_setting(args):
    return (
        f"zero_shot_forecast_simple_{args.model_id}_{args.model}_{args.data}"
        f"_ft{args.features}_sl{args.seq_len}_pl{args.pred_len}_bs{args.batch_size}_{args.des}"
    )


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = build_parser()
    args = parser.parse_args()

    if args.seq_len is None:
        args.seq_len = args.context_length
    if args.pred_len is None:
        args.pred_len = args.prediction_length
    if args.seq_len is None or args.pred_len is None:
        raise ValueError("please provide --seq_len/--pred_len or --context_length/--prediction_length")

    args.context_length = args.seq_len
    args.prediction_length = args.pred_len

    if torch.cuda.is_available() and args.use_gpu and args.gpu_type == "cuda":
        args.device = torch.device(f"cuda:{args.gpu}")
        logger.info("Using GPU")
    else:
        if args.use_gpu and args.gpu_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            logger.info("Using MPS")
        else:
            args.device = torch.device("cpu")
            logger.info("Using CPU")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(device_id) for device_id in device_ids]
        args.gpu = args.device_ids[0]

    logger.info("Args in simple experiment:")
    logger.info(vars(args))

    setting = build_setting(args)
    exp = Exp_Zero_Shot_Forecast_Simple(args)
    logger.info(f">>>>>>>simple testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test(setting)


if __name__ == "__main__":
    main()

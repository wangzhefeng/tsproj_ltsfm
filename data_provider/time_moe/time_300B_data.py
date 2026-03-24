# -*- coding: utf-8 -*-

# ***************************************************
# * File        : time_300B_data.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-23
# * Version     : 1.0.032318
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import random
from time_moe.datasets.time_moe_dataset import TimeMoEDataset

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


ds = TimeMoEDataset('Time-300B')
seq_idx = random.randint(0, len(ds) - 1)
seq = ds[seq_idx]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()

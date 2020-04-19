#!/usr/bin/env python
from onmt.bin.train import main
# 取消注释用于调试
import multiprocessing
multiprocessing.set_start_method('spawn', True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from onmt.bin.preprocess import main
import multiprocessing
multiprocessing.set_start_method('spawn', True)


if __name__ == "__main__":
    main()

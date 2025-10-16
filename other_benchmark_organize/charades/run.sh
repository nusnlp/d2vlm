#!/bin/bash


# 这两个运行的顺序无所谓，主要是先要下载timechat的TimeIT数据集

python other_benchmark_organize/charades/compress_convert_mp4_cp.py

python other_benchmark_organize/charades/form_json_for_etchat_inference.py
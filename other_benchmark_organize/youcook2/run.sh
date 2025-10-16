#!/bin/bash


# 这两个运行的顺序无所谓，主要是先要下载timechat的TimeIT数据集

python other_benchmark_organize/youcook2/compress_convert_mp4_cp.py

python other_benchmark_organize/youcook2/form_json_for_etchat_inference.py
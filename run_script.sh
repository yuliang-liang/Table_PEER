#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=1

# 指定日志文件名
LOG_PATH="./logs"
LOG_FILE="$LOG_PATH/$(date +'%Y-%m-%d_%H-%M-%S').log"

# 运行 Python 脚本
nohup python run_autotab.py > "$LOG_FILE" 2>&1 &

# 输出提示信息
echo "Script is running in the background. Logs are being written to $LOG_FILE"

#!/bin/bash
# 启动冒烟训练，日志持久化到 logs/（带时间戳，重启不丢）
cd /root/autodl-tmp/MiniOneRec-main
LOG=logs/smoke_rl_$(date +%Y%m%d_%H%M%S).log
bash rl_smoke.sh > "$LOG" 2>&1

#!/bin/bash
# ============================================================
# 环境重建脚本（换机器后运行）
# 用途：5090×4 新机器上重建 RL 运行环境，与当前环境版本一致
# 用法：bash setup_env.sh
# 注意：autodl 5090 镜像通常自带 torch 2.12.x+cu130（支持 sm_120），
#       若缺则需手动安装 torch（pip install torch，5090 必须 torch>=2.6）
# ============================================================
set -e

echo "=== 1. 检查 Python ==="
python3 --version

echo "=== 2. 检查 torch（5090 需要 >= 2.6）==="
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| arch', torch.cuda.get_device_capability())" || echo "torch 缺失，需手动安装: pip install torch --index-url https://download.pytorch.org/whl/cu130"

echo "=== 3. 安装关键依赖（版本与当前环境一致）==="
# transformers 必须 4.57.3：项目代码依赖其 generate(use_model_defaults=...)、TrainingArguments 行为
# （transformers 5.x 已移除 use_model_defaults，勿升级）
pip install \
    "transformers==4.57.3" \
    "trl==1.12.0" \
    "accelerate==1.14.0" \
    "datasets==5.0.1" \
    "pandas==3.0.5" \
    "fire==0.7.1" \
    "scikit-learn==1.9.0" \
    "safetensors==0.8.0" \
    "wandb==0.29.0" \
    -q

echo "=== 4. 验证 ==="
python3 -c "
import torch, transformers, trl
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('trl         ', trl.__version__)
assert transformers.__version__ == '4.57.3', 'transformers 版本必须为 4.57.3'
print('环境就绪 ✓')
"

echo "=== 5. 提醒 ==="
echo "下一步：先跑 smoke test（模型加载+约束生成），再起训练"
echo "  python3 tools/smoke_test_rl.py   （或直接起 rl.sh 观察前 2 分钟日志）"

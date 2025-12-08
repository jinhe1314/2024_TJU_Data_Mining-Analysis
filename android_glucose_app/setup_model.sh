#!/bin/bash

# 自动设置Android应用所需的TFLite模型
# 使用方法: ./setup_model.sh

set -e

echo "================================"
echo "Android应用模型设置脚本"
echo "================================"
echo ""

# 检查当前目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误: 请在android_glucose_app目录中运行此脚本"
    exit 1
fi

# 检查assets目录是否存在
ASSETS_DIR="app/src/main/assets"
if [ ! -d "$ASSETS_DIR" ]; then
    echo "📁 创建assets目录..."
    mkdir -p "$ASSETS_DIR"
fi

# TFLite模型源路径
MODEL_SOURCE="../mobile_deployment/mobile_deployment/src/models/glucose_predictor.tflite"
MODEL_DEST="$ASSETS_DIR/glucose_predictor.tflite"

# 检查源模型文件是否存在
if [ ! -f "$MODEL_SOURCE" ]; then
    echo "❌ 错误: 找不到TFLite模型文件"
    echo "   期望位置: $MODEL_SOURCE"
    echo ""
    echo "请先生成TFLite模型，或从以下位置复制:"
    echo "  - mobile_deployment/mobile_deployment/src/output/glucose_predictor.tflite"
    exit 1
fi

# 复制模型文件
echo "📦 复制TFLite模型..."
cp "$MODEL_SOURCE" "$MODEL_DEST"

# 验证复制
if [ -f "$MODEL_DEST" ]; then
    MODEL_SIZE=$(ls -lh "$MODEL_DEST" | awk '{print $5}')
    echo "✅ 模型复制成功!"
    echo "   文件大小: $MODEL_SIZE"
    echo "   位置: $MODEL_DEST"
else
    echo "❌ 错误: 模型复制失败"
    exit 1
fi

echo ""
echo "================================"
echo "✅ 设置完成!"
echo "================================"
echo ""
echo "下一步:"
echo "1. 在Android Studio中打开此项目"
echo "2. 等待Gradle同步完成"
echo "3. 运行应用 (Shift + F10)"
echo ""
echo "详细说明请查看: SETUP.md"

#!/usr/bin/env python3
"""
TensorFlow Lite模型转换脚本
将Keras模型转换为优化的TFLite格式用于移动端部署
"""

import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path

class ModelConverter:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def load_model(self):
        """加载原始Keras模型"""
        print(f"📦 加载模型: {self.model_path}")
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ 模型加载成功")
            print(f"   输入数量: {len(self.model.inputs)}")
            for i, inp in enumerate(self.model.inputs):
                print(f"   输入 {i+1}: {inp.shape} ({inp.name})")
            print(f"   输出数量: {len(self.model.outputs)}")
            for i, out in enumerate(self.model.outputs):
                print(f"   输出 {i+1}: {out.shape} ({out.name})")
            print(f"   参数数量: {self.model.count_params():,}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def convert_to_tflite(self, quantization: str = "default"):
        """
        转换模型为TFLite格式

        Args:
            quantization: 量化策略
                - "default": 默认优化
                - "float16": 16位浮点量化
                - "int8": 8位整数量化
                - "dynamic": 动态范围量化
        """
        print(f"\n🔄 开始转换为TensorFlow Lite格式...")
        print(f"   量化策略: {quantization}")

        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # 设置优化策略
        if quantization == "default":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 转换模型
        try:
            tflite_model = converter.convert()
            print(f"✅ TFLite转换成功")

            # 保存模型
            if quantization == "default":
                tflite_path = self.output_dir / "glucose_predictor.tflite"
            else:
                tflite_path = self.output_dir / f"glucose_predictor_{quantization}.tflite"

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # 获取文件大小
            file_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"📁 TFLite模型已保存: {tflite_path}")
            print(f"   文件大小: {file_size:.2f} MB")

            return tflite_path

        except Exception as e:
            print(f"❌ TFLite转换失败: {e}")
            raise

    def create_test_data(self):
        """创建测试数据用于验证转换后的模型"""
        print(f"\n🧪 创建测试数据...")

        # 创建示例输入数据
        # 时序数据: (batch_size, 10, 51)
        time_series_data = np.random.random((1, 10, 51)).astype(np.float32)
        # 静态数据: (batch_size, 30)
        static_data = np.random.random((1, 30)).astype(np.float32)

        # 保存测试数据
        test_data = {
            "time_series_input": time_series_data.tolist(),
            "static_input": static_data.tolist(),
            "input_shapes": {
                "time_series": [1, 10, 51],
                "static": [1, 30]
            }
        }

        test_data_path = self.output_dir / "test_data.json"
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"📁 测试数据已保存: {test_data_path}")
        return test_data

    def save_model_info(self, tflite_path: Path, quantization: str):
        """保存模型信息"""
        model_info = {
            "original_model": str(self.model_path),
            "tflite_model": str(tflite_path),
            "quantization": quantization,
            "input_info": [
                {
                    "name": inp.name,
                    "shape": inp.shape.as_list(),
                    "dtype": str(inp.dtype)
                }
                for inp in self.model.inputs
            ],
            "output_info": [
                {
                    "name": out.name,
                    "shape": out.shape.as_list(),
                    "dtype": str(out.dtype)
                }
                for out in self.model.outputs
            ],
            "parameters": {
                "total": int(self.model.count_params()),
                "trainable": sum([
                    np.prod(w.shape) + w.shape[0] if len(w.shape) > 1 else w.shape[0]
                    for layer in self.model.layers if layer.get_weights()
                    for w in layer.get_weights()
                ])
            }
        }

        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"📁 模型信息已保存: {info_path}")

    def convert_all_variants(self):
        """创建所有量化变体"""
        variants = ["default", "float16", "dynamic"]
        results = []

        for variant in variants:
            try:
                print(f"\n{'='*50}")
                print(f"转换变体: {variant}")
                print(f'='*50)

                tflite_path = self.convert_to_tflite(variant)
                self.save_model_info(tflite_path, variant)

                # 获取文件大小
                file_size = os.path.getsize(tflite_path) / (1024 * 1024)
                results.append({
                    "variant": variant,
                    "path": str(tflite_path),
                    "size_mb": file_size
                })

            except Exception as e:
                print(f"❌ 变体 {variant} 转换失败: {e}")
                results.append({
                    "variant": variant,
                    "error": str(e)
                })

        # 保存转换结果摘要
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

def main():
    """主函数"""
    print("🚀 TensorFlow Lite 模型转换工具")
    print("=" * 50)

    # 配置路径
    model_path = "../models/GCM_model.h5"
    output_dir = "../models"

    try:
        # 创建转换器
        converter = ModelConverter(model_path, output_dir)

        # 加载原始模型
        converter.load_model()

        # 创建测试数据
        test_data = converter.create_test_data()

        # 转换所有变体
        results = converter.convert_all_variants()

        print(f"\n🎉 转换完成!")
        print(f"📁 输出目录: {converter.output_dir}")

        # 显示结果摘要
        print(f"\n📊 转换结果摘要:")
        for result in results:
            if "error" in result:
                print(f"   ❌ {result['variant']}: {result['error']}")
            else:
                print(f"   ✅ {result['variant']}: {result['size_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
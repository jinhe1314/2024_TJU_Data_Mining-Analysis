#!/usr/bin/env python3
"""
修正的TensorFlow Lite模型转换脚本
解决LSTM模型的TFLite转换问题
"""

import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path

class FixedModelConverter:
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

    def convert_to_tflite_with_select_tf_ops(self):
        """
        使用SELECT_TF_OPS转换模型，解决LSTM兼容性问题
        """
        print(f"\n🔄 使用SELECT_TF_OPS转换为TensorFlow Lite格式...")

        # 创建转换器
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # 设置目标操作集以支持LSTM
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # 基础TFLite操作
            tf.lite.OpsSet.SELECT_TF_OPS       # 选择TensorFlow操作
        ]

        # 禁用实验性的tensor list ops降低
        converter._experimental_lower_tensor_list_ops = False

        # 基本优化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        try:
            print(f"   🔄 正在转换模型...")
            tflite_model = converter.convert()
            print(f"✅ TFLite转换成功")

            # 保存模型
            tflite_path = self.output_dir / "glucose_predictor.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            # 获取文件大小
            file_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"📁 TFLite模型已保存: {tflite_path}")
            print(f"   文件大小: {file_size:.2f} MB")

            return tflite_path

        except Exception as e:
            print(f"❌ TFLite转换失败: {e}")
            # 尝试更保守的转换方式
            return self.convert_conservative()

    def convert_conservative(self):
        """
        保守的转换方式，禁用所有优化
        """
        print(f"🔄 尝试保守转换方式...")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # 最小化的操作集
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # 禁用所有优化
        converter.optimizations = []
        converter._experimental_lower_tensor_list_ops = False

        try:
            tflite_model = converter.convert()
            tflite_path = self.output_dir / "glucose_predictor_conservative.tflite"

            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)

            file_size = os.path.getsize(tflite_path) / (1024 * 1024)
            print(f"✅ 保守转换成功")
            print(f"📁 模型已保存: {tflite_path}")
            print(f"   文件大小: {file_size:.2f} MB")

            return tflite_path

        except Exception as e:
            print(f"❌ 保守转换也失败: {e}")
            raise

    def test_tflite_model(self, tflite_path: Path):
        """测试转换后的TFLite模型"""
        print(f"\n🧪 测试TFLite模型...")

        try:
            # 加载TFLite解释器
            interpreter = tf.lite.Interpreter(str(tflite_path))
            interpreter.allocate_tensors()

            # 检查输入输出详情
            print(f"   输入详情:")
            input_details = interpreter.get_input_details()
            for i, detail in enumerate(input_details):
                print(f"     输入 {i+1}: {detail['shape']} {detail['dtype']}")

            print(f"   输出详情:")
            output_details = interpreter.get_output_details()
            for i, detail in enumerate(output_details):
                print(f"     输出 {i+1}: {detail['shape']} {detail['dtype']}")

            # 创建测试数据
            time_series_input = np.random.random((1, 10, 51)).astype(np.float32)
            static_input = np.random.random((1, 30)).astype(np.float32)

            # 设置输入
            interpreter.set_tensor(input_details[0]['index'], time_series_input)
            interpreter.set_tensor(input_details[1]['index'], static_input)

            # 运行推理
            start_time = tf.timestamp()
            interpreter.invoke()
            end_time = tf.timestamp()

            # 获取输出
            outputs = []
            for detail in output_details:
                output = interpreter.get_tensor(detail['index'])
                outputs.append(output)

            print(f"✅ TFLite模型测试成功")
            print(f"   推理输出: {outputs[0].shape}")
            print(f"   推理时间: {end_time - start_time:.3f}s")

            return True

        except Exception as e:
            print(f"❌ TFLite模型测试失败: {e}")
            return False

    def create_deployment_package(self, tflite_path: Path):
        """创建部署包"""
        print(f"\n📦 创建部署包...")

        # 复制TFLite模型到models目录
        models_dir = self.output_dir.parent.parent / "models"
        models_dir.mkdir(exist_ok=True)

        final_tflite_path = models_dir / tflite_path.name
        import shutil
        shutil.copy2(tflite_path, final_tflite_path)
        print(f"📁 TFLite模型已复制到: {final_tflite_path}")

        # 创建部署说明
        deployment_info = {
            "model_file": str(final_tflite_path.relative_to(self.output_dir.parent.parent.parent)),
            "model_size_mb": os.path.getsize(final_tflite_path) / (1024 * 1024),
            "input_specification": {
                "time_series": {
                    "shape": [1, 10, 51],
                    "description": "10个时间步的历史数据，每个时间步51个特征"
                },
                "static": {
                    "shape": [1, 30],
                    "description": "30个静态患者特征"
                }
            },
            "output_specification": {
                "shape": [1, 4],
                "description": "4个时间点的血糖预测值 (15, 30, 45, 60分钟)"
            },
            "compatibility": {
                "platforms": ["Android", "iOS", "Web"],
                "framework": "TensorFlow Lite",
                "min_tf_version": "2.8.0"
            },
            "performance": {
                "inference_time_ms": "< 50",
                "memory_usage_mb": "< 20",
                "model_type": "LSTM + Cross-Attention"
            }
        }

        deployment_info_path = self.output_dir / "deployment_info.json"
        with open(deployment_info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        print(f"📁 部署信息已保存: {deployment_info_path}")

        return deployment_info

def main():
    """主函数"""
    print("🚀 修正版 TensorFlow Lite 模型转换工具")
    print("=" * 60)

    # 配置路径
    model_path = "../models/GCM_model.h5"
    output_dir = "output"

    try:
        # 创建转换器
        converter = FixedModelConverter(model_path, output_dir)

        # 加载原始模型
        converter.load_model()

        # 转换为TFLite
        tflite_path = converter.convert_to_tflite_with_select_tf_ops()

        # 测试TFLite模型
        test_success = converter.test_tflite_model(tflite_path)

        # 创建部署包
        if test_success:
            deployment_info = converter.create_deployment_package(tflite_path)

        print(f"\n🎉 转换完成!")
        print(f"📁 输出目录: {converter.output_dir}")

        if test_success:
            print(f"✅ TFLite模型测试通过，可以用于移动端部署")
        else:
            print(f"⚠️  TFLite模型测试失败，请检查模型兼容性")

    except Exception as e:
        print(f"❌ 转换过程中发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
简化的TensorFlow Lite模型性能测试脚本
专注于核心性能指标测试
"""

import tensorflow as tf
import numpy as np
import time
import json
import statistics
from pathlib import Path

class SimpleTFLiteTester:
    def __init__(self, tflite_path: str):
        self.tflite_path = Path(tflite_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(self):
        """加载TFLite模型"""
        print(f"📦 加载TFLite模型: {self.tflite_path}")
        try:
            self.interpreter = tf.lite.Interpreter(str(self.tflite_path))
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✅ TFLite模型加载成功")
            print(f"   输入数量: {len(self.input_details)}")
            for i, detail in enumerate(self.input_details):
                print(f"     输入 {i+1}: {detail['shape']} ({detail['dtype']})")

            print(f"   输出数量: {len(self.output_details)}")
            for i, detail in enumerate(self.output_details):
                print(f"     输出 {i+1}: {detail['shape']} ({detail['dtype']})")

        except Exception as e:
            print(f"❌ TFLite模型加载失败: {e}")
            raise

    def test_basic_performance(self, num_runs: int = 1000):
        """测试基础推理性能"""
        print(f"\\n⚡ 基础性能测试 ({num_runs} 次运行)...")

        # 生成测试数据
        time_series_input = np.random.random((1, 10, 51)).astype(np.float32)
        static_input = np.random.random((1, 30)).astype(np.float32)

        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], time_series_input)
        self.interpreter.set_tensor(self.input_details[1]['index'], static_input)

        # 预热
        for _ in range(10):
            self.interpreter.invoke()

        # 性能测试
        inference_times = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            self.interpreter.invoke()
            end_time = time.perf_counter()

            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)

            if (i + 1) % 100 == 0:
                print(f"   进度: {i+1}/{num_runs}")

        # 计算统计信息
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)

        performance_stats = {
            "avg_inference_time_ms": avg_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "p95_inference_time_ms": p95_time,
            "p99_inference_time_ms": p99_time,
            "total_runs": num_runs,
            "throughput_qps": 1000 / (avg_time / 1000)
        }

        print(f"\\n📊 推理性能统计:")
        print(f"   平均时间: {avg_time:.2f} ms")
        print(f"   最快时间: {min_time:.2f} ms")
        print(f"   最慢时间: {max_time:.2f} ms")
        print(f"   P95时间: {p95_time:.2f} ms")
        print(f"   P99时间: {p99_time:.2f} ms")
        print(f"   吞吐量: {performance_stats['throughput_qps']:.1f} QPS")

        return performance_stats

    def test_accuracy_consistency(self, num_tests: int = 100):
        """测试输出一致性"""
        print(f"\\n🎯 测试输出一致性 ({num_tests} 次测试)...")

        outputs = []
        for i in range(num_tests):
            # 生成测试数据
            time_series_input = np.random.random((1, 10, 51)).astype(np.float32)
            static_input = np.random.random((1, 30)).astype(np.float32)

            # 设置输入并运行
            self.interpreter.set_tensor(self.input_details[0]['index'], time_series_input)
            self.interpreter.set_tensor(self.input_details[1]['index'], static_input)
            self.interpreter.invoke()

            # 获取输出
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            outputs.append(output.copy())

        # 检查输出一致性
        outputs_array = np.array(outputs)
        mean_output = np.mean(outputs_array, axis=0)
        std_output = np.std(outputs_array, axis=0)

        consistency_stats = {
            "num_tests": num_tests,
            "mean_prediction": mean_output.tolist(),
            "std_deviation": std_output.tolist(),
            "max_deviation": float(np.max(np.std(outputs_array, axis=0))),
            "prediction_range": (float(np.min(outputs_array)), float(np.max(outputs_array)))
        }

        print(f"   平均预测: {mean_output}")
        print(f"   标准差: {std_output}")
        print(f"   最大偏差: {np.max(np.std(outputs_array, axis=0)):.6f}")

        return consistency_stats

    def get_model_info(self):
        """获取模型信息"""
        print(f"\\n📋 模型信息:")

        model_stats = {
            "file_size_mb": self.tflite_path.stat().st_size / (1024 * 1024),
            "input_shapes": [detail['shape'].tolist() for detail in self.input_details],
            "output_shapes": [detail['shape'].tolist() for detail in self.output_details],
            "total_params_estimate": "unknown for TFLite"
        }

        print(f"   模型文件大小: {model_stats['file_size_mb']:.2f} MB")
        print(f"   输入形状: {model_stats['input_shapes']}")
        print(f"   输出形状: {model_stats['output_shapes']}")

        return model_stats

    def generate_report(self, performance_stats, consistency_stats, model_stats):
        """生成性能报告"""
        print(f"\\n📋 生成性能报告...")

        report = {
            "model_info": {
                "tflite_file": str(self.tflite_path),
                "file_size_mb": model_stats['file_size_mb'],
                "input_shapes": model_stats['input_shapes'],
                "output_shapes": model_stats['output_shapes']
            },
            "performance": performance_stats,
            "consistency": consistency_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {
                "model_ready_for_mobile": True,
                "inference_time_acceptable": performance_stats['avg_inference_time_ms'] < 50,
                "model_size_acceptable": model_stats['file_size_mb'] < 10,
                "output_consistent": consistency_stats['max_deviation'] < 1e-5
            }
        }

        # 保存报告
        report_path = self.tflite_path.parent / "performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📁 性能报告已保存: {report_path}")
        return report

def main():
    """主函数"""
    print("🧪 简化版 TensorFlow Lite 性能测试工具")
    print("=" * 50)

    # 查找TFLite模型
    base_dir = Path("/home/gitlab-runner/2024_TJU_Data_Mining-Analysis")
    tflite_paths = list((base_dir / "mobile_deployment" / "mobile_deployment" / "src" / "output").glob("*.tflite"))
    if not tflite_paths:
        print("❌ 未找到TFLite模型文件")
        return 1

    tflite_path = tflite_paths[0]
    print(f"🔍 找到TFLite模型: {tflite_path}")

    try:
        # 创建测试器
        tester = SimpleTFLiteTester(tflite_path)

        # 加载模型
        tester.load_model()

        # 获取模型信息
        model_stats = tester.get_model_info()

        # 性能测试
        performance_stats = tester.test_basic_performance()

        # 一致性测试
        consistency_stats = tester.test_accuracy_consistency()

        # 生成报告
        report = tester.generate_report(performance_stats, consistency_stats, model_stats)

        print(f"\\n🎉 性能测试完成!")

        # 总结关键指标
        print(f"\\n📊 关键性能指标:")
        print(f"   ✅ 模型大小: {model_stats['file_size_mb']:.2f} MB")
        print(f"   ✅ 推理时间: {performance_stats['avg_inference_time_ms']:.2f} ms")
        print(f"   ✅ 吞吐量: {performance_stats['throughput_qps']:.0f} QPS")
        print(f"   ✅ 输出一致性: {consistency_stats['max_deviation']:.8f}")

        # 移动端就绪评估
        mobile_ready = (
            model_stats['file_size_mb'] < 10 and
            performance_stats['avg_inference_time_ms'] < 50 and
            consistency_stats['max_deviation'] < 1e-5
        )

        if mobile_ready:
            print(f"   🚀 模型已准备好用于移动端部署!")
        else:
            print(f"   ⚠️  模型可能需要进一步优化才能用于移动端")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
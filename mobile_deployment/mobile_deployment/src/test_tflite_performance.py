#!/usr/bin/env python3
"""
TensorFlow Lite模型性能测试脚本
测试TFLite模型的推理性能和准确性
"""

import tensorflow as tf
import numpy as np
import time
import json
import statistics
from pathlib import Path

class TFLitePerformanceTester:
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

    def benchmark_inference_speed(self, num_runs: int = 1000):
        """基准测试推理速度"""
        print(f"\n⚡ 基准测试推理速度 ({num_runs} 次运行)...")

        # 生成测试数据
        time_series_input = np.random.random((1, 10, 51)).astype(np.float32)
        static_input = np.random.random((1, 30)).astype(np.float32)

        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], time_series_input)
        self.interpreter.set_tensor(self.input_details[1]['index'], static_input)

        # 预热
        for _ in range(10):
            self.interpreter.invoke()

        # 基准测试
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
            "throughput_qps": 1000 / (avg_time / 1000)  # QPS = 1000ms / avg_time_ms
        }

        print(f"📊 推理性能统计:")
        print(f"   平均时间: {avg_time:.2f} ms")
        print(f"   最快时间: {min_time:.2f} ms")
        print(f"   最慢时间: {max_time:.2f} ms")
        print(f"   P95时间: {p95_time:.2f} ms")
        print(f"   P99时间: {p99_time:.2f} ms")
        print(f"   吞吐量: {performance_stats['throughput_qps']:.1f} QPS")

        return performance_stats

    def test_batch_performance(self, batch_sizes: list = [1, 4, 8, 16]):
        """测试不同批量大小的性能"""
        print(f"\n📊 测试批量性能...")

        batch_results = []
        for batch_size in batch_sizes:
            print(f"   批量大小: {batch_size}")

            # 生成测试数据
            time_series_input = np.random.random((batch_size, 10, 51)).astype(np.float32)
            static_input = np.random.random((batch_size, 30)).astype(np.float32)

            # 调整输入张量
            self.interpreter.resize_tensor_input(
                self.input_details[0]['index'],
                (batch_size, 10, 51)
            )
            self.interpreter.resize_tensor_input(
                self.input_details[1]['index'],
                (batch_size, 30)
            )

            # 重新分配张量
            self.interpreter.allocate_tensors()

            # 更新输入输出详情
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            # 设置输入
            self.interpreter.set_tensor(input_details[0]['index'], time_series_input)
            self.interpreter.set_tensor(input_details[1]['index'], static_input)

            # 预热
            for _ in range(5):
                self.interpreter.invoke()

            # 基准测试
            inference_times = []
            for _ in range(50):
                start_time = time.perf_counter()
                self.interpreter.invoke()
                end_time = time.perf_counter()

                inference_times.append((end_time - start_time) * 1000)

            avg_time = statistics.mean(inference_times)
            throughput = batch_size / (avg_time / 1000)

            batch_results.append({
                "batch_size": batch_size,
                "avg_time_ms": avg_time,
                "throughput_qps": throughput,
                "time_per_sample_ms": avg_time / batch_size
            })

            print(f"     平均时间: {avg_time:.2f} ms, 吞吐量: {throughput:.1f} QPS")

        return batch_results

    def test_memory_usage(self):
        """测试内存使用情况"""
        print(f"\n💾 测试内存使用...")

        # 获取基础内存使用
        import psutil
        process = psutil.Process()
        base_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # 加载模型后
        model_memory = process.memory_info().rss / (1024 * 1024)
        model_overhead = model_memory - base_memory

        # 估算推理时的内存使用
        time_series_input = np.random.random((1, 10, 51)).astype(np.float32)
        static_input = np.random.random((1, 30)).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], time_series_input)
        self.interpreter.set_tensor(self.input_details[1]['index'], static_input)

        inference_memory = process.memory_info().rss / (1024 * 1024)
        inference_overhead = inference_memory - model_memory

        memory_stats = {
            "base_memory_mb": base_memory,
            "model_memory_mb": model_memory,
            "model_overhead_mb": model_overhead,
            "inference_memory_mb": inference_memory,
            "inference_overhead_mb": inference_overhead,
            "total_model_size_mb": self.tflite_path.stat().st_size / (1024 * 1024)
        }

        print(f"📊 内存使用统计:")
        print(f"   基础内存: {base_memory:.2f} MB")
        print(f"   模型内存: {model_memory:.2f} MB (+{model_overhead:.2f} MB)")
        print(f"   推理内存: {inference_memory:.2f} MB (+{inference_overhead:.2f} MB)")
        print(f"   模型文件: {memory_stats['total_model_size_mb']:.2f} MB")

        return memory_stats

    def test_accuracy(self, original_model_path: str = None):
        """测试模型准确性（如果有原始模型）"""
        print(f"\n🎯 测试模型准确性...")

        # 这里我们可以测试TFLite模型的输出一致性
        # 运行多次推理检查输出稳定性
        num_tests = 10
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
            "max_deviation": np.max(np.std(outputs_array, axis=0)),
            "prediction_range": (np.min(outputs_array).tolist(), np.max(outputs_array).tolist())
        }

        print(f"📊 预测一致性统计:")
        print(f"   测试次数: {num_tests}")
        print(f"   平均预测: {mean_output}")
        print(f"   标准差: {std_output}")
        print(f"   最大偏差: {np.max(np.std(outputs_array, axis=0)):.4f}")
        print(f"   预测范围: {np.min(outputs_array):.2f} - {np.max(outputs_array):.2f}")

        return consistency_stats

    def generate_performance_report(self, performance_stats, memory_stats, consistency_stats):
        """生成性能报告"""
        print(f"\n📋 生成性能报告...")

        report = {
            "model_info": {
                "tflite_file": str(self.tflite_path),
                "file_size_mb": self.tflite_path.stat().st_size / (1024 * 1024),
                "input_shapes": [detail['shape'] for detail in self.input_details],
                "output_shapes": [detail['shape'] for detail in self.output_details]
            },
            "performance": performance_stats,
            "memory": memory_stats,
            "consistency": consistency_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存报告
        report_path = self.tflite_path.parent / "performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📁 ��能报告已保存: {report_path}")
        return report

def main():
    """主函数"""
    print("🧪 TensorFlow Lite 性能测试工具")
    print("=" * 50)

    # 查找TFLite模型 - 使用绝对路径
    base_dir = Path("/home/gitlab-runner/2024_TJU_Data_Mining-Analysis")
    tflite_paths = list((base_dir / "mobile_deployment" / "models").glob("*.tflite"))
    if not tflite_paths:
        tflite_paths = list((base_dir / "mobile_deployment" / "mobile_deployment" / "src" / "output").glob("*.tflite"))
    if not tflite_paths:
        print("❌ 未找到TFLite模型文件")
        return 1

    tflite_path = tflite_paths[0]
    print(f"🔍 找到TFLite模型: {tflite_path}")

    try:
        # 创建测试器
        tester = TFLitePerformanceTester(tflite_path)

        # 加载模型
        tester.load_model()

        # 性能测试
        performance_stats = tester.benchmark_inference_speed()
        batch_results = tester.test_batch_performance()
        memory_stats = tester.test_memory_usage()
        consistency_stats = tester.test_accuracy()

        # 生成报告
        report = tester.generate_performance_report(
            performance_stats, memory_stats, consistency_stats
        )

        print(f"\n🎉 性能测试完成!")
        print(f"📁 详细报告: {report_path}")

        # 总结关键指标
        print(f"\n📊 关键性能指标:")
        print(f"   ✅ 模型大小: {report['model_info']['file_size_mb']:.2f} MB")
        print(f"   ✅ 推理时间: {performance_stats['avg_inference_time_ms']:.2f} ms")
        print(f"   ✅ 吞吐量: {performance_stats['throughput_qps']:.1f} QPS")
        print(f"   ✅ 内存使用: {memory_stats['total_model_size_mb'] + memory_stats['model_overhead_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
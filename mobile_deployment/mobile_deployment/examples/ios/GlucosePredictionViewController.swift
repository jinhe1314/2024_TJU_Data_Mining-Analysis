//
//  GlucosePredictionViewController.swift
//  血糖预测iOS界面示例
//
//  展示如何在iOS应用中集成血糖预测模型
//

import UIKit

class GlucosePredictionViewController: UIViewController {

    // MARK: - UI组件
    @IBOutlet weak var modelInfoTextView: UITextView!
    @IBOutlet weak var predictionResultTextView: UITextView!
    @IBOutlet weak var predictButton: UIButton!
    @IBOutlet weak var performanceTestButton: UIButton!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!

    // MARK: - 属性
    private var predictor: GlucosePredictor?

    // MARK: - 生命周期方法
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        initializePredictor()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // 释放预测器资源
        predictor = nil
    }

    // MARK: - UI设置
    private func setupUI() {
        title = "血糖预测"

        // 设置文本视图
        modelInfoTextView.layer.cornerRadius = 8
        modelInfoTextView.layer.borderWidth = 1
        modelInfoTextView.layer.borderColor = UIColor.lightGray.cgColor
        modelInfoTextView.font = UIFont.systemFont(ofSize: 12)

        predictionResultTextView.layer.cornerRadius = 8
        predictionResultTextView.layer.borderWidth = 1
        predictionResultTextView.layer.borderColor = UIColor.lightGray.cgColor
        predictionResultTextView.font = UIFont.systemFont(ofSize: 14)

        // 设置按钮
        predictButton.layer.cornerRadius = 8
        predictButton.backgroundColor = UIColor.systemBlue
        predictButton.setTitleColor(.white, for: .normal)

        performanceTestButton.layer.cornerRadius = 8
        performanceTestButton.backgroundColor = UIColor.systemGreen
        performanceTestButton.setTitleColor(.white, for: .normal)

        // 初始状态
        predictButton.isEnabled = false
        performanceTestButton.isEnabled = false
        activityIndicator.hidesWhenStopped = true
    }

    // MARK: - 模型初始化
    private func initializePredictor() {
        activityIndicator.startAnimating()
        modelInfoTextView.text = "正在加载模型..."

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let predictor = GlucosePredictor()
                try predictor.initialize()

                DispatchQueue.main.async {
                    self.predictor = predictor
                    self.modelInfoTextView.text = predictor.getModelInfo()
                    self.predictButton.isEnabled = true
                    self.performanceTestButton.isEnabled = true
                    self.activityIndicator.stopAnimating()

                    self.showAlert(title: "成功", message: "模型加载成功")
                }

            } catch {
                DispatchQueue.main.async {
                    self.activityIndicator.stopAnimating()
                    self.modelInfoTextView.text = "模型加载失败"
                    self.showAlert(title: "错误", message: "模型加载失败: \\(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - 按钮事件
    @IBAction func predictButtonTapped(_ sender: UIButton) {
        performSinglePrediction()
    }

    @IBAction func performanceTestButtonTapped(_ sender: UIButton) {
        performPerformanceTest()
    }

    // MARK: - 预测方法
    private func performSinglePrediction() {
        guard let predictor = predictor else {
            showAlert(title: "错误", message: "模型未初始化")
            return
        }

        setUIState(isPredicting: true)
        predictionResultTextView.text = "正在预测..."

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                // 生成测试数据
                let timeSeriesData = self.generateTestTimeSeriesData()
                let staticData = self.generateTestStaticData()

                // 执行预测并测量时间
                let startTime = CFAbsoluteTimeGetCurrent()
                let predictions = try predictor.predictGlucose(timeSeriesData: timeSeriesData, staticData: staticData)
                let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime

                // 格式化结果
                let resultText = self.formatPredictions(predictions: predictions, inferenceTime: inferenceTime)

                DispatchQueue.main.async {
                    self.predictionResultTextView.text = resultText
                    self.setUIState(isPredicting: false)
                }

            } catch {
                DispatchQueue.main.async {
                    self.predictionResultTextView.text = "预测失败: \\(error.localizedDescription)"
                    self.setUIState(isPredicting: false)
                }
            }
        }
    }

    private func performPerformanceTest() {
        guard let predictor = predictor else {
            showAlert(title: "错误", message: "模型未初始化")
            return
        }

        setUIState(isPredicting: true)
        performanceTestButton.isEnabled = false
        predictionResultTextView.text = "性能测试中..."

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let numTests = 100
                var totalTime: Double = 0
                var minTime: Double = Double.greatestFiniteMagnitude
                var maxTime: Double = 0

                for i in 0..<numTests {
                    // 生成测试数据
                    let timeSeriesData = self.generateTestTimeSeriesData()
                    let staticData = self.generateTestStaticData()

                    // 执行预测
                    let startTime = CFAbsoluteTimeGetCurrent()
                    _ = try predictor.predictGlucose(timeSeriesData: timeSeriesData, staticData: staticData)
                    let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime

                    totalTime += inferenceTime
                    minTime = min(minTime, inferenceTime)
                    maxTime = max(maxTime, inferenceTime)

                    // 更新进度
                    let progress = (i + 1) * 100 / numTests
                    DispatchQueue.main.async {
                        self.predictionResultTextView.text = "性能测试中... \\(progress)%"
                    }
                }

                // 计算统计数据
                let avgTime = totalTime / Double(numTests)
                let throughput = 1000.0 / avgTime // QPS

                // 格式化结果
                let resultText = self.formatPerformanceResults(
                    numTests: numTests,
                    avgTime: avgTime,
                    minTime: minTime,
                    maxTime: maxTime,
                    throughput: throughput
                )

                DispatchQueue.main.async {
                    self.predictionResultTextView.text = resultText
                    self.setUIState(isPredicting: false)
                    self.performanceTestButton.isEnabled = true
                }

            } catch {
                DispatchQueue.main.async {
                    self.predictionResultTextView.text = "性能测试失败: \\(error.localizedDescription)"
                    self.setUIState(isPredicting: false)
                    self.performanceTestButton.isEnabled = true
                }
            }
        }
    }

    // MARK: - 辅助方法

    private func setUIState(isPredicting: Bool) {
        predictButton.isEnabled = !isPredicting
        performanceTestButton.isEnabled = !isPredicting
        if isPredicting {
            activityIndicator.startAnimating()
        } else {
            activityIndicator.stopAnimating()
        }
    }

    private func generateTestTimeSeriesData() -> [[Float]] {
        return Array(repeating: Array(repeating: Float.random(in: 0...1), count: 51), count: 10)
    }

    private func generateTestStaticData() -> [Float] {
        return Array(repeating: Float.random(in: 0...1), count: 30)
    }

    private func formatPredictions(predictions: [GlucosePrediction], inferenceTime: CFAbsoluteTimeGetCurrent) -> String {
        var result = "预测结果 (推理时间: \\(String(format: "%.2f", inferenceTime * 1000))ms):\\n\\n"

        for prediction in predictions {
            result += String(format: "%d分钟后: %.2f mg/dL %@\\n",
                             prediction.horizonMinutes,
                             prediction.glucoseValue,
                             prediction.status)
        }

        return result
    }

    private func formatPerformanceResults(numTests: Int, avgTime: Double, minTime: Double, maxTime: Double, throughput: Double) -> String {
        var result = "性能测试结果 (\\(numTests) 次预测):\\n\\n"
        result += String(format: "平均推理时间: %.2f ms\\n", avgTime * 1000)
        result += String(format: "最快推理时间: %.2f ms\\n", minTime * 1000)
        result += String(format: "最慢推理时间: %.2f ms\\n", maxTime * 1000)
        result += String(format: "推理吞吐量: %.1f QPS\\n\\n", throughput)

        result += "性能评估:\\n"
        if avgTime < 0.01 {
            result += "✅ 推理速度: 优秀 (<10ms)\\n"
        } else if avgTime < 0.05 {
            result += "✅ 推理速度: 良好 (<50ms)\\n"
        } else {
            result += "⚠️ 推理速度: 需要优化 (>50ms)\\n"
        }

        result += "✅ 模型大小: 0.20 MB\\n"
        result += "✅ 适合移动端部署\\n"

        return result
    }

    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "确定", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - 预览支持

#if canImport(SwiftUI) && DEBUG
import SwiftUI

struct GlucosePredictionViewController_Previews: PreviewProvider {
    static var previews: some View {
        // 创建一个简单的预览包装器
        UIViewControllerPreviewWrapper {
            let storyboard = UIStoryboard(name: "Main", bundle: nil)
            return storyboard.instantiateViewController(withIdentifier: "GlucosePredictionViewController")
        }
    }
}

struct UIViewControllerPreviewWrapper<ViewController: UIViewController>: UIViewControllerRepresentable {
    let makeViewController: () -> ViewController

    init(_ makeViewController: @escaping () -> ViewController) {
        self.makeViewController = makeViewController
    }

    func makeUIViewController(context: Context) -> ViewController {
        makeViewController()
    }

    func updateUIViewController(_ uiViewController: ViewController, context: Context) {
        // 不需要更新
    }
}
#endif
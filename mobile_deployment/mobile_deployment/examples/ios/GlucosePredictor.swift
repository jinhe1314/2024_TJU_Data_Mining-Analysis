//
//  GlucosePredictor.swift
//  血糖预测模型 - iOS TensorFlow Lite集成示例
//
//  基于LSTM + Cross-Attention架构的血糖水平预测模型
//
//  功能：
//  - 预测未来15、30、45、60分钟的血糖水平
//  - 输入：10个时间步的历史数据(51特征) + 静态患者特征(30特征)
//  - 输出：4个时间点的血糖预测值
//

import Foundation
import TensorFlowLite
import UIKit

/**
 * 血糖预测结果
 */
struct GlucosePrediction {
    let horizonMinutes: Int    // 预测时间点(分钟)
    let glucoseValue: Float    // 血糖预测值
    let status: String         // 血糖状态
}

/**
 * 血糖预测器
 */
class GlucosePredictor {
    // MARK: - 常量
    private static let modelName = "glucose_predictor"
    private static let timeSeriesLength = 10
    private static let timeSeriesFeatures = 51
    private static let staticFeatures = 30
    private static let outputLength = 4

    // 预测时间点（分钟）
    private static let predictionHorizons = [15, 30, 45, 60]

    // MARK: - 属性
    private var interpreter: Interpreter?
    private var isInitialized = false

    // MARK: - 初始化

    /**
     * 初始化模型
     * - Throws: 模型加载错误
     */
    func initialize() throws {
        guard !isInitialized else { return }

        // 构建模型路径
        guard let modelPath = Bundle.main.path(
            forResource: GlucosePredictor.modelName,
            ofType: "tflite"
        ) else {
            throw PredictorError.modelNotFound("无法找到模型文件: \(GlucosePredictor.modelName).tflite")
        }

        do {
            // 创建解释器
            interpreter = try Interpreter(modelPath: modelPath)

            // 分配张量
            try interpreter?.allocateTensors()

            isInitialized = true

            // 输出模型信息
            print("血糖预测模型初始化成功")
            print(getModelInfo())

        } catch {
            throw PredictorError.initializationFailed("模型初始化失败: \(error.localizedDescription)")
        }
    }

    // MARK: - 预测方法

    /**
     * 预测血糖水平
     * - Parameters:
     *   - timeSeriesData: 10个时间步的历史数据 [10][51]
     *   - staticData: 静态患者特征 [30]
     * - Returns: 预测结果数组
     * - Throws: 预测错误
     */
    func predictGlucose(timeSeriesData: [[Float]], staticData: [Float]) throws -> [GlucosePrediction] {
        guard isInitialized else {
            throw PredictorError.modelNotInitialized("模型未初始化，请先调用initialize()")
        }

        // 验证输入数据维度
        try validateInputData(timeSeriesData: timeSeriesData, staticData: staticData)

        // 准备输入数据
        let timeSeriesDataFlat = timeSeriesData.flatMap { $0 }
        let staticDataCopy = staticData

        // 创建输入数据数组
        var inputData: Data

        // 时序数据 (Float32)
        guard let timeSeriesBytes = timeSeriesDataFlat.withUnsafeBytes({ Data($0) }) else {
            throw PredictorError.dataProcessingFailed("时序数据转换失败")
        }

        // 静态数据 (Float32)
        guard let staticBytes = staticDataCopy.withUnsafeBytes({ Data($0) }) else {
            throw PredictorError.dataProcessingFailed("静态数据转换失败")
        }

        // 合并输入数据
        inputData = timeSeriesBytes + staticBytes

        // 准备输出数据
        var outputData = Data(count: GlucosePredictor.outputLength * MemoryLayout<Float>.size)

        // 执行推理
        guard let interpreter = interpreter else {
            throw PredictorError.modelNotInitialized("解释器未初始化")
        }

        // 设置输入
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.copy(staticBytes, toInputAt: 1)

        // 执行推理
        try interpreter.invoke()

        // 获取输出
        try interpreter.copy(&outputData, fromOutputAt: 0)

        // 解析输出结果
        let predictions = try parseOutputData(outputData)

        return predictions
    }

    /**
     * 批量预测（用于性能测试）
     * - Parameter batchData: 批量输入数据
     * - Returns: 批量预测结果
     * - Throws: 预测错误
     */
    func predictBatch(batchData: (timeSeries: [[[Float]]], static: [[Float]])) throws -> [[GlucosePrediction]] {
        let batchSize = batchData.timeSeries.count
        var results: [[GlucosePrediction]] = []

        for i in 0..<batchSize {
            let prediction = try predictGlucose(
                timeSeriesData: batchData.timeSeries[i],
                staticData: batchData.static[i]
            )
            results.append(prediction)
        }

        return results
    }

    // MARK: - 模型信息

    /**
     * 获取模型信息
     * - Returns: 模型信息字符串
     */
    func getModelInfo() -> String {
        guard isInitialized, let interpreter = interpreter else {
            return "模型未初始化"
        }

        var info = "血糖预测模型信息:\\n"
        info += "- 模型类型: LSTM + Cross-Attention\\n"
        info += "- 输入: 时序数据[10,51] + 静态特征[30]\\n"
        info += "- 输出: 血糖预测[4] (15/30/45/60分钟)\\n"

        // 获取输入输出张量信息
        let inputTensorCount = interpreter.inputTensorCount
        let outputTensorCount = interpreter.outputTensorCount

        info += String(format: "- 输入数量: %d\\n", inputTensorCount)
        for i in 0..<inputTensorCount {
            let tensor = try? interpreter.input(at: i)
            if let tensor = tensor {
                info += String(format: "  输入%d: %s (%s)\\n",
                              i + 1,
                              tensor.shape.description,
                              tensor.dataType.description)
            }
        }

        info += String(format: "- 输出数量: %d\\n", outputTensorCount)
        for i in 0..<outputTensorCount {
            let tensor = try? interpreter.output(at: i)
            if let tensor = tensor {
                info += String(format: "  输出%d: %s (%s)\\n",
                              i + 1,
                              tensor.shape.description,
                              tensor.dataType.description)
            }
        }

        return info
    }

    // MARK: - 私有方法

    /**
     * 验证输入数据维度
     */
    private func validateInputData(timeSeriesData: [[Float]], staticData: [Float]) throws {
        if timeSeriesData.count != GlucosePredictor.timeSeriesLength {
            throw PredictorError.invalidInput(
                "时序数据长度错误，期望\\(GlucosePredictor.timeSeriesLength)，实际\\(timeSeriesData.count)"
            )
        }

        for (i, timeStep) in timeSeriesData.enumerated() {
            if timeStep.count != GlucosePredictor.timeSeriesFeatures {
                throw PredictorError.invalidInput(
                    "时序数据特征数错误，期望\\(GlucosePredictor.timeSeriesFeatures)，实际\\(timeStep.count) (第\\(i)步)"
                )
            }
        }

        if staticData.count != GlucosePredictor.staticFeatures {
            throw PredictorError.invalidInput(
                "静态特征数错误，期望\\(GlucosePredictor.staticFeatures)，实际\\(staticData.count)"
            )
        }
    }

    /**
     * 解析输出数据
     */
    private func parseOutputData(_ data: Data) throws -> [GlucosePrediction] {
        var predictions: [GlucosePrediction] = []

        let predictionsArray = data.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(to: Float.self))
        }

        guard predictionsArray.count == GlucosePredictor.outputLength else {
            throw PredictorError.dataProcessingFailed("输出数据长度错误")
        }

        for (i, glucoseValue) in predictionsArray.enumerated() {
            let horizonMinutes = GlucosePredictor.predictionHorizons[i]
            let status = getGlucoseStatus(glucoseValue)

            let prediction = GlucosePrediction(
                horizonMinutes: horizonMinutes,
                glucoseValue: glucoseValue,
                status: status
            )

            predictions.append(prediction)
        }

        return predictions
    }

    /**
     * 获取血糖状态
     */
    private func getGlucoseStatus(_ glucoseLevel: Float) -> String {
        // 将标准化值转换为mmol/L (假设标准化范围为3-15)
        let glucoseMMOL = glucoseLevel * 12 + 3

        switch glucoseMMOL {
        case 0..<3.9:
            return "低血糖 ⚠️"
        case 3.9..<5.6:
            return "正常 ✅"
        case 5.6..<6.9:
            return "轻度升高 ⚠️"
        case 6.9..<11.1:
            return "中度升高 ⚠️"
        default:
            return "高血糖 ❌"
        }
    }

    deinit {
        // 释放资源
        interpreter = nil
        isInitialized = false
    }
}

// MARK: - 错误类型

enum PredictorError: Error, LocalizedError {
    case modelNotFound(String)
    case initializationFailed(String)
    case modelNotInitialized(String)
    case invalidInput(String)
    case dataProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let message):
            return "模型文件错误: \\(message)"
        case .initializationFailed(let message):
            return "初始化失败: \\(message)"
        case .modelNotInitialized(let message):
            return "模型未初始化: \\(message)"
        case .invalidInput(let message):
            return "输入数据错误: \\(message)"
        case .dataProcessingFailed(let message):
            return "数据处理错误: \\(message)"
        }
    }
}

// MARK: - 使用示例

extension GlucosePredictor {
    /**
     * 使用示例代码
     */
    static func usageExample() {
        let predictor = GlucosePredictor()

        do {
            // 初始化模型
            try predictor.initialize()

            // 准备测试数据
            let timeSeriesData: [[Float]] = Array(repeating: Array(repeating: Float.random(in: 0...1), count: 51), count: 10)
            let staticData: [Float] = Array(repeating: Float.random(in: 0...1), count: 30)

            // 执行预测
            let predictions = try predictor.predictGlucose(timeSeriesData: timeSeriesData, staticData: staticData)

            // 输出结果
            print("血糖预测结果:")
            for prediction in predictions {
                print(String(format: "%d分钟后: %.2f mg/dL - %@",
                              prediction.horizonMinutes,
                              prediction.glucoseValue,
                              prediction.status))
            }

        } catch {
            print("预测失败: \\(error.localizedDescription)")
        }
    }
}
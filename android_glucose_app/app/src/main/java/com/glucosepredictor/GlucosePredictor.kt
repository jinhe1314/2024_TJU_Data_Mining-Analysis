package com.glucosepredictor

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * 血糖预测器 - 基于TFLite模型
 * 对应Python demo3_tflite_model.py的逻辑
 */
class GlucosePredictor(context: Context) {

    private var interpreter: Interpreter? = null
    private val tag = "GlucosePredictor"

    // 模型输入输出规格
    private val timeSeriesShape = intArrayOf(1, 10, 51)  // (batch, timesteps, features)
    private val staticShape = intArrayOf(1, 30)          // (batch, features)
    private val outputShape = intArrayOf(1, 4)           // (batch, predictions)

    // 标准化参数 (需要从训练数据计算，这里使用示例值)
    private val tsXMean = FloatArray(51) // 时序特征均值
    private val tsXStd = FloatArray(51)  // 时序特征标准差
    private val staticMean = FloatArray(30) // 静态特征均值
    private val staticStd = FloatArray(30)  // 静态特征标准差
    private val yMean = FloatArray(4)    // 目标均值
    private val yStd = FloatArray(4)     // 目标标准差

    init {
        try {
            // 加载TFLite模型
            val modelFile = loadModelFile(context, "glucose_predictor.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            interpreter = Interpreter(modelFile, options)

            // 初始化标准化参数（实际应用中应从文件加载）
            initScalerParams()

            Log.d(tag, "✓ 模型加载成功")
        } catch (e: Exception) {
            Log.e(tag, "模型加载失败: ${e.message}")
            e.printStackTrace()
        }
    }

    /**
     * 加载TFLite模型文件
     */
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * 初始化标准化参数
     * 注意：实际应用中应从训练时保存的参数文件加载
     */
    private fun initScalerParams() {
        // 这里使用简化的初始化，实际应该从JSON文件加载
        for (i in tsXMean.indices) {
            tsXMean[i] = 0f
            tsXStd[i] = 1f
        }
        for (i in staticMean.indices) {
            staticMean[i] = 0f
            staticStd[i] = 1f
        }
        for (i in yMean.indices) {
            yMean[i] = 130f  // 血糖均值约130 mg/dL
            yStd[i] = 30f    // 标准差约30 mg/dL
        }
    }

    /**
     * 标准化时序数据
     */
    private fun standardizeTimeSeries(data: FloatArray): FloatArray {
        val result = FloatArray(data.size)
        for (i in data.indices) {
            val featureIdx = i % 51
            result[i] = if (tsXStd[featureIdx] != 0f) {
                (data[i] - tsXMean[featureIdx]) / tsXStd[featureIdx]
            } else {
                0f
            }
        }
        return result
    }

    /**
     * 标准化静态数据
     */
    private fun standardizeStatic(data: FloatArray): FloatArray {
        val result = FloatArray(data.size)
        for (i in data.indices) {
            result[i] = if (staticStd[i] != 0f) {
                (data[i] - staticMean[i]) / staticStd[i]
            } else {
                0f
            }
        }
        return result
    }

    /**
     * 反标准化预测结果
     */
    private fun inverseStandardizePredictions(data: FloatArray): FloatArray {
        val result = FloatArray(data.size)
        for (i in data.indices) {
            result[i] = data[i] * yStd[i] + yMean[i]
        }
        return result
    }

    /**
     * 执行预测
     */
    fun predict(timeSeriesData: FloatArray, staticData: FloatArray): FloatArray? {
        try {
            // 标准化输入
            val normalizedTS = standardizeTimeSeries(timeSeriesData)
            val normalizedStatic = standardizeStatic(staticData)

            // 准备输入ByteBuffer
            val tsBuffer = ByteBuffer.allocateDirect(4 * timeSeriesShape[1] * timeSeriesShape[2])
                .order(ByteOrder.nativeOrder())
            normalizedTS.forEach { tsBuffer.putFloat(it) }

            val staticBuffer = ByteBuffer.allocateDirect(4 * staticShape[1])
                .order(ByteOrder.nativeOrder())
            normalizedStatic.forEach { staticBuffer.putFloat(it) }

            // 准备输出
            val outputBuffer = ByteBuffer.allocateDirect(4 * outputShape[1])
                .order(ByteOrder.nativeOrder())

            // 执行推理
            val inputs = arrayOf<Any>(tsBuffer, staticBuffer)
            val outputs = mapOf(0 to outputBuffer)

            interpreter?.runForMultipleInputsOutputs(inputs, outputs)

            // 解析输出
            outputBuffer.rewind()
            val predictions = FloatArray(4)
            outputBuffer.asFloatBuffer().get(predictions)

            // 反标准化
            return inverseStandardizePredictions(predictions)

        } catch (e: Exception) {
            Log.e(tag, "预测失败: ${e.message}")
            e.printStackTrace()
            return null
        }
    }

    /**
     * 预测所有场景
     */
    fun predictAllScenarios(
        timeSeriesData: FloatArray,
        staticData: FloatArray
    ): PredictionResult {
        // 场景1: 完整输入
        val predFull = predict(timeSeriesData, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        // 场景2: 无患者信息
        val staticZero = FloatArray(30) { 0f }
        val predNoStatic = predict(timeSeriesData, staticZero) ?: floatArrayOf(0f, 0f, 0f, 0f)

        // 场景3: 普通进餐 (Dietary intake = 1)
        val tsMeal = timeSeriesData.copyOf()
        tsMeal[tsMeal.size - 51 + 1] = 1.0f  // 最后一个时间步的Dietary intake索引
        val predMeal = predict(tsMeal, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        // 场景4: 高热量进餐 (Dietary intake = 3)
        val tsHighMeal = timeSeriesData.copyOf()
        tsHighMeal[tsHighMeal.size - 51 + 1] = 3.0f
        val predHighMeal = predict(tsHighMeal, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        return PredictionResult(
            fullInput = predFull,
            noPatientInfo = predNoStatic,
            normalMeal = predMeal,
            highCalorieMeal = predHighMeal
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * 预测结果数据类
 */
data class PredictionResult(
    val fullInput: FloatArray,           // 完整输入预测
    val noPatientInfo: FloatArray,       // 无患者信息预测
    val normalMeal: FloatArray,          // 普通进餐预测
    val highCalorieMeal: FloatArray      // 高热量进餐预测
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PredictionResult

        if (!fullInput.contentEquals(other.fullInput)) return false
        if (!noPatientInfo.contentEquals(other.noPatientInfo)) return false
        if (!normalMeal.contentEquals(other.normalMeal)) return false
        if (!highCalorieMeal.contentEquals(other.highCalorieMeal)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = fullInput.contentHashCode()
        result = 31 * result + noPatientInfo.contentHashCode()
        result = 31 * result + normalMeal.contentHashCode()
        result = 31 * result + highCalorieMeal.contentHashCode()
        return result
    }
}

/**
 * 患者数据类
 */
data class PatientData(
    val patientId: String,
    val historicalGlucose: FloatArray,  // 历史血糖值 (9个点)
    val timeSeriesFeatures: FloatArray, // 时序特征 (10 x 51 = 510)
    val staticFeatures: FloatArray       // 静态特征 (30)
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PatientData

        if (patientId != other.patientId) return false
        if (!historicalGlucose.contentEquals(other.historicalGlucose)) return false
        if (!timeSeriesFeatures.contentEquals(other.timeSeriesFeatures)) return false
        if (!staticFeatures.contentEquals(other.staticFeatures)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = patientId.hashCode()
        result = 31 * result + historicalGlucose.contentHashCode()
        result = 31 * result + timeSeriesFeatures.contentHashCode()
        result = 31 * result + staticFeatures.contentHashCode()
        return result
    }
}

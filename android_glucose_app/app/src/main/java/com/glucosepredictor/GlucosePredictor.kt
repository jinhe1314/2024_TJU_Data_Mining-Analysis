package com.glucosepredictor

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
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

            // 从JSON文件加载标准化参数
            loadScalerParams(context)

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
     * 从JSON文件加载标准化参数
     */
    private fun loadScalerParams(context: Context) {
        try {
            // 读取JSON文件
            val jsonString = context.assets.open("scaler_params.json").bufferedReader().use { it.readText() }
            val json = JSONObject(jsonString)

            // 加载时序特征参数
            val tsJson = json.getJSONObject("time_series")
            val tsMeanArray = tsJson.getJSONArray("mean")
            val tsStdArray = tsJson.getJSONArray("std")
            for (i in 0 until tsMeanArray.length()) {
                tsXMean[i] = tsMeanArray.getDouble(i).toFloat()
                tsXStd[i] = tsStdArray.getDouble(i).toFloat()
            }

            // 加载静态特征参数
            val staticJson = json.getJSONObject("static")
            val staticMeanArray = staticJson.getJSONArray("mean")
            val staticStdArray = staticJson.getJSONArray("std")
            for (i in 0 until staticMeanArray.length()) {
                staticMean[i] = staticMeanArray.getDouble(i).toFloat()
                staticStd[i] = staticStdArray.getDouble(i).toFloat()
            }

            // 加载目标参数
            val targetJson = json.getJSONObject("target")
            val targetMeanArray = targetJson.getJSONArray("mean")
            val targetStdArray = targetJson.getJSONArray("std")
            for (i in 0 until targetMeanArray.length()) {
                yMean[i] = targetMeanArray.getDouble(i).toFloat()
                yStd[i] = targetStdArray.getDouble(i).toFloat()
            }

            Log.d(tag, "✓ Scaler参数加载成功")
            Log.d(tag, "  时序: mean[0]=${tsXMean[0]}, std[0]=${tsXStd[0]}")
            Log.d(tag, "  目标: mean[0]=${yMean[0]}, std[0]=${yStd[0]}")

        } catch (e: Exception) {
            Log.e(tag, "Scaler参数加载失败: ${e.message}")
            e.printStackTrace()

            // 使用默认值作为后备
            for (i in tsXMean.indices) {
                tsXMean[i] = 0f
                tsXStd[i] = 1f
            }
            for (i in staticMean.indices) {
                staticMean[i] = 0f
                staticStd[i] = 1f
            }
            for (i in yMean.indices) {
                yMean[i] = 130f
                yStd[i] = 30f
            }
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
     * 基于demo3_tflite_model.py的时间模拟策略
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

        // 场景3: 低热量进餐 (30分钟前轻食)
        // Dietary intake索引=1，在倒数第3个时间步（索引-3）设置为1
        val tsLowMeal = timeSeriesData.copyOf()
        val dietaryIdxLow = tsLowMeal.size - 3 * 51 + 1  // 倒数第3个时间步的Dietary intake
        tsLowMeal[dietaryIdxLow] = 1.0f
        val predLowMeal = predict(tsLowMeal, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        // 场景4: 中热量进餐 (15分钟前正常进餐)
        // Dietary intake索引=1，在倒数第2个时间步（索引-2）设置为1
        val tsMidMeal = timeSeriesData.copyOf()
        val dietaryIdxMid = tsMidMeal.size - 2 * 51 + 1  // 倒数第2个时间步的Dietary intake
        tsMidMeal[dietaryIdxMid] = 1.0f
        val predMidMeal = predict(tsMidMeal, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        // 场景5: 高热量进餐 (持续进餐，30-15分钟前)
        // 在倒数第3和倒数第2个时间步都设置Dietary intake=1
        val tsHighMeal = timeSeriesData.copyOf()
        tsHighMeal[dietaryIdxLow] = 1.0f  // 30分钟前开始进餐
        tsHighMeal[dietaryIdxMid] = 1.0f  // 15分钟前仍在进餐
        val predHighMeal = predict(tsHighMeal, staticData) ?: floatArrayOf(0f, 0f, 0f, 0f)

        return PredictionResult(
            fullInput = predFull,
            noPatientInfo = predNoStatic,
            lowCalorieMeal = predLowMeal,
            mediumCalorieMeal = predMidMeal,
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
    val lowCalorieMeal: FloatArray,      // 低热量进餐预测 (30分钟前轻食)
    val mediumCalorieMeal: FloatArray,   // 中热量进餐预测 (15分钟前正常进餐)
    val highCalorieMeal: FloatArray      // 高热量进餐预测 (持续大餐)
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PredictionResult

        if (!fullInput.contentEquals(other.fullInput)) return false
        if (!noPatientInfo.contentEquals(other.noPatientInfo)) return false
        if (!lowCalorieMeal.contentEquals(other.lowCalorieMeal)) return false
        if (!mediumCalorieMeal.contentEquals(other.mediumCalorieMeal)) return false
        if (!highCalorieMeal.contentEquals(other.highCalorieMeal)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = fullInput.contentHashCode()
        result = 31 * result + noPatientInfo.contentHashCode()
        result = 31 * result + lowCalorieMeal.contentHashCode()
        result = 31 * result + mediumCalorieMeal.contentHashCode()
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

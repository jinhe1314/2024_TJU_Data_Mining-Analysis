package com.glucosepredictor

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * MainActivity - 血糖预测应用主界面
 * 对应Python demo3_tflite_model.py的可视化逻辑
 */
class MainActivity : AppCompatActivity() {

    private lateinit var chart: LineChart
    private lateinit var resultText: TextView
    private lateinit var predictButton: Button
    private lateinit var patientInfoText: TextView

    private var predictor: GlucosePredictor? = null
    private val tag = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化视图
        chart = findViewById(R.id.chart)
        resultText = findViewById(R.id.resultText)
        predictButton = findViewById(R.id.predictButton)
        patientInfoText = findViewById(R.id.patientInfoText)

        // 初始化预测器
        predictor = GlucosePredictor(this)

        // 设置图表
        setupChart()

        // 显示示例患者信息
        displayPatientInfo()

        // 预测按钮点击事件
        predictButton.setOnClickListener {
            performPrediction()
        }

        // 自动执行一次预测
        performPrediction()
    }

    /**
     * 设置图表样式
     */
    private fun setupChart() {
        chart.apply {
            description.isEnabled = true
            description.text = "Blood Glucose Prediction"
            description.textSize = 12f

            setDrawGridBackground(false)
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)

            // X轴设置
            xAxis.apply {
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(true)
                granularity = 15f
                valueFormatter = object : ValueFormatter() {
                    override fun getFormattedValue(value: Float): String {
                        return "${value.toInt()} min"
                    }
                }
                textSize = 10f
            }

            // 左Y轴设置
            axisLeft.apply {
                setDrawGridLines(true)
                axisMinimum = 70f
                axisMaximum = 200f
                textSize = 10f
            }

            // 右Y轴禁用
            axisRight.isEnabled = false

            // 图例设置
            legend.apply {
                isEnabled = true
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                horizontalAlignment = Legend.LegendHorizontalAlignment.LEFT
                orientation = Legend.LegendOrientation.VERTICAL
                setDrawInside(true)
                textSize = 9f
                xEntrySpace = 5f
                yEntrySpace = 2f
            }
        }
    }

    /**
     * 显示患者信息
     */
    private fun displayPatientInfo() {
        val info = """
            |Patient ID: 2035_0_20210629
            |Gender: Male | Age: 78y | BMI: 24.3
            |Type: T1DM | Duration: 20y
            |HbA1c: 100.0 mmol/mol | FPG: 133.2 mg/dL
            |
            |Input: 10 timesteps × 51 features + 30 static features
            |Model: TFLite (202KB, optimized for mobile)
        """.trimMargin()

        patientInfoText.text = info
    }

    /**
     * 执行预测
     */
    private fun performPrediction() {
        predictButton.isEnabled = false
        resultText.text = "预测中..."

        CoroutineScope(Dispatchers.Default).launch {
            try {
                // 示例患者数据 (患者2035_0_20210629的前9个数据点)
                val patientData = getSamplePatientData()

                // 执行预测
                val result = predictor?.predictAllScenarios(
                    patientData.timeSeriesFeatures,
                    patientData.staticFeatures
                )

                withContext(Dispatchers.Main) {
                    if (result != null) {
                        // 更新图表
                        updateChart(patientData.historicalGlucose, result)

                        // 显示预测结果
                        displayResults(result)
                    } else {
                        resultText.text = "预测失败"
                    }
                    predictButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(tag, "预测出错: ${e.message}")
                withContext(Dispatchers.Main) {
                    resultText.text = "预测出错: ${e.message}"
                    predictButton.isEnabled = true
                }
            }
        }
    }

    /**
     * 更新图表
     */
    private fun updateChart(historicalGlucose: FloatArray, predictions: PredictionResult) {
        val dataSets = ArrayList<LineDataSet>()

        // 1. 历史血糖数据 (0-120分钟)
        val historicalEntries = ArrayList<Entry>()
        for (i in historicalGlucose.indices) {
            historicalEntries.add(Entry((i * 15).toFloat(), historicalGlucose[i]))
        }
        val historicalDataSet = LineDataSet(historicalEntries, "Historical CGM").apply {
            color = Color.BLUE
            setCircleColor(Color.BLUE)
            lineWidth = 2f
            circleRadius = 4f
            setDrawCircleHole(true)
            setDrawValues(true)
            valueTextSize = 8f
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(historicalDataSet)

        // 预测时间点 (135, 150, 165, 180分钟)
        val predictionTimes = floatArrayOf(135f, 150f, 165f, 180f)

        // 2. 完整输入预测 (红线)
        val fullEntries = ArrayList<Entry>()
        for (i in predictions.fullInput.indices) {
            fullEntries.add(Entry(predictionTimes[i], predictions.fullInput[i]))
        }
        val fullDataSet = LineDataSet(fullEntries, "With Patient Info").apply {
            color = Color.RED
            setCircleColor(Color.RED)
            lineWidth = 2f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(true)
            valueTextSize = 8f
            valueTextColor = Color.RED
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(fullDataSet)

        // 3. 无患者信息预测 (绿线)
        val noStaticEntries = ArrayList<Entry>()
        for (i in predictions.noPatientInfo.indices) {
            noStaticEntries.add(Entry(predictionTimes[i], predictions.noPatientInfo[i]))
        }
        val noStaticDataSet = LineDataSet(noStaticEntries, "Without Patient Info").apply {
            color = Color.GREEN
            setCircleColor(Color.GREEN)
            lineWidth = 2f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(true)
            valueTextSize = 8f
            valueTextColor = Color.GREEN
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(noStaticDataSet)

        // 4. 普通进餐预测 (橙线)
        val mealEntries = ArrayList<Entry>()
        for (i in predictions.normalMeal.indices) {
            mealEntries.add(Entry(predictionTimes[i], predictions.normalMeal[i]))
        }
        val mealDataSet = LineDataSet(mealEntries, "Normal Meal (intake=1)").apply {
            color = Color.rgb(255, 165, 0) // Orange
            setCircleColor(Color.rgb(255, 165, 0))
            lineWidth = 2f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(true)
            valueTextSize = 8f
            valueTextColor = Color.rgb(255, 140, 0)
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(mealDataSet)

        // 5. 高热量进餐预测 (紫线)
        val highMealEntries = ArrayList<Entry>()
        for (i in predictions.highCalorieMeal.indices) {
            highMealEntries.add(Entry(predictionTimes[i], predictions.highCalorieMeal[i]))
        }
        val highMealDataSet = LineDataSet(highMealEntries, "High-Calorie Meal (intake=3)").apply {
            color = Color.rgb(128, 0, 128) // Purple
            setCircleColor(Color.rgb(128, 0, 128))
            lineWidth = 2f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(true)
            valueTextSize = 8f
            valueTextColor = Color.rgb(128, 0, 128)
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(highMealDataSet)

        // 设置数据
        val lineData = LineData(dataSets.map { it as com.github.mikephil.charting.interfaces.datasets.ILineDataSet })
        chart.data = lineData
        chart.invalidate()
    }

    /**
     * 显示预测结果文本
     */
    private fun displayResults(predictions: PredictionResult) {
        val result = StringBuilder()
        result.append("预测结果 (15/30/45/60 分钟):\n\n")

        result.append("完整输入:\n")
        result.append(formatPredictions(predictions.fullInput))
        result.append("\n")

        result.append("无患者信息:\n")
        result.append(formatPredictions(predictions.noPatientInfo))
        result.append("\n")

        result.append("普通进餐:\n")
        result.append(formatPredictions(predictions.normalMeal))
        result.append("\n")

        result.append("高热量进餐:\n")
        result.append(formatPredictions(predictions.highCalorieMeal))
        result.append("\n")

        // 影响分析
        result.append("\n影响分析:\n")
        val mealImpact = FloatArray(4) { predictions.normalMeal[it] - predictions.fullInput[it] }
        val highMealImpact = FloatArray(4) { predictions.highCalorieMeal[it] - predictions.fullInput[it] }

        result.append("普通进餐影响: ${formatImpact(mealImpact)}\n")
        result.append("高热量进餐影响: ${formatImpact(highMealImpact)}\n")

        resultText.text = result.toString()
    }

    /**
     * 格式化预测值
     */
    private fun formatPredictions(values: FloatArray): String {
        return values.joinToString(" → ") { "%.1f".format(it) } + " mg/dL"
    }

    /**
     * 格式化影响值
     */
    private fun formatImpact(values: FloatArray): String {
        return values.joinToString(", ") { "%+.1f".format(it) } + " mg/dL"
    }

    /**
     * 获取示例患者数据
     */
    private fun getSamplePatientData(): PatientData {
        // 患者2035_0_20210629的真实数据
        val historicalGlucose = floatArrayOf(
            142.2f, 158.4f, 172.8f, 172.8f, 172.8f,
            172.8f, 172.8f, 165.6f, 153.0f
        )

        // 时序特征 (10 timesteps × 51 features = 510)
        // 这里简化处理，实际应该从完整数据加载
        val timeSeriesFeatures = FloatArray(510) { 0f }
        // 填充CGM值（索引0是CGM特征）
        for (i in 0..9) {
            if (i < historicalGlucose.size) {
                timeSeriesFeatures[i * 51] = historicalGlucose[i]
            } else {
                timeSeriesFeatures[i * 51] = historicalGlucose.last()
            }
        }

        // 静态特征 (30个)
        val staticFeatures = floatArrayOf(
            1f,      // 性别 (男性=1)
            78f,     // 年龄
            1.66f,   // 身高
            67.0f,   // 体重
            24.3f,   // BMI
            1f,      // 糖尿病类型 (T1DM=1)
            20f,     // 病程
            100.0f,  // HbA1c
            133.2f,  // 空腹血糖
            *FloatArray(21) { 0f } // 其他特征
        )

        return PatientData(
            patientId = "2035_0_20210629",
            historicalGlucose = historicalGlucose,
            timeSeriesFeatures = timeSeriesFeatures,
            staticFeatures = staticFeatures
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        predictor?.close()
    }
}

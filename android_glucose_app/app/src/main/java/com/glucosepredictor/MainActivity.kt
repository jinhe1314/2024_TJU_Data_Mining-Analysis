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
    private lateinit var predictButton: Button
    private lateinit var showDetailsButton: Button
    private lateinit var patientInfoText: TextView

    private var predictor: GlucosePredictor? = null
    private var currentPredictions: PredictionResult? = null
    private val tag = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化视图
        chart = findViewById(R.id.chart)
        predictButton = findViewById(R.id.predictButton)
        showDetailsButton = findViewById(R.id.showDetailsButton)
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

        // 显示详情按钮点击事件
        showDetailsButton.setOnClickListener {
            showPredictionDetailsDialog()
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
                        // 保存预测结果
                        currentPredictions = result

                        // 更新图表
                        updateChart(patientData.historicalGlucose, result)
                    }
                    predictButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e(tag, "预测出错: ${e.message}")
                withContext(Dispatchers.Main) {
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
            lineWidth = 2.5f
            circleRadius = 4f
            setDrawCircleHole(true)
            setDrawValues(false)
            valueTextSize = 8f
            mode = LineDataSet.Mode.LINEAR
        }
        dataSets.add(historicalDataSet)

        // 预测时间点 (135, 150, 165, 180分钟)
        val predictionTimes = floatArrayOf(135f, 150f, 165f, 180f)

        // 历史数据最后一个点 (120分钟)
        val lastHistoricalTime = 120f
        val lastHistoricalValue = historicalGlucose.last()

        // 2. 完整输入预测 (红线，虚线连接)
        val fullEntries = ArrayList<Entry>()
        fullEntries.add(Entry(lastHistoricalTime, lastHistoricalValue))  // 连接点
        for (i in predictions.fullInput.indices) {
            fullEntries.add(Entry(predictionTimes[i], predictions.fullInput[i]))
        }
        val fullDataSet = LineDataSet(fullEntries, "With Patient Info").apply {
            color = Color.RED
            setCircleColor(Color.RED)
            lineWidth = 2.5f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(false)
            valueTextSize = 8f
            valueTextColor = Color.RED
            mode = LineDataSet.Mode.LINEAR
            enableDashedLine(10f, 10f, 0f)  // 虚线效果
        }
        dataSets.add(fullDataSet)

        // 3. 无患者信息预测 (绿线，虚线连接)
        val noStaticEntries = ArrayList<Entry>()
        noStaticEntries.add(Entry(lastHistoricalTime, lastHistoricalValue))
        for (i in predictions.noPatientInfo.indices) {
            noStaticEntries.add(Entry(predictionTimes[i], predictions.noPatientInfo[i]))
        }
        val noStaticDataSet = LineDataSet(noStaticEntries, "Without Patient Info").apply {
            color = Color.GREEN
            setCircleColor(Color.GREEN)
            lineWidth = 2.5f
            circleRadius = 5f
            setDrawCircleHole(false)
            setDrawValues(false)
            valueTextSize = 8f
            valueTextColor = Color.GREEN
            mode = LineDataSet.Mode.LINEAR
            enableDashedLine(10f, 10f, 0f)
        }
        dataSets.add(noStaticDataSet)

        // 4. 低热量进餐预测 (橙线，虚线连接)
        val lowMealEntries = ArrayList<Entry>()
        lowMealEntries.add(Entry(lastHistoricalTime, lastHistoricalValue))
        for (i in predictions.lowCalorieMeal.indices) {
            lowMealEntries.add(Entry(predictionTimes[i], predictions.lowCalorieMeal[i]))
        }
        val lowMealDataSet = LineDataSet(lowMealEntries, "Low-Calorie (30min ago)").apply {
            color = Color.rgb(255, 165, 0) // Orange
            setCircleColor(Color.rgb(255, 165, 0))
            lineWidth = 2f
            circleRadius = 4f
            setDrawCircleHole(false)
            setDrawValues(false)
            valueTextSize = 7f
            mode = LineDataSet.Mode.LINEAR
            enableDashedLine(10f, 10f, 0f)
        }
        dataSets.add(lowMealDataSet)

        // 5. 中热量进餐预测 (紫线，虚线连接)
        val midMealEntries = ArrayList<Entry>()
        midMealEntries.add(Entry(lastHistoricalTime, lastHistoricalValue))
        for (i in predictions.mediumCalorieMeal.indices) {
            midMealEntries.add(Entry(predictionTimes[i], predictions.mediumCalorieMeal[i]))
        }
        val midMealDataSet = LineDataSet(midMealEntries, "Medium-Calorie (15min ago)").apply {
            color = Color.rgb(128, 0, 128) // Purple
            setCircleColor(Color.rgb(128, 0, 128))
            lineWidth = 2f
            circleRadius = 4f
            setDrawCircleHole(false)
            setDrawValues(false)
            valueTextSize = 7f
            mode = LineDataSet.Mode.LINEAR
            enableDashedLine(10f, 10f, 0f)
        }
        dataSets.add(midMealDataSet)

        // 6. 高热量进餐预测 (棕线，虚线连接)
        val highMealEntries = ArrayList<Entry>()
        highMealEntries.add(Entry(lastHistoricalTime, lastHistoricalValue))
        for (i in predictions.highCalorieMeal.indices) {
            highMealEntries.add(Entry(predictionTimes[i], predictions.highCalorieMeal[i]))
        }
        val highMealDataSet = LineDataSet(highMealEntries, "High-Calorie (continuous)").apply {
            color = Color.rgb(139, 69, 19) // Brown
            setCircleColor(Color.rgb(139, 69, 19))
            lineWidth = 2f
            circleRadius = 4f
            setDrawCircleHole(false)
            setDrawValues(false)
            valueTextSize = 7f
            mode = LineDataSet.Mode.LINEAR
            enableDashedLine(10f, 10f, 0f)
        }
        dataSets.add(highMealDataSet)

        // 设置数据
        val lineData = LineData(dataSets.map { it as com.github.mikephil.charting.interfaces.datasets.ILineDataSet })
        chart.data = lineData
        chart.invalidate()
    }

    /**
     * 显示预测详情对话框
     */
    private fun showPredictionDetailsDialog() {
        val predictions = currentPredictions
        if (predictions == null) {
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("提示")
                .setMessage("请先点击\"重新预测\"按钮进行预测")
                .setPositiveButton("确定", null)
                .show()
            return
        }

        val result = StringBuilder()
        result.append("预测结果 (15/30/45/60 分钟):\n\n")

        result.append("完整输入:\n")
        result.append(formatPredictions(predictions.fullInput))
        result.append("\n\n")

        result.append("无患者信息:\n")
        result.append(formatPredictions(predictions.noPatientInfo))
        result.append("\n\n")

        result.append("低热量进餐 (30分钟前轻食):\n")
        result.append(formatPredictions(predictions.lowCalorieMeal))
        result.append("\n\n")

        result.append("中热量进餐 (15分钟前正常进餐):\n")
        result.append(formatPredictions(predictions.mediumCalorieMeal))
        result.append("\n\n")

        result.append("高热量进餐 (持续大餐):\n")
        result.append(formatPredictions(predictions.highCalorieMeal))
        result.append("\n\n")

        // 影响分析 (仅显示60分钟后的影响)
        result.append("60分钟后血糖影响:\n")
        val lowImpact = predictions.lowCalorieMeal[3] - predictions.fullInput[3]
        val midImpact = predictions.mediumCalorieMeal[3] - predictions.fullInput[3]
        val highImpact = predictions.highCalorieMeal[3] - predictions.fullInput[3]

        result.append("低热量: %+.1f mg/dL\n".format(lowImpact))
        result.append("中热量: %+.1f mg/dL\n".format(midImpact))
        result.append("高热量: %+.1f mg/dL".format(highImpact))

        // 创建对话框
        val scrollView = android.widget.ScrollView(this)
        val textView = android.widget.TextView(this).apply {
            text = result.toString()
            setPadding(40, 20, 40, 20)
            textSize = 14f
            typeface = android.graphics.Typeface.MONOSPACE
        }
        scrollView.addView(textView)

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("预测详情")
            .setView(scrollView)
            .setPositiveButton("关闭", null)
            .show()
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
     * 获取示例患者数据 - 从JSON文件加载真实数据
     */
    private fun getSamplePatientData(): PatientData {
        try {
            // 从assets加载真实患者数据
            val jsonString = assets.open("patient_2035_data.json").bufferedReader().use { it.readText() }
            val json = org.json.JSONObject(jsonString)

            // 解析历史血糖值
            val glucoseArray = json.getJSONArray("historical_glucose")
            val historicalGlucose = FloatArray(glucoseArray.length()) { i ->
                glucoseArray.getDouble(i).toFloat()
            }

            // 解析时序特征
            val tsArray = json.getJSONArray("time_series_features")
            val timeSeriesFeatures = FloatArray(tsArray.length()) { i ->
                tsArray.getDouble(i).toFloat()
            }

            // 解析静态特征
            val staticArray = json.getJSONArray("static_features")
            val staticFeatures = FloatArray(staticArray.length()) { i ->
                staticArray.getDouble(i).toFloat()
            }

            Log.d(tag, "✓ 患者数据加载成功")
            Log.d(tag, "  CGM值: ${historicalGlucose.take(3).joinToString(", ")}")
            Log.d(tag, "  时序特征: ${timeSeriesFeatures.size} 个值")
            Log.d(tag, "  静态特征: ${staticFeatures.size} 个值")

            return PatientData(
                patientId = json.getString("patient_id"),
                historicalGlucose = historicalGlucose,
                timeSeriesFeatures = timeSeriesFeatures,
                staticFeatures = staticFeatures
            )
        } catch (e: Exception) {
            Log.e(tag, "患者数据加载失败: ${e.message}")
            e.printStackTrace()

            // 返回默认数据作为后备
            val historicalGlucose = floatArrayOf(
                142.2f, 158.4f, 172.8f, 172.8f, 172.8f,
                172.8f, 172.8f, 165.6f, 153.0f
            )
            val timeSeriesFeatures = FloatArray(510) { 0f }
            val staticFeatures = FloatArray(30) { 0f }

            return PatientData(
                patientId = "2035_0_20210629",
                historicalGlucose = historicalGlucose,
                timeSeriesFeatures = timeSeriesFeatures,
                staticFeatures = staticFeatures
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        predictor?.close()
    }
}

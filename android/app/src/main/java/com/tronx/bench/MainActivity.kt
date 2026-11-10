package com.tronx.bench

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.sqrt

/**
 * TronX on-device benchmark (assignment Section 9). Every metric is printed on screen and can be
 * copied/shared (nothing is saved on the phone). Models are unpacked to app-private storage: <app files>/models/<variant>/model.onnx(+_data).
 */
class MainActivity : Activity() {
    private lateinit var out: TextView
    private lateinit var scroll: ScrollView
    private lateinit var variantsGroup: RadioGroup
    private val full = StringBuilder()
    private val report = JSONObject()
    private lateinit var ambient: EditText
    private lateinit var threads: EditText
    private lateinit var repeats: EditText
    private lateinit var nPrompts: EditText
    private lateinit var thermalMin: EditText
    private lateinit var cooldown: EditText
    @Volatile private var busy = false

    private val bundledDir by lazy { File(filesDir, "models").apply { mkdirs() } }

    // ---------------- chat state
    private var engine: LlmEngine? = null
    private var engineName: String? = null
    private var tokens: Tokens? = null
    private lateinit var input: EditText
    private lateinit var reply: TextView
    private val chats = JSONObject()          // variant -> [ {command, output, metrics} ]

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        fun btnTo(parent: LinearLayout, t: String, f: () -> Unit) = parent.addView(Button(this).apply { text = t; isAllCaps = false; setOnClickListener { f() } })
        root.addView(TextView(this).apply { text = "TronX voice-command parser (Granite-4.0-H-350M, ONNX Runtime ${BuildConfig.ORT_VERSION})"; textSize = 15f; setTypeface(null, Typeface.BOLD) })
        root.addView(TextView(this).apply { text = "1) Pick a model  2) type or pick a command  3) Send.  Share sends everything to the report."; textSize = 12f })
        variantsGroup = RadioGroup(this); root.addView(variantsGroup)
        variantsGroup.setOnCheckedChangeListener { _, id -> variants().getOrNull(id - 1000)?.let { v -> bg { loadVariant(v) } } }
        // test-prompt picker fills the input box
        val picker = android.widget.Spinner(this)
        root.addView(picker)
        input = EditText(this).apply { hint = "Type a voice command, e.g. Set an alarm for 6:30 AM tomorrow."; textSize = 15f }
        root.addView(input)
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        root.addView(row)
        fun rowBtn(t: String, f: () -> Unit) = row.addView(Button(this).apply { text = t; isAllCaps = false; setOnClickListener { f() } }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        rowBtn("Send") { val c = input.text.toString().trim(); if (c.isNotEmpty()) bg { send(c) } }
        rowBtn("Share") { share() }
        rowBtn("Copy") { copyAll() }
        rowBtn("Clear") { full.setLength(0); out.text = "" }
        // latest answer, shown right under the Send button
        reply = TextView(this).apply {
            textSize = 15f; typeface = Typeface.MONOSPACE; setTextIsSelectable(true); setPadding(16, 16, 16, 16)
            setBackgroundColor(0x22888888); text = "Model reply will appear here."
        }
        root.addView(reply)
        // advanced Section 9 benchmark (KV cache 512/1K/2K, repeated runs, thermal) - collapsed
        val adv = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; visibility = android.view.View.GONE }
        btnTo(root, "Section 9 benchmark (KV cache, repeated runs, thermal) ▾") { adv.visibility = if (adv.visibility == android.view.View.GONE) android.view.View.VISIBLE else android.view.View.GONE }
        root.addView(adv)
        fun field(label: String, def: String): EditText {
            val r = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            r.addView(TextView(this).apply { text = label; textSize = 12f }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1.4f))
            val e = EditText(this).apply { setText(def); textSize = 12f; inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL }
            r.addView(e, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)); adv.addView(r); return e
        }
        ambient = field("Ambient temperature (°C)", "")
        threads = field("CPU threads", "4")
        repeats = field("Repeated runs", "3")
        nPrompts = field("Prompts per run", "10")
        thermalMin = field("Thermal test minutes", "10")
        cooldown = field("Cool-down between variants (s)", "60")
        btnTo(adv, "Device info") { bg { deviceInfo() } }
        btnTo(adv, "Run benchmark on SELECTED model (A-F)") { bg { selected()?.let { closeEngine(); runVariant(it) } } }
        btnTo(adv, "Run benchmark on ALL models (A-F)") { bg { closeEngine(); runAll() } }
        btnTo(adv, "Thermal test on SELECTED model (G)") { bg { selected()?.let { closeEngine(); thermal(it) } } }
        out = TextView(this).apply { typeface = Typeface.MONOSPACE; textSize = 11f; setTextIsSelectable(true) }
        root.addView(out)
        scroll = ScrollView(this).apply { addView(root) }
        setContentView(scroll, android.view.ViewGroup.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        refreshVariants()
        bg {
            unpackBundledModels()
            val t = try { Tokens(this) } catch (e: Throwable) { log("TOKENIZER FAILED: ${e.javaClass.simpleName}: ${e.message}"); throw e }
            tokens = t
            // self-check: on-device tokenizer must reproduce the Mac (Hugging Face) token ids of the 20 benchmark prompts
            val bad = t.prompts.count { (txt, ids) -> !t.chatIds(txt).contentEquals(ids) }
            log(if (bad == 0) "Tokenizer self-check: 20/20 prompts identical to Mac tokenizer" else "WARNING tokenizer self-check: $bad/20 prompts differ from Mac tokenizer")
            runOnUiThread {
                val items = listOf("— pick one of the 20 test prompts —") + t.testPrompts.mapIndexed { i, s -> "${i + 1}. $s" }
                picker.adapter = android.widget.ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
                picker.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(p: android.widget.AdapterView<*>?, v: android.view.View?, pos: Int, id: Long) { if (pos > 0) input.setText(t.testPrompts[pos - 1]) }
                    override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
                }
            }
            deviceInfoQuiet()
            variants().firstOrNull()?.let { loadVariant(it) }
        }
    }

    private fun closeEngine() { engine?.close(); engine = null; engineName = null; System.gc() }

    /** Load a model variant and record Section 9 A/B (file size, loading memory, loading time). */
    private fun loadVariant(dir: File) {
        if (engineName == dir.name) return
        closeEngine(); Thread.sleep(300)
        val files = dir.listFiles()!!.filter { it.isFile && !it.name.startsWith(".") }
        val bytes = files.sumOf { it.length() }
        val before = DeviceStats.rssMb(); val sampler = DeviceStats.PeakSampler(); sampler.start()
        val t0 = System.nanoTime()
        engine = LlmEngine(File(dir, "model.onnx"), num(threads, 4.0).toInt())
        val loadMs = (System.nanoTime() - t0) / 1e6
        val after = DeviceStats.rssMb(); val peak = sampler.stop()
        engineName = dir.name
        log("\n=== MODEL ${dir.name} loaded: file ${f1(bytes / 1048576.0)} MB | load ${f1(loadMs)} ms | RAM ${f1(before)} -> ${f1(after)} MB (+${f1(after - before)}), peak ${f1(peak)} MB ===")
        val v = chats.optJSONObject(dir.name) ?: JSONObject().also { chats.put(dir.name, it) }
        v.put("load", JSONObject(mapOf("file_mb" to bytes / 1048576.0, "load_ms" to loadMs, "rss_before_mb" to before, "rss_after_mb" to after,
            "rss_peak_mb" to peak, "state_dtype" to engine!!.stateType.name)))
        if (!v.has("chats")) v.put("chats", JSONArray())
    }

    /** One chat turn: command -> JSON, with per-command metrics. */
    private fun send(command: String) {
        val t = tokens ?: run { log("Tokenizer still loading..."); return }
        val dir = selected() ?: return
        loadVariant(dir)
        val eng = engine ?: return
        val ids = t.chatIds(command)
        runOnUiThread { reply.text = "[${dir.name}]  $command\n\nthinking... (prompt ${ids.size} tokens)" }
        val g = eng.generate(ids, 64, t.eos)
        val outText = t.decode(g.tokens).trim()
        val totalMs = g.ttftMs + g.decodeMs
        val metricsLine = "prompt ${g.promptTokens} tok | TTFT ${f1(g.ttftMs)} ms | prefill ${f1(g.prefillTps)} tok/s | decode ${f2(g.decodeTps)} tok/s | " +
            "out ${g.tokens.size} tok | total ${f1(totalMs)} ms | peak RAM ${f1(g.peakRssMb)} MB"
        runOnUiThread { reply.text = "[${dir.name}]  $command\n\n$outText\n\n$metricsLine" }
        log("\nYOU  > $command")
        log("${dir.name} > $outText")
        log("       prompt ${g.promptTokens} tok | TTFT ${f1(g.ttftMs)} ms | prefill ${f1(g.prefillTps)} tok/s | decode ${f2(g.decodeTps)} tok/s | " +
            "out ${g.tokens.size} tok | total ${f1(totalMs)} ms | peak RAM ${f1(g.peakRssMb)} MB (+${f1(g.peakRssMb - g.baselineRssMb)})")
        chats.getJSONObject(dir.name).getJSONArray("chats").put(JSONObject(mapOf("command" to command, "output" to outText,
            "prompt_tokens" to g.promptTokens, "out_tokens" to g.tokens.size, "ttft_ms" to g.ttftMs, "prefill_tps" to g.prefillTps,
            "decode_tps" to g.decodeTps, "total_ms" to totalMs, "peak_rss_mb" to g.peakRssMb, "baseline_rss_mb" to g.baselineRssMb,
            "batt_temp_c" to DeviceStats.battery(this).tempC, "thermal_status" to DeviceStats.thermalStatusInt(this))))
    }

    /** Device info collected silently for the shared report (also shown in the Section 9 panel). */
    private fun deviceInfoQuiet() {
        val b = DeviceStats.battery(this)
        report.put("device", JSONObject(mapOf("model" to "${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})", "soc" to DeviceStats.socName(),
            "cpu" to DeviceStats.cpuDescription(), "gpu" to DeviceStats.gpuName(), "ram_total_mb" to DeviceStats.totalRamMb(this),
            "ram_avail_mb" to DeviceStats.availRamMb(this), "android" to "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}) ${Build.DISPLAY}",
            "runtime" to "ONNX Runtime Android ${BuildConfig.ORT_VERSION} CPU EP, threads ${num(threads, 4.0).toInt()}",
            "power_save" to DeviceStats.powerSave(this), "battery_pct" to b.levelPct, "battery_temp_c" to b.tempC, "plugged" to b.plugged,
            "app" to packageName)))
        log("Device: ${Build.MANUFACTURER} ${Build.MODEL} | ${DeviceStats.socName()} | Android ${Build.VERSION.RELEASE} | battery ${b.levelPct}% ${f1(b.tempC)}°C")
    }

    private fun refreshVariants() {
        variantsGroup.removeAllViews()
        val vs = variants()
        vs.forEachIndexed { i, v -> variantsGroup.addView(RadioButton(this).apply { text = v.name; id = 1000 + i; isChecked = i == 0 }) }
        if (vs.isEmpty()) variantsGroup.addView(TextView(this).apply { text = "  (none found - copy model folders first)"; textSize = 12f })
    }

    /** Models bundled in the APK (assets/models/<variant>/) are copied once to internal storage, because
     *  ONNX Runtime needs real file paths for model.onnx + model.onnx_data. */
    private fun unpackBundledModels() {
        val names = assets.list("models")?.toList() ?: emptyList()
        if (names.isEmpty()) { log("No models bundled in this APK."); return }
        for (v in names) {
            val files = assets.list("models/$v") ?: emptyArray()
            if (files.none { it == "model.onnx" }) continue   // not one of our model folders (e.g. system dnn.tflite)
            val dst = File(bundledDir, v); val done = File(dst, ".complete")
            if (done.exists()) continue
            dst.mkdirs(); log("Unpacking bundled model $v (first launch only, ~30-60 s)...")
            for (f in files) {
                assets.open("models/$v/$f").use { i -> File(dst, f).outputStream().use { o -> i.copyTo(o, 1 shl 20) } }
            }
            done.writeText("ok")
        }
        runOnUiThread { refreshVariants() }
        log("Models ready: ${variants().joinToString { it.name }}")
    }

    private fun variants(): List<File> = (listOf(bundledDir).flatMap { d ->
        d.listFiles { f -> File(f, "model.onnx").exists() && (d != bundledDir || File(f, ".complete").exists()) }?.toList() ?: emptyList()
    }).distinctBy { it.name }.sortedBy { it.name }
    private fun selected(): File? { val id = variantsGroup.checkedRadioButtonId; return variants().getOrNull(id - 1000) ?: run { log("No model selected (wait for unpacking to finish)."); null } }
    private fun num(e: EditText, d: Double) = e.text.toString().toDoubleOrNull() ?: d

    private fun bg(task: () -> Unit) {
        if (busy) { log("Busy - wait for the current test to finish."); return }
        busy = true
        Thread {
            try { task() } catch (t: Throwable) { log("ERROR: ${t.javaClass.simpleName}: ${t.message}") }
            finally { busy = false }
        }.start()
    }

    private fun log(s: String) {
        full.append(s).append('\n')
        runOnUiThread { out.append(s + "\n"); scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) } }
    }

    private fun h(title: String) = log("\n==================== $title ====================")
    private fun f1(x: Double) = String.format(Locale.US, "%.1f", x)
    private fun f2(x: Double) = String.format(Locale.US, "%.2f", x)
    private fun mean(x: List<Double>) = if (x.isEmpty()) 0.0 else x.sum() / x.size
    private fun std(x: List<Double>): Double { if (x.size < 2) return 0.0; val m = mean(x); return sqrt(x.sumOf { (it - m) * (it - m) } / (x.size - 1)) }
    private fun stat(x: List<Double>) = "avg ${f2(mean(x))} | std ${f2(std(x))} | min ${f2(x.minOrNull() ?: 0.0)} | max ${f2(x.maxOrNull() ?: 0.0)} | n ${x.size}"

    // ------------------------------------------------------------------ device info (Section 9 header)
    private fun deviceInfo() {
        h("DEVICE")
        val b = DeviceStats.battery(this)
        val info = linkedMapOf(
            "Manufacturer / model" to "${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})",
            "SoC / chipset" to DeviceStats.socName(),
            "CPU" to DeviceStats.cpuDescription(),
            "GPU" to DeviceStats.gpuName(),
            "NPU" to "Not used: ONNX Runtime CPU execution provider (no NPU delegate in this build)",
            "RAM total / available" to "${DeviceStats.totalRamMb(this)} MB / ${DeviceStats.availRamMb(this)} MB",
            "Android version" to "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}), build ${Build.DISPLAY}",
            "Runtime and version" to "ONNX Runtime Android ${BuildConfig.ORT_VERSION}, CPU EP, intra-op threads ${num(threads, 4.0).toInt()}, graph opt ALL",
            "Power mode" to if (DeviceStats.powerSave(this)) "Battery saver ON" else "Normal (battery saver OFF)",
            "Ambient temperature" to (ambient.text.toString().ifBlank { "NOT ENTERED (type it in the field above)" } + " °C"),
            "Battery at start" to "${b.levelPct}% | ${f1(b.tempC)} °C | ${b.voltageMv} mV | ${if (b.plugged) "CHARGING (plugged)" else "on battery"}",
            "Thermal status / headroom" to "${DeviceStats.thermalStatus(this)} / ${DeviceStats.thermalHeadroom(this)}",
            "CPU freqs now (MHz)" to DeviceStats.cpuFreqsMhz().joinToString(","),
            "Readable thermal zones" to DeviceStats.thermalZones().joinToString("; ") { "${it.first}=${f1(it.second)}°C" }.ifBlank { "none readable (permission)" },
            "Process RSS now" to "${f1(DeviceStats.rssMb())} MB",
            "Time" to SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date()))
        info.forEach { (k, v) -> log(String.format("%-28s %s", "$k:", v)) }
        report.put("device", JSONObject(info as Map<*, *>))
    }

    // ------------------------------------------------------------------ sections A-F for one variant
    private fun runAll() {
        deviceInfo()
        val vs = variants()
        vs.forEachIndexed { i, v ->
            runVariant(v)
            if (i < vs.size - 1) {
                val s = num(cooldown, 60.0).toInt(); log("\nCool-down $s s (same starting conditions for every variant)...")
                Thread.sleep(s * 1000L)
            }
        }
        h("ALL VARIANTS DONE"); summaryTable()
    }

    private fun runVariant(dir: File) {
        val tok = Tokens(this)
        val name = dir.name
        val r = JSONObject(); report.put(name, r)
        h("VARIANT: $name")
        // ---- A. file size
        val files = dir.listFiles()!!.filter { it.isFile && !it.name.startsWith(".") }
        val bytes = files.sumOf { it.length() }
        log("A. MODEL FILE SIZE"); files.forEach { log("   ${it.name}: ${f1(it.length() / 1048576.0)} MB") }
        log("   TOTAL: ${f1(bytes / 1048576.0)} MB (${bytes} bytes)")
        r.put("file_size_mb", bytes / 1048576.0)
        // ---- B. loading memory
        System.gc(); Thread.sleep(500)
        val before = DeviceStats.rssMb(); val hwmBefore = DeviceStats.hwmMb()
        val sampler = DeviceStats.PeakSampler(); sampler.start()
        val t0 = System.nanoTime()
        val eng = LlmEngine(File(dir, "model.onnx"), num(threads, 4.0).toInt())
        val loadMs = (System.nanoTime() - t0) / 1e6
        val afterLoad = DeviceStats.rssMb(); val loadPeak = sampler.stop()
        System.gc(); Thread.sleep(1500); val settled = DeviceStats.rssMb()
        log("B. MODEL LOADING MEMORY (process RSS, /proc/self/status, sampled every 15 ms)")
        log("   memory before loading : ${f1(before)} MB")
        log("   memory after loading  : ${f1(afterLoad)} MB  (+${f1(afterLoad - before)} MB)")
        log("   model-loading peak    : ${f1(loadPeak)} MB  (+${f1(loadPeak - before)} MB); VmHWM ${f1(DeviceStats.hwmMb())} MB (was ${f1(hwmBefore)})")
        log("   loading time          : ${f1(loadMs)} ms")
        log("   after 1.5 s settle    : ${f1(settled)} MB  -> released after loading ${f1(loadPeak - settled)} MB")
        log("   state tensors dtype   : ${eng.stateType} (KV + Mamba states), KV-cache quantisation: NO")
        r.put("load", JSONObject(mapOf("before_mb" to before, "after_mb" to afterLoad, "peak_mb" to loadPeak, "load_ms" to loadMs, "settled_mb" to settled, "released_mb" to loadPeak - settled)))
        try {
            // ---- C/E/F. command prompts (real use case: schema system prompt + command)
            val reps = num(repeats, 3.0).toInt(); val np = num(nPrompts, 10.0).toInt().coerceIn(1, tok.prompts.size)
            log("C/E/F. COMMAND PROMPTS: ${np} prompts x ${reps} repeated runs, greedy, max 48 new tokens, stop at EOS")
            log("   prompt length: ${tok.prompts.take(np).map { it.second.size }.let { "${it.minOrNull()}-${it.maxOrNull()} tokens (schema system prompt ${tok.schemaPromptTokens} tok)" }}")
            log("   warm-up: 2 un-timed runs (prompts 1-2) before measuring")
            tok.prompts.take(2).forEachIndexed { k, it -> val g = eng.generate(it.second, 48, tok.eos); log("   warm-up ${k + 1}/2 done: TTFT ${f1(g.ttftMs)} ms") }
            val ttft = ArrayList<Double>(); val dec = ArrayList<Double>(); val pre = ArrayList<Double>(); val ntok = ArrayList<Double>()
            var peak = 0.0; var base = Double.MAX_VALUE; var after = 0.0
            val samples = JSONArray()
            val totalRuns = reps * np; var runNo = 0; val tStart = System.currentTimeMillis()
            for (rep in 1..reps) for ((i, p) in tok.prompts.take(np).withIndex()) {
                val g = eng.generate(p.second, 48, tok.eos)
                runNo++
                val el = (System.currentTimeMillis() - tStart) / 1000.0; val eta = el / runNo * (totalRuns - runNo)
                log("   run $runNo/$totalRuns (${100 * runNo / totalRuns}%): TTFT ${f1(g.ttftMs)} ms, decode ${f2(g.decodeTps)} tok/s, ${g.tokens.size} tok | elapsed ${f1(el / 60)} min, ETA ${f1(eta / 60)} min")
                ttft += g.ttftMs; pre += g.prefillTps; if (g.decodeTokens > 0) dec += g.decodeTps; ntok += g.tokens.size.toDouble()
                peak = maxOf(peak, g.peakRssMb); base = minOf(base, g.baselineRssMb); after = g.afterRssMb
                if (rep == 1 && i < 4) log("   sample ${i + 1}: \"${p.first}\"\n      -> ${tok.decode(g.tokens)}")
                samples.put(JSONObject(mapOf("rep" to rep, "prompt" to i, "ttft_ms" to g.ttftMs, "decode_tps" to g.decodeTps, "prefill_tps" to g.prefillTps, "gen_tokens" to g.tokens.size)))
            }
            log("C. PEAK INFERENCE RAM")
            log("   baseline process memory (model loaded, idle): ${f1(base)} MB")
            log("   peak memory during inference              : ${f1(peak)} MB")
            log("   memory after inference                    : ${f1(after)} MB")
            log("   increase attributable to inference        : ${f1(peak - base)} MB")
            log("   method: VmRSS sampled every 15 ms during generation")
            log("E. TOKENS PER SECOND")
            log("   prompt processing (prefill) tok/s : ${stat(pre)}")
            log("   decode tok/s                      : ${stat(dec)}")
            log("   generated tokens per answer       : ${stat(ntok)}")
            log("F. TIME TO FIRST TOKEN (model loading EXCLUDED, tokenization EXCLUDED - pre-tokenised;")
            log("   boundary = start of first session.run until first token id known = prefill + 1st argmax)")
            log("   TTFT ms : ${stat(ttft)}")
            r.put("prompts", JSONObject(mapOf("n_prompts" to np, "repeats" to reps, "ttft_ms_avg" to mean(ttft), "ttft_ms_std" to std(ttft),
                "decode_tps_avg" to mean(dec), "decode_tps_std" to std(dec), "prefill_tps_avg" to mean(pre), "prefill_tps_std" to std(pre),
                "gen_tokens_avg" to mean(ntok), "baseline_mb" to base, "peak_mb" to peak, "after_mb" to after, "inference_increase_mb" to peak - base)))
            r.put("samples", samples)
            // ---- D. KV cache at 512/1024/2048
            log("D. KV CACHE vs CONTEXT (prompt filled to exact length, then exactly 32 tokens decoded, EOS ignored; 2 runs each, avg)")
            log(String.format("   %-8s %-10s %-12s %-12s %-14s %-10s %-10s", "context", "prompt", "KV(attn)", "Mamba state", "peak RAM", "TTFT ms", "decode t/s"))
            val kvj = JSONObject()
            for (n in listOf(512, 1024, 2048)) {
                val ids = tok.ctxPrompts[n] ?: continue
                log("   ... context $n: running 2 x (prefill $n tokens + 32 decode)")
                val runs = (1..2).map { eng.generate(ids, 32, tok.eos, forceTokens = 32) }
                val g = runs.last()
                val tt = runs.map { it.ttftMs }.average(); val dt = runs.map { it.decodeTps }.average(); val pk = runs.maxOf { it.peakRssMb }
                log(String.format("   %-8d %-10s %-12s %-12s %-14s %-10s %-10s", n, "${g.promptTokens}+32", "${f2(g.kvBytes / 1048576.0)} MB", "${f2(g.ssmBytes / 1048576.0)} MB", "${f1(pk)} MB", f1(tt), f2(dt)))
                kvj.put(n.toString(), JSONObject(mapOf("prompt_tokens" to g.promptTokens, "output_tokens" to 32, "total_context" to g.promptTokens + 32,
                    "kv_mb" to g.kvBytes / 1048576.0, "mamba_state_mb" to g.ssmBytes / 1048576.0, "peak_mb" to pk, "ttft_ms" to tt, "decode_tps" to dt, "dtype" to g.kvDtype)))
            }
            log("   KV dtype: ${eng.stateType}; KV quantisation: no. KV = 4 attention layers x (K+V) x 4 heads x 64 dims per token;")
            log("   Mamba conv/SSM state is FIXED size (does not grow with context).")
            r.put("kv_cache", kvj)
        } finally { eng.close() }
        System.gc(); Thread.sleep(1000)
        log("   memory after unloading model: ${f1(DeviceStats.rssMb())} MB")
    }

    // ------------------------------------------------------------------ G. thermal
    private fun thermal(dir: File) {
        val tok = Tokens(this)
        val minutes = num(thermalMin, 10.0)
        h("G. THERMAL TEST: ${dir.name}, ${f1(minutes)} min sustained inference")
        val b0 = DeviceStats.battery(this)
        log("initial: battery ${b0.levelPct}% ${f1(b0.tempC)}°C | thermal ${DeviceStats.thermalStatus(this)} | headroom ${DeviceStats.thermalHeadroom(this)} | ${if (b0.plugged) "PLUGGED IN (battery drain not measurable)" else "on battery"}")
        log("zones: " + DeviceStats.thermalZones().joinToString("; ") { "${it.first}=${f1(it.second)}" }.ifBlank { "none readable" })
        val eng = LlmEngine(File(dir, "model.onnx"), num(threads, 4.0).toInt())
        val rows = JSONArray()
        log(String.format("%-6s %-9s %-9s %-8s %-11s %-9s %-6s %-9s %s", "t(s)", "dec t/s", "TTFT ms", "batt°C", "thermal", "headroom", "batt%", "curr mA", "CPU MHz (per core)"))
        val t0 = System.currentTimeMillis(); val end = t0 + (minutes * 60_000).toLong()
        var i = 0; var maxStatus = 0; var maxTemp = b0.tempC
        val windows = ArrayList<Double>()
        try {
            while (System.currentTimeMillis() < end) {
                val wStart = System.currentTimeMillis(); val dec = ArrayList<Double>(); val tt = ArrayList<Double>()
                while (System.currentTimeMillis() - wStart < 30_000 && System.currentTimeMillis() < end) {
                    val g = eng.generate(tok.prompts[i % tok.prompts.size].second, 48, tok.eos); i++
                    if (g.decodeTokens > 0) dec += g.decodeTps; tt += g.ttftMs
                }
                val b = DeviceStats.battery(this); val st = DeviceStats.thermalStatusInt(this)
                maxStatus = maxOf(maxStatus, st); maxTemp = maxOf(maxTemp, b.tempC); windows += mean(dec)
                val t = (System.currentTimeMillis() - t0) / 1000
                val freqs = DeviceStats.cpuFreqsMhz().joinToString(",")
                log(String.format("%-6d %-9s %-9s %-8s %-11s %-9s %-6d %-9s %s", t, f2(mean(dec)), f1(mean(tt)), f1(b.tempC), DeviceStats.thermalStatus(this), String.format(Locale.US, "%.2f", DeviceStats.thermalHeadroom(this)), b.levelPct, (b.currentUa / 1000).toString(), freqs))
                rows.put(JSONObject(mapOf("t_s" to t, "decode_tps" to mean(dec), "ttft_ms" to mean(tt), "batt_temp_c" to b.tempC, "thermal_status" to st,
                    "headroom" to DeviceStats.thermalHeadroom(this).toDouble(), "batt_pct" to b.levelPct, "current_ma" to b.currentUa / 1000, "cpu_mhz" to freqs,
                    "zones" to DeviceStats.thermalZones().joinToString(";") { "${it.first}=${f1(it.second)}" })))
            }
        } finally { eng.close() }
        val b1 = DeviceStats.battery(this)
        val first = windows.firstOrNull() ?: 0.0; val last = windows.lastOrNull() ?: 0.0
        h("G. THERMAL SUMMARY: ${dir.name}")
        log("initial battery temp : ${f1(b0.tempC)} °C")
        log("max during inference : ${f1(maxTemp)} °C")
        log("after sustained load : ${f1(b1.tempC)} °C")
        log("decode tok/s first 30 s window: ${f2(first)} | last window: ${f2(last)} | change: ${f1(if (first > 0) 100 * (last - first) / first else 0.0)} %")
        log("worst thermal status reached  : $maxStatus (0 NONE,1 LIGHT,2 MODERATE,3 SEVERE,4 CRITICAL); final headroom ${DeviceStats.thermalHeadroom(this)}")
        log("battery: ${b0.levelPct}% -> ${b1.levelPct}% (${b0.levelPct - b1.levelPct} % used in ${f1(minutes)} min) ${if (b0.plugged) "[PLUGGED - not valid]" else ""}")
        log("generations completed: $i")
        log("THROTTLING VERDICT: use thermal status >= 1 / headroom >= 1.0 / CPU MHz drop together with tok/s drop - not temperature alone.")
        report.put("thermal_${dir.name}", JSONObject(mapOf("minutes" to minutes, "rows" to rows, "batt_temp_start" to b0.tempC, "batt_temp_max" to maxTemp,
            "batt_temp_end" to b1.tempC, "tps_first" to first, "tps_last" to last, "max_thermal_status" to maxStatus,
            "batt_pct_start" to b0.levelPct, "batt_pct_end" to b1.levelPct, "plugged" to b0.plugged, "generations" to i)))
    }

    private fun summaryTable() {
        log(String.format("%-14s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s", "variant", "file MB", "load MB+", "peak inf", "TTFT ms", "dec t/s", "KV512 MB", "KV1K MB", "KV2K MB"))
        variants().forEach { v ->
            val r = report.optJSONObject(v.name) ?: return@forEach
            val p = r.optJSONObject("prompts"); val l = r.optJSONObject("load"); val kv = r.optJSONObject("kv_cache")
            fun kvmb(n: String) = kv?.optJSONObject(n)?.optDouble("kv_mb")?.let { f2(it) } ?: "-"
            log(String.format("%-14s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s", v.name, f1(r.optDouble("file_size_mb")),
                f1((l?.optDouble("after_mb") ?: 0.0) - (l?.optDouble("before_mb") ?: 0.0)), f1(p?.optDouble("peak_mb") ?: 0.0),
                f1(p?.optDouble("ttft_ms_avg") ?: 0.0), f2(p?.optDouble("decode_tps_avg") ?: 0.0), kvmb("512"), kvmb("1024"), kvmb("2048")))
        }
    }

    private fun copyAll() {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        report.put("chats", chats)
        cm.setPrimaryClip(ClipData.newPlainText("tronx-bench", full.toString() + "\n\nJSON:\n" + report.toString()))
        log("(copied ${full.length} chars + JSON to clipboard)")
    }

    private fun share() {
        report.put("chats", chats)
        val i = Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_SUBJECT, "TronX bench report"); putExtra(Intent.EXTRA_TEXT, full.toString() + "\n\nJSON:\n" + report.toString()) }
        startActivity(Intent.createChooser(i, "Share report"))
    }
}

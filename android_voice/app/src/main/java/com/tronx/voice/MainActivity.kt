package com.tronx.voice

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

/**
 * TronX Voice: on-device voice/text -> JSON chat.
 *   mic -> Silero VAD -> 2 s silence -> Moonshine v2 small (ONNX q8) -> Granite-4.0-H-350M + LoRA D (ONNX INT8) -> JSON
 * Every reply shows the raw JSON plus RAM and timings. No network, no sharing.
 */
class MainActivity : Activity() {
    // palette
    private val bg = Color.parseColor("#0E1116"); private val surface = Color.parseColor("#171B22")
    private val card = Color.parseColor("#1D222B"); private val stroke = Color.parseColor("#2A313C")
    private val textMain = Color.parseColor("#E8EDF3"); private val textDim = Color.parseColor("#8B95A5")
    private val accent = Color.parseColor("#4C8DFF"); private val good = Color.parseColor("#3DD68C")
    private val warn = Color.parseColor("#FFB547"); private val bad = Color.parseColor("#FF5C5C")

    private lateinit var chat: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var status: TextView
    private lateinit var input: EditText
    private lateinit var mic: TextView
    private lateinit var send: TextView
    private val worker = Executors.newSingleThreadExecutor()
    @Volatile private var busy = true
    @Volatile private var listening = false

    private var llm: LlmEngine? = null
    private var tokens: Tokens? = null
    private var stt: MoonshineStt? = null
    private var vad: SileroVad? = null
    private val threads = 4

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun round(color: Int, r: Int, strokeColor: Int? = null) = GradientDrawable().apply {
        setColor(color); cornerRadius = dp(r).toFloat(); if (strokeColor != null) setStroke(dp(1), strokeColor)
    }
    private fun f1(x: Double) = String.format(Locale.US, "%.1f", x)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.statusBarColor = bg; window.navigationBarColor = bg
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(bg) }

        // header
        val header = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(18), dp(16), dp(18), dp(10)) }
        header.addView(TextView(this).apply { text = "TronX Voice"; textSize = 22f; setTextColor(textMain); setTypeface(null, Typeface.BOLD) })
        header.addView(TextView(this).apply {
            text = "On-device · Granite-4.0-H-350M + LoRA · INT8 · Moonshine STT"; textSize = 12f; setTextColor(textDim)
        })
        status = TextView(this).apply {
            text = "Preparing models…"; textSize = 12f; setTextColor(warn); setPadding(dp(10), dp(4), dp(10), dp(4))
            background = round(surface, 12, stroke)
        }
        header.addView(status, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { topMargin = dp(8) })
        root.addView(header)

        // chat
        chat = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), dp(6), dp(12), dp(12)) }
        scroll = ScrollView(this).apply { addView(chat); isFillViewport = true }
        root.addView(scroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))

        // preset hard prompts (tap = send)
        val chipsRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(dp(10), dp(6), dp(10), dp(6)) }
        val chipsScroll = HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false; addView(chipsRow) }
        root.addView(chipsScroll)

        // input bar
        val bar = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(dp(10), dp(6), dp(10), dp(12)) }
        input = EditText(this).apply {
            hint = "Type a command…"; setHintTextColor(textDim); setTextColor(textMain); textSize = 15f
            background = round(surface, 22, stroke); setPadding(dp(16), dp(10), dp(16), dp(10))
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES; imeOptions = EditorInfo.IME_ACTION_SEND
            setOnEditorActionListener { _, a, _ -> if (a == EditorInfo.IME_ACTION_SEND) { sendTyped(); true } else false }
        }
        bar.addView(input, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        fun roundBtn(label: String, color: Int) = TextView(this).apply {
            text = label; textSize = 20f; gravity = Gravity.CENTER; setTextColor(Color.WHITE)
            background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(color) }
        }
        mic = roundBtn("🎤", surface).apply { setOnClickListener { onMic() } }
        send = roundBtn("➤", accent).apply { setOnClickListener { sendTyped() } }
        bar.addView(mic, LinearLayout.LayoutParams(dp(48), dp(48)).apply { leftMargin = dp(8) })
        bar.addView(send, LinearLayout.LayoutParams(dp(48), dp(48)).apply { leftMargin = dp(8) })
        root.addView(bar)
        setContentView(root)

        addInfo("Tap 🎤 and speak (it sends after 2 s of silence), type a command, or tap a preset below.")
        worker.execute { prepare(chipsRow) }
    }

    // ------------------------------------------------------------------ model setup
    private fun unpack(sub: String): File {
        val dst = File(filesDir, "models/$sub"); val done = File(dst, ".complete")
        if (!done.exists()) {
            dst.mkdirs()
            for (f in assets.list("models/$sub") ?: emptyArray()) {
                assets.open("models/$sub/$f").use { i -> File(dst, f).outputStream().use { o -> i.copyTo(o, 1 shl 20) } }
            }
            done.writeText("ok")
        }
        return dst
    }

    private fun prepare(chipsRow: LinearLayout) {
        try {
            setStatus("Unpacking models (first launch only)…", warn)
            val llmDir = unpack("llm"); val sttDir = unpack("stt"); val vadDir = unpack("vad")
            setStatus("Loading models…", warn)
            val t0 = System.nanoTime()
            tokens = Tokens(this)
            llm = LlmEngine(File(llmDir, "model.onnx"), threads)
            stt = MoonshineStt(sttDir, this, threads)
            vad = SileroVad(File(vadDir, "silero_vad.onnx"))
            val loadS = (System.nanoTime() - t0) / 1e9
            val t = tokens!!
            runOnUiThread {
                t.testPrompts.forEachIndexed { i, p ->
                    chipsRow.addView(TextView(this).apply {
                        text = "${i + 1}. ${if (p.length > 34) p.take(32) + "…" else p}"; textSize = 12f; setTextColor(textMain)
                        background = round(card, 16, stroke); setPadding(dp(12), dp(7), dp(12), dp(7))
                        setOnClickListener { if (!busy) worker.execute { handle(p, "preset ${i + 1}", null) } }
                    }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { rightMargin = dp(8) })
                }
            }
            addInfo("Models ready in ${f1(loadS)} s · app RAM ${f1(DeviceStats.rssMb())} MB")
            ready()
        } catch (e: Throwable) { setStatus("Error: ${e.message}", bad) }
    }

    private fun ready() { busy = false; setStatus("Ready", good) }
    private fun setStatus(s: String, c: Int) = runOnUiThread { status.text = s; status.setTextColor(c) }

    // ------------------------------------------------------------------ inputs
    private fun sendTyped() {
        val txt = input.text.toString().trim()
        if (txt.isEmpty() || busy) return
        input.setText("")
        worker.execute { handle(txt, "typed", null) }
    }

    private fun onMic() {
        if (busy && !listening) return
        if (listening) { listening = false; return }                      // tap again = stop now
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return
        }
        worker.execute { listen() }
    }

    override fun onRequestPermissionsResult(rc: Int, p: Array<out String>, g: IntArray) {
        if (rc == 1 && g.firstOrNull() == PackageManager.PERMISSION_GRANTED) worker.execute { listen() }
    }

    /** Record until 2 s of silence after speech (Silero VAD), then STT -> LLM. */
    private fun listen() {
        busy = true; listening = true
        runOnUiThread { mic.background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(bad) } }
        setStatus("Listening… pause for 2 s to send", bad)
        val sr = 16000
        val minBuf = AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuf, 512 * 8))
        val v = vad!!; v.reset()
        val utter = ArrayList<FloatArray>(); val sh = ShortArray(512)
        var speech = false; var speechWins = 0; var silenceWins = 0; var waitWins = 0
        val silenceNeeded = (2.0 * sr / 512).toInt(); val maxWait = (8.0 * sr / 512).toInt(); val maxUtter = (15.0 * sr / 512).toInt()
        rec.startRecording()
        try {
            while (listening) {
                var got = 0; while (got < 512) { val n = rec.read(sh, got, 512 - got); if (n <= 0) break; got += n }
                val win = FloatArray(512) { sh[it] / 32768f }
                val p = v.prob(win)
                if (p >= 0.5f) { speech = true; speechWins++; silenceWins = 0; utter += win }
                else if (speech) { silenceWins++; utter += win; if (silenceWins >= silenceNeeded) break }
                else { waitWins++; if (waitWins > maxWait) break }
                if (utter.size > maxUtter) break
            }
        } finally { rec.stop(); rec.release(); listening = false
            runOnUiThread { mic.background = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(surface) } } }
        if (!speech || speechWins * 512 < 0.25 * sr) { addInfo("No speech detected - discarded."); ready(); return }
        // drop most of the trailing silence (keep 0.3 s)
        val keep = utter.size - maxOf(0, silenceWins - (0.3 * sr / 512).toInt())
        val pcm = FloatArray(keep * 512); for (i in 0 until keep) System.arraycopy(utter[i], 0, pcm, i * 512, 512)
        setStatus("Transcribing…", warn)
        val t0 = System.nanoTime(); val text = stt!!.transcribe(pcm); val sttMs = (System.nanoTime() - t0) / 1e6
        if (!text.any { it.isLetter() }) { addInfo("Heard nothing usable (\"$text\") - discarded."); ready(); return }
        handle(text, "voice · ${f1(pcm.size / sr.toDouble())} s audio", sttMs)
    }

    // ------------------------------------------------------------------ one turn
    private fun handle(command: String, source: String, sttMs: Double?) {
        busy = true
        addUser(command, source)
        setStatus("Generating…", warn)
        val t = tokens!!; val eng = llm!!
        val ids = t.chatIds(command)
        val g = eng.generate(ids, 64, t.eos)
        val out = t.decode(g.tokens).trim()
        val llmMs = g.ttftMs + g.decodeMs
        val ramNow = DeviceStats.rssMb()
        addReply(out, g, llmMs, sttMs, ramNow)
        ready()
    }

    // ------------------------------------------------------------------ bubbles
    private fun addUser(text: String, source: String) = runOnUiThread {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; gravity = Gravity.END }
        box.addView(TextView(this).apply { this.text = source; textSize = 11f; setTextColor(textDim); gravity = Gravity.END })
        box.addView(TextView(this).apply {
            this.text = text; textSize = 15f; setTextColor(Color.WHITE); background = round(accent, 18)
            setPadding(dp(14), dp(10), dp(14), dp(10)); setTextIsSelectable(true)
        }, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { gravity = Gravity.END })
        chat.addView(box, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(14); leftMargin = dp(48) })
        scrollDown()
    }

    private fun addReply(raw: String, g: LlmEngine.GenResult, llmMs: Double, sttMs: Double?, ramNow: Double) = runOnUiThread {
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; background = round(card, 16, stroke); setPadding(dp(14), dp(12), dp(14), dp(12))
        }
        box.addView(TextView(this).apply {
            text = raw.ifBlank { "(empty output)" }; textSize = 14f; typeface = Typeface.MONOSPACE; setTextColor(textMain); setTextIsSelectable(true)
        })
        box.addView(View(this).apply { setBackgroundColor(stroke) }, LinearLayout.LayoutParams(MATCH_PARENT, dp(1)).apply { topMargin = dp(10); bottomMargin = dp(8) })
        val valid = raw.trim().let { it.startsWith("{") && it.endsWith("}") }
        box.addView(TextView(this).apply {
            text = "RAM ${f1(ramNow)} MB  ·  peak ${f1(g.peakRssMb)} MB"; textSize = 13f; setTextColor(good); setTypeface(null, Typeface.BOLD)
        })
        val total = llmMs + (sttMs ?: 0.0)
        val lines = buildString {
            if (sttMs != null) append("Speech → JSON ${f1(total / 1000)} s   ·   STT ${f1(sttMs / 1000)} s   ·   LLM ${f1(llmMs / 1000)} s\n")
            else append("Response ${f1(llmMs / 1000)} s\n")
            append("First token ${f1(g.ttftMs / 1000)} s   ·   decode ${f1(g.decodeTps)} tok/s\n")
            append("Tokens ${g.promptTokens} in → ${g.tokens.size} out   ·   prefill ${f1(g.prefillTps)} tok/s\n")
            append(if (valid) "Valid JSON ✓" else "Not valid JSON ✗")
        }
        box.addView(TextView(this).apply { text = lines; textSize = 12f; setTextColor(textDim); setLineSpacing(0f, 1.15f) })
        chat.addView(box, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(8); rightMargin = dp(24) })
        scrollDown()
    }

    private fun addInfo(s: String) = runOnUiThread {
        chat.addView(TextView(this).apply { text = s; textSize = 12f; setTextColor(textDim); gravity = Gravity.CENTER; setPadding(dp(8), dp(8), dp(8), dp(4)) },
            LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        scrollDown()
    }

    private fun scrollDown() = scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }

    override fun onDestroy() { listening = false; worker.shutdownNow(); llm?.close(); stt?.close(); vad?.close(); super.onDestroy() }
}

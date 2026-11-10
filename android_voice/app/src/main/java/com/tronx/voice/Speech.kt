package com.tronx.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Moonshine v2 streaming-small, ONNX q8 (Workmind export, verified on the Mac: WER 9.07% vs PyTorch 8.42%).
 * Full-utterance mode: encoder once, then greedy merged-decoder loop with KV cache. Same logic as
 * scripts/moonshine_onnx.py.
 */
class MoonshineStt(dir: File, ctxAssets: android.content.Context, threads: Int) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val enc: OrtSession
    private val dec: OrtSession
    private val pastNames: List<String>
    private val hasCacheFlag: Boolean
    private val pieces: Array<String>
    private val special: Set<Int>
    private val start: Int
    private val eos: Int

    init {
        val o = OrtSession.SessionOptions(); o.setIntraOpNumThreads(threads)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        enc = env.createSession(File(dir, "encoder.onnx").absolutePath, o)
        dec = env.createSession(File(dir, "decoder.onnx").absolutePath, o)
        pastNames = dec.inputNames.filter { it.startsWith("past_key_values") }
        hasCacheFlag = dec.inputNames.contains("use_cache_branch")
        val v = JSONObject(ctxAssets.assets.open("stt_vocab.json").bufferedReader().readText())
        val p = v.getJSONArray("pieces"); pieces = Array(p.length()) { p.getString(it) }
        val s = v.getJSONArray("special"); special = (0 until s.length()).map { s.getInt(it) }.toSet()
        start = v.getInt("start"); eos = v.getInt("eos")
    }

    fun transcribe(pcm: FloatArray): String {
        val pad = (80 - pcm.size % 80) % 80                      // v2 frontend needs whole 5 ms frames
        val x = if (pad == 0) pcm else pcm.copyOf(pcm.size + pad)
        val owned = ArrayList<AutoCloseable>()
        try {
            val inp = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, x.size.toLong())); owned += inp
            val er = enc.run(mapOf("input_values" to inp)); owned += er
            val hidden = er.get(0) as OnnxTensor
            val maxLen = maxOf(8, (pcm.size / 16000.0 * 6.5).toInt() + 4)   // model-card guard against loops
            val cache = HashMap<String, OnnxTensor>()
            for (n in pastNames) { val z = OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(1, 8, 0, 64)); owned += z; cache[n] = z }
            val ids = ArrayList<Int>(); var cur = start.toLong()
            for (step in 0 until maxLen) {
                val feed = HashMap<String, OnnxTensor>()
                val ii = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(cur)), longArrayOf(1, 1)); owned += ii
                feed["input_ids"] = ii; feed["encoder_hidden_states"] = hidden; feed.putAll(cache)
                if (hasCacheFlag) { val f = OnnxTensor.createTensor(env, booleanArrayOf(step > 0)); owned += f; feed["use_cache_branch"] = f }
                val r = dec.run(feed); owned += r
                val logits = (r.get("logits").get() as OnnxTensor).floatBuffer
                val vocab = logits.limit()
                var best = 0; var bv = Float.NEGATIVE_INFINITY
                for (i in 0 until vocab) { val vv = logits.get(i); if (vv > bv) { bv = vv; best = i } }
                for (n in pastNames) if (step == 0 || n.contains(".decoder.")) cache[n] = r.get(n.replace("past_key_values", "present")).get() as OnnxTensor
                if (best == eos) break
                ids += best; cur = best.toLong()
            }
            return decode(ids)
        } finally { owned.asReversed().forEach { try { it.close() } catch (_: Exception) {} } }
    }

    /** Llama-style SentencePiece BPE decode: '▁' -> space, <0xNN> byte tokens fused, strip 1 leading space. */
    private fun decode(ids: List<Int>): String {
        val out = java.io.ByteArrayOutputStream()
        val byteTok = Regex("<0x([0-9A-Fa-f]{2})>")
        for (i in ids) {
            if (i in special || i >= pieces.size) continue
            val p = pieces[i]; val m = byteTok.matchEntire(p)
            if (m != null) out.write(m.groupValues[1].toInt(16)) else out.write(p.replace('▁', ' ').toByteArray(Charsets.UTF_8))
        }
        val s = out.toString("UTF-8"); return (if (s.startsWith(" ")) s.substring(1) else s).trim()
    }

    override fun close() { enc.close(); dec.close() }
}

/** Silero VAD v5: 512-sample windows (+64 context) at 16 kHz -> speech probability. */
class SileroVad(file: File) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val s = env.createSession(file.absolutePath, OrtSession.SessionOptions())
    private var state = FloatArray(2 * 128)
    private var ctx = FloatArray(64)

    fun reset() { state = FloatArray(2 * 128); ctx = FloatArray(64) }

    fun prob(win: FloatArray): Float {
        val x = FloatArray(64 + 512); System.arraycopy(ctx, 0, x, 0, 64); System.arraycopy(win, 0, x, 64, 512)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 576)).use { inp ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)).use { st ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(16000)), longArrayOf()).use { sr ->
                    s.run(mapOf("input" to inp, "state" to st, "sr" to sr)).use { r ->
                        val p = (r.get(0) as OnnxTensor).floatBuffer.get(0)
                        val ns = (r.get(1) as OnnxTensor).floatBuffer; val a = FloatArray(ns.remaining()); ns.get(a); state = a
                        System.arraycopy(win, 512 - 64, ctx, 0, 64)
                        return p
                    }
                }
            }
        }
    }

    override fun close() = s.close()
}

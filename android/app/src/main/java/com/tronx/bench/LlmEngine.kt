package com.tronx.bench

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxValue
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Greedy decoder for the Granite-4.0-H hybrid ONNX graph (onnx-community layout):
 * inputs  input_ids, attention_mask, num_logits_to_keep, past_conv.i / past_ssm.i (Mamba layers),
 *         past_key_values.i.key / .value (attention layers)
 * outputs logits, present_conv.i / present_ssm.i, present.i.key / present.i.value
 * Present tensors are fed straight back as the next step's past tensors (no copies).
 */
class LlmEngine(modelFile: File, threads: Int) : AutoCloseable {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    val session: OrtSession
    val stateInputs: List<String>
    val stateType: OnnxJavaType
    private val inputInfo: Map<String, NodeInfo>

    init {
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(threads)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelFile.absolutePath, opts)
        inputInfo = session.inputInfo
        stateInputs = inputInfo.keys.filter { it.startsWith("past_") }
        stateType = (inputInfo[stateInputs.first()]!!.info as TensorInfo).type
    }

    data class GenResult(
        val tokens: List<Int>, val promptTokens: Int, val ttftMs: Double, val prefillMs: Double,
        val decodeMs: Double, val decodeTokens: Int, val kvBytes: Long, val ssmBytes: Long, val kvDtype: String,
        val peakRssMb: Double, val baselineRssMb: Double, val afterRssMb: Double
    ) {
        val decodeTps get() = if (decodeMs > 0) decodeTokens / (decodeMs / 1000.0) else 0.0
        val prefillTps get() = if (prefillMs > 0) promptTokens / (prefillMs / 1000.0) else 0.0
    }

    private fun outName(past: String) = past.replace("past_key_values", "present").replace("past_", "present_")

    private fun zeroState(name: String): OnnxTensor {
        val info = inputInfo[name]!!.info as TensorInfo
        // dynamic dims: batch -> 1, past_sequence_length -> 0
        val shape = info.shape.mapIndexed { i, d -> if (d >= 0) d else if (i == 0) 1L else 0L }.toLongArray()
        val n = shape.fold(1L) { a, b -> a * b }.toInt()
        return when (info.type) {
            OnnxJavaType.FLOAT16 -> OnnxTensor.createTensor(env, ByteBuffer.allocateDirect(n * 2).order(ByteOrder.nativeOrder()), shape, OnnxJavaType.FLOAT16)
            else -> OnnxTensor.createTensor(env, FloatBuffer.allocate(n), shape)
        }
    }

    private fun halfToFloat(h: Int): Float {
        val s = (h shr 15) and 1; val e = (h shr 10) and 0x1f; val m = h and 0x3ff
        val v = when (e) {
            0 -> (m / 1024.0f) * Math.pow(2.0, -14.0).toFloat()
            31 -> if (m == 0) Float.POSITIVE_INFINITY else Float.NaN
            else -> (1 + m / 1024.0f) * Math.pow(2.0, (e - 15).toDouble()).toFloat()
        }
        return if (s == 1) -v else v
    }

    private fun argmaxLast(logits: OnnxTensor): Int {
        val info = logits.info
        val vocab = info.shape.last().toInt()
        return if (info.type == OnnxJavaType.FLOAT16) {
            val bb = logits.byteBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
            val off = bb.limit() - vocab
            var best = 0; var bv = Float.NEGATIVE_INFINITY
            for (i in 0 until vocab) { val v = halfToFloat(bb.get(off + i).toInt() and 0xffff); if (v > bv) { bv = v; best = i } }
            best
        } else {
            val fb = logits.floatBuffer
            val off = fb.limit() - vocab
            var best = 0; var bv = Float.NEGATIVE_INFINITY
            for (i in 0 until vocab) { val v = fb.get(off + i); if (v > bv) { bv = v; best = i } }
            best
        }
    }

    private fun tensorBytes(v: OnnxValue): Long {
        val t = v as OnnxTensor
        val n = t.info.shape.fold(1L) { a, b -> a * b }
        val sz = when (t.info.type) { OnnxJavaType.FLOAT16, OnnxJavaType.BFLOAT16 -> 2; OnnxJavaType.FLOAT -> 4; else -> 8 }
        return n * sz
    }

    /**
     * Greedy generation. Timing boundary: TTFT = from the start of the first session.run (token ids already
     * prepared, model already loaded) until the first generated token id is known (prefill + first argmax).
     * If [forceTokens] > 0 EOS is ignored and exactly that many tokens are generated (fixed-length KV tests).
     */
    fun generate(promptIds: IntArray, maxNew: Int, eos: Int, forceTokens: Int = 0): GenResult {
        val sampler = DeviceStats.PeakSampler(); val baseline = DeviceStats.rssMb(); sampler.start()
        val owned = mutableListOf<OnnxTensor>()
        var prev: OrtSession.Result? = null
        val states = HashMap<String, OnnxValue>()
        for (s in stateInputs) { val z = zeroState(s); owned += z; states[s] = z }
        var ids = promptIds.map { it.toLong() }.toLongArray()
        var total = ids.size
        val out = ArrayList<Int>()
        var ttft = 0.0; var decodeMs = 0.0; var kv = 0L; var ssm = 0L
        val limit = if (forceTokens > 0) forceTokens else maxNew
        val keep = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(1)), longArrayOf()); owned += keep
        val t0 = System.nanoTime()
        try {
            for (step in 0 until limit) {
                val tStep = System.nanoTime()
                val inIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()))
                val mask = OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(total) { 1L }), longArrayOf(1, total.toLong()))
                val feed = HashMap<String, OnnxTensor>()
                feed["input_ids"] = inIds; feed["attention_mask"] = mask; feed["num_logits_to_keep"] = keep
                for ((k, v) in states) feed[k] = v as OnnxTensor
                val res = session.run(feed)
                val next = argmaxLast(res.get("logits").get() as OnnxTensor)
                val dt = (System.nanoTime() - tStep) / 1e6
                if (step == 0) ttft = (System.nanoTime() - t0) / 1e6 else decodeMs += dt
                inIds.close(); mask.close()
                // swap states: new present -> next past
                for (s in stateInputs) states[s] = res.get(outName(s)).get()
                prev?.close(); prev = res
                out += next
                if (forceTokens == 0 && next == eos) break
                ids = longArrayOf(next.toLong()); total += 1
            }
            for (s in stateInputs) {
                val b = tensorBytes(states[s]!!)
                if (s.startsWith("past_key_values")) kv += b else ssm += b
            }
        } finally {
            prev?.close(); owned.forEach { it.close() }
        }
        val peak = sampler.stop()
        val decodeTokens = (out.size - 1).coerceAtLeast(0)
        return GenResult(out, promptIds.size, ttft, ttft, decodeMs, decodeTokens, kv, ssm, stateType.name, peak, baseline, DeviceStats.rssMb())
    }

    override fun close() { session.close() }
}

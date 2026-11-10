package com.tronx.bench

import android.content.Context
import org.json.JSONObject

/**
 * Prompts are PRE-TOKENISED on the Mac with the exact Granite tokenizer + chat template (assets/prompts.json),
 * so every model variant receives identical token ids (assignment Section 9E). Only decoding happens on device.
 * Decoding = byte-level BPE: token string -> GPT-2 byte map -> UTF-8 bytes.
 */
class Tokens(ctx: Context) {
    val eos: Int
    val prompts: List<Pair<String, IntArray>>          // (command text, ids incl. schema system prompt + chat template)
    val ctxPrompts: Map<Int, IntArray>                 // target context length -> ids (exact length)
    val schemaPromptTokens: Int
    private val id2tok: Array<String?>
    private val special: Set<Int>
    private val byteDec: Map<Char, Int>
    private val byteEnc = CharArray(256)
    private val tok2id = HashMap<String, Int>(110_000)
    private val ranks = HashMap<String, Int>(110_000)
    private val cache = HashMap<String, IntArray>()
    val chatPrefix: IntArray
    val chatSuffix: IntArray
    val testPrompts: List<String>
    // Granite pre-tokenizer regex (GPT-4 style). Android's java.util.regex is ICU-based: \s and \p{L} are already
    // Unicode-aware and the UNICODE_CHARACTER_CLASS flag is NOT supported (it threw IllegalArgumentException on device).
    private val pat = java.util.regex.Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")

    init {
        val p = JSONObject(ctx.assets.open("prompts.json").bufferedReader().readText())
        eos = p.getInt("eos")
        schemaPromptTokens = p.getInt("schema_prompt_tokens")
        val arr = p.getJSONArray("bench_prompts")
        prompts = (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i); val ids = o.getJSONArray("ids")
            o.getString("text") to IntArray(ids.length()) { ids.getInt(it) }
        }
        val c = p.getJSONObject("ctx_prompts")
        ctxPrompts = c.keys().asSequence().associate { k -> val a = c.getJSONArray(k); k.toInt() to IntArray(a.length()) { a.getInt(it) } }
        val v = JSONObject(ctx.assets.open("vocab.json").bufferedReader().readText())
        val maxId = v.keys().asSequence().maxOf { v.getInt(it) }
        id2tok = arrayOfNulls(maxId + 1)
        v.keys().forEach { k -> id2tok[v.getInt(k)] = k; tok2id[k] = v.getInt(k) }
        ctx.assets.open("merges.txt").bufferedReader(Charsets.UTF_8).useLines { lines ->
            var r = 0; lines.forEach { if (it.isNotEmpty()) { ranks[it] = r; r++ } }
        }
        val cp = p.getJSONArray("chat_prefix_ids"); chatPrefix = IntArray(cp.length()) { cp.getInt(it) }
        val cs2 = p.getJSONArray("chat_suffix_ids"); chatSuffix = IntArray(cs2.length()) { cs2.getInt(it) }
        val tp = p.getJSONArray("test_prompts"); testPrompts = (0 until tp.length()).map { tp.getString(it) }
        val sp = p.getJSONArray("special_ids"); special = (0 until sp.length()).map { sp.getInt(it) }.toSet()
        // GPT-2 bytes_to_unicode inverse
        val bs = ArrayList<Int>(); (33..126).forEach { bs += it }; (161..172).forEach { bs += it }; (174..255).forEach { bs += it }
        val cs = ArrayList(bs); var n = 0
        for (b in 0..255) if (b !in bs) { bs += b; cs += 256 + n; n++ }
        byteDec = bs.indices.associate { cs[it].toChar() to bs[it] }
        bs.indices.forEach { byteEnc[bs[it]] = cs[it].toChar() }
    }

    /** Byte-level BPE, identical algorithm to scripts/bpe_reference.py (verified 0 mismatches vs HF on 3,484 texts). */
    private fun bpe(word: String): IntArray {
        cache[word]?.let { return it }
        var sym = word.map { it.toString() }.toMutableList()
        while (sym.size > 1) {
            var best = Int.MAX_VALUE; var bi = -1
            for (i in 0 until sym.size - 1) { val r = ranks[sym[i] + " " + sym[i + 1]]; if (r != null && r < best) { best = r; bi = i } }
            if (bi < 0) break
            val a = sym[bi]; val b = sym[bi + 1]; val out = ArrayList<String>(sym.size)
            var i = 0
            while (i < sym.size) { if (i < sym.size - 1 && sym[i] == a && sym[i + 1] == b) { out += a + b; i += 2 } else { out += sym[i]; i++ } }
            sym = out
        }
        val ids = IntArray(sym.size) { tok2id[sym[it]] ?: 0 }
        cache[word] = ids; return ids
    }

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        val m = pat.matcher(text)
        while (m.find()) {
            val piece = m.group().toByteArray(Charsets.UTF_8)
            val w = StringBuilder(); for (b in piece) w.append(byteEnc[b.toInt() and 0xff])
            bpe(w.toString()).forEach { out += it }
        }
        return out.toIntArray()
    }

    /** Full model input: schema system prompt + chat template around the user's command. */
    fun chatIds(command: String): IntArray = chatPrefix + encode(command) + chatSuffix

    fun decode(ids: List<Int>): String {
        val out = java.io.ByteArrayOutputStream()
        for (id in ids) {
            if (id in special || id >= id2tok.size) continue
            val t = id2tok[id] ?: continue
            for (ch in t) { val b = byteDec[ch]; if (b != null) out.write(b) }
        }
        return out.toString("UTF-8")
    }
}

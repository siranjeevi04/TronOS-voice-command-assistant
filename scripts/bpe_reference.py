"""Pure re-implementation of the Granite byte-level BPE (same algorithm as the Android Kotlin port), used to
verify the on-device tokenizer token-for-token against Hugging Face."""
import json
import regex
PAT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPE:
    def __init__(self, vocab_path, merges_path):
        self.vocab = json.load(open(vocab_path))
        self.ranks = {}
        for i, line in enumerate(open(merges_path, encoding="utf-8").read().split("\n")):
            if line:
                a, b = line.split(" ")
                self.ranks[(a, b)] = i
        self.b2u = bytes_to_unicode(); self.cache = {}

    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        sym = list(word)
        while len(sym) > 1:
            best, bi = None, -1
            for i in range(len(sym) - 1):
                r = self.ranks.get((sym[i], sym[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            a, b = sym[bi], sym[bi + 1]; out = []; i = 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(sym[i]); i += 1
            sym = out
        ids = [self.vocab[s] for s in sym]; self.cache[word] = ids; return ids

    def encode(self, text):
        out = []
        for m in regex.findall(PAT, text):
            out += self.bpe("".join(self.b2u[b] for b in m.encode("utf-8")))
        return out

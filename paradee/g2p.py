"""Paradee's own grapheme-to-phoneme model: text -> the phonemes Paradee was trained on, without eSpeak.

Paradee learned Hindi, French and Spanish from eSpeak NG's phonemes, and English from misaki, which hands
the words it has no entry for (names, mostly) to eSpeak. eSpeak is GPL-3.0, so it can't ship inside a
closed-source app. This module gives the same phonemes from three small Apache-2.0 parts, all in data/:

- a 0.76M-parameter model (g2p.onnx, 0.9 MB int8) that reads one word, plus the last letter of the word
  before it and the first letter of the word after it, and writes the word's phonemes with stress marks.
  The neighbouring letters are enough for French liaison ("les amis" -> lez), the Spanish b, d and g that
  soften after a vowel, and the stress eSpeak puts at the end of a phrase.
- word lists (g2p.json.gz): the words the model gets wrong, and the 3,000 most common words of each language
  so they skip the model.
- a table of number pieces per language (0 to 99, the hundreds, thousand, lakh, million...) that numbers
  are put together from.

The model and lists were learned from eSpeak's output on 400k Wikipedia sentences per language. On held-out
sentences they differ from eSpeak in 0.80% (Hindi), 0.95% (French) and 0.59% (Spanish) of phoneme
characters, mostly a moved stress mark.
"""
import gzip, json, os, re, unicodedata
import numpy as np
import onnxruntime as ort

_DATA = os.path.join(os.path.dirname(__file__), "data")
_PERCENT = {"hi": "प्रतिशत", "fr": "pour cent", "es": "por ciento"}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_MAX_WORD = 60   # letters the model reads of one word


def _is_edge(c):
    """Punctuation, symbols, spaces and zero-width characters, which are trimmed off the ends of a word."""
    return unicodedata.category(c)[0] in "PSZ" or c in "​‌‍﻿"


def _strip(w):
    i, j = 0, len(w)
    while i < j and _is_edge(w[i]):
        i += 1
    while j > i and _is_edge(w[j - 1]):
        j -= 1
    return w[i:j]


def _words(text):
    """Each whitespace-separated piece of `text` as (word, punctuation before it?, punctuation after it?).
    A piece with no letters or digits is (None, True, True)."""
    out = []
    for raw in text.split():
        core = _strip(raw)
        if not core or not any(c.isalnum() for c in core):
            out.append((None, True, True))
            continue
        i = raw.find(core)
        out.append((core, i > 0, i + len(core) < len(raw)))
    return out


def _contexts(words):
    """(before, after) for each word: the last letter of the word before and the first letter of the word
    after, lowercased, or "#" when punctuation or the end of the sentence is in between."""
    res = []
    for i, (w, lead, trail) in enumerate(words):
        if w is None:
            res.append(None)
            continue
        if i == 0 or lead or words[i - 1][0] is None or words[i - 1][2]:
            before = "#"
        else:
            before = words[i - 1][0][-1].lower()
        if i == len(words) - 1 or trail or words[i + 1][0] is None or words[i + 1][1]:
            after = "#"
        else:
            after = words[i + 1][0][0].lower()
        res.append((before, after))
    return res


# Numbers, put together from pieces the way eSpeak reads them.
def _unstress(s):
    return s.replace("ˈ", "").replace("ˌ", "")


def _hi(n, t):
    if n < 100:
        return [t[str(n)]]
    out = []
    for div, name in [(10**7, "crore"), (10**5, "lakh"), (1000, "thousand")]:
        if n >= div:
            out += _hi(n // div, t) + [t[name]]
            n %= div
    if n >= 100:
        out.append(t[str(n // 100 * 100)])
        n %= 100
    if n:
        out.append(t[str(n)])
    return out


def _fr(n, t, top=True):
    out = []
    for div, one, many in [(10**9, "milliard", "milliards"), (10**6, "million", "millions")]:
        if n >= div:
            q = n // div
            out += (["œ̃"] if q == 1 else _fr(q, t, False)) + [t[one if q == 1 else many]]
            n %= div
    if n >= 1000:
        q = n // 1000
        out += ([] if q == 1 else _fr(q, t, False)) + [t["thousand"]]
        n %= 1000
    if n >= 100:
        out.append(t[str(n // 100 * 100)])
        n %= 100
    if n or not out:
        out.append(t[str(n)])
    if top:   # French stresses only the last word of a number
        out = [_unstress(x) for x in out[:-1]] + out[-1:]
    return out


def _es(n, t):
    out = []
    if n >= 10**6:
        q = n // 10**6
        out += [t["millón"]] if q == 1 else _es(q, t) + [t["millones"]]
        n %= 10**6
    if n >= 1000:
        q = n // 1000
        if q == 1:
            out.append(t["thousand"])
        else:
            p = _es(q, t)
            if q % 10 == 1 and q % 100 != 11 and p[-1].endswith("o"):   # veintiuno mil -> veintiún mil
                p[-1] = p[-1][:-1]
            out += p + [t["thousand"]]
        n %= 1000
    if n >= 100:
        h = n // 100 * 100
        n %= 100
        out.append(t["100+"] if h == 100 and n else t[str(h)])
    if n or not out:
        out.append(t[str(n)])
    return out


_NUMBERS = {"hi": _hi, "fr": _fr, "es": _es}
_SOFT = {"b": "β", "d": "ð", "ɡ": "ɣ"}


def _soften(p, before):
    """Spanish b, d and g are said β, ð and ɣ after a vowel and most consonants, also across words, as eSpeak
    writes them. `before` is the sound or letter before, "#" after a pause."""
    i = 0
    while i < len(p) and p[i] in "ˈˌ":
        i += 1
    c = p[i:i + 1]
    if c in _SOFT and before not in "#nmɲŋ" and not (c == "d" and before == "l"):
        p = p[:i] + _SOFT[c] + p[i + 1:]
    return p


class G2P:
    """Phonemes for Paradee without eSpeak. One instance serves every language.

    g2p("Bonjour à tous.", "fr") -> "bɔ̃ʒˈuʁ a tˈus."   languages: "hi", "fr", "es"
    g2p.english_fallback(british) -> a fallback for misaki's English G2P, in place of misaki's EspeakFallback
    """

    def __init__(self, threads=1):
        with gzip.open(os.path.join(_DATA, "g2p.json.gz"), "rt", encoding="utf-8") as f:
            data = json.load(f)
        self.ins, self.outs = data["vocab"]["in"], data["vocab"]["out"]
        self.index = {c: i for i, c in enumerate(self.ins)}
        self.numbers = data["numbers"]
        self.lexicon = {lang: (d["w"], {tuple(k.split("\t")): v for k, v in d["c"].items()})
                        for lang, d in data["lexicon"].items()}
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(os.path.join(_DATA, "g2p.onnx"), so, providers=["CPUExecutionProvider"])

    def model(self, items):
        """Phonemes for each (language, letter before, word, letter after), from the model alone."""
        if not items:
            return []
        ix, unk = self.index, self.index["<unk>"]
        rows = [[ix[f"<{lang}>"], ix.get(a, unk), ix["<s1>"]] + [ix.get(c, unk) for c in w.lower()[:_MAX_WORD]]
                + [ix["<s2>"], ix.get(b, unk)] for lang, a, w, b in items]
        T = max(map(len, rows))
        chars = np.zeros((len(rows), T), dtype=np.int64)
        pad = np.ones((len(rows), T), dtype=bool)
        for i, r in enumerate(rows):
            chars[i, :len(r)] = r
            pad[i, :len(r)] = False
        best = self.session.run(None, {"chars": chars, "pad": pad})[0]
        res = []
        for r, row in zip(best, rows):   # CTC: merge repeats, drop blanks (0); each letter has 3 output steps
            out, prev = [], 0
            for k in r[:3 * len(row)]:
                if k != prev and k != 0:
                    out.append(self.outs[k])
                prev = k
            res.append("".join(out))
        return res

    def words(self, lang, items):
        """Phonemes for each (word, letter before, letter after): the word list first, then the model."""
        words, in_context = self.lexicon.get(lang, ({}, {}))
        out = [in_context.get((w, a, b)) or words.get(w) for w, a, b in items]
        todo = [i for i, p in enumerate(out) if p is None]
        for i, p in zip(todo, self.model([(lang, items[i][1], items[i][0], items[i][2]) for i in todo])):
            out[i] = p
        return out

    def number(self, n, lang, before="#"):
        parts = _NUMBERS[lang](n, self.numbers[lang])
        if lang == "es":
            soft = []
            for k, p in enumerate(parts):
                soft.append(_soften(p, before if k == 0 else soft[-1][-1]))
            parts = soft
        return " ".join(parts)

    def __call__(self, text, lang):
        """One sentence of Hindi ("hi"), French ("fr") or Spanish ("es") -> phonemes, one space between words,
        punctuation kept, like misaki's EspeakG2P."""
        text = text.translate(_DEVANAGARI_DIGITS).replace("«", "“").replace("»", "”")
        text = re.sub("[​‌‍﻿]", "", text)
        text = re.sub(r"([।॥])(?=\S)", r"\1 ", text)
        if lang == "fr":   # 12 000 is one number
            text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)
        text = re.sub(r"(\d)\s?%", lambda m: m.group(1) + " " + _PERCENT[lang], text)
        words = _words(text)
        jobs, plan = [], []
        for raw, (core, _, _), ctx in zip(text.split(), words, _contexts(words)):
            if core is None:
                plan.append((raw, None, ""))
                continue
            i = raw.find(core)
            parts = []
            for piece in re.findall(r"\d+|[^\d]+", core):
                if piece.isdigit():
                    parts.append(("number", piece))
                else:
                    piece = piece.strip("-.,/:")
                    if any(c.isalnum() for c in piece):
                        parts.append(("word", piece))
            ids = []
            for kind, piece in parts:
                first, last = piece == parts[0][1], piece == parts[-1][1]
                if kind == "number":
                    if len(piece) < 10:
                        ids.append(("text", self.number(int(piece), lang, ctx[0] if first else "#")))
                    else:   # a long string of digits is read digit by digit
                        ids.append(("text", " ".join(self.number(int(d), lang) for d in piece)))
                else:
                    jobs.append((piece, ctx[0] if first else "#", ctx[1] if last else "#"))
                    ids.append(("job", len(jobs) - 1))
            plan.append((raw[:i], ids, raw[i + len(core):]))
        res = self.words(lang, jobs)
        out = []
        for lead, ids, trail in plan:
            if ids is None:
                out.append(lead)
            else:
                out.append(lead + " ".join(x if kind == "text" else res[x] for kind, x in ids) + trail)
        return " ".join(out).replace("-", "")

    def english_fallback(self, british):
        """For misaki's en.G2P(fallback=...): phonemes for the English words misaki has no entry for."""
        return _EnglishFallback(self, british)


class _EnglishFallback:
    def __init__(self, g2p, british):
        from misaki import en
        self.g2p, self.lang = g2p, "en-gb" if british else "en-us"
        self.plain = en.G2P(trf=False, british=british, fallback=None, unk="")   # reads digits on its own

    def __call__(self, token):
        """Like eSpeak, keeps the punctuation around the word ("St." -> "sənt.")."""
        words = self.g2p.lexicon[self.lang][0]
        out = []
        for piece in token.text.split():
            core = _strip(piece)
            if not core or not any(c.isalnum() for c in core):
                continue
            i = piece.find(core)
            said = []
            for part in re.findall(r"\d+|\D+", core):   # digits ("22:", "Se8") are read by misaki itself
                if part.isdigit():
                    said.append(self.plain(part)[0])
                else:
                    part = part.strip("-.,/:'")
                    if any(c.isalnum() for c in part):
                        said.append(words.get(part) or self.g2p.model([(self.lang, "#", part, "#")])[0])
            out.append(piece[:i] + " ".join(said) + piece[i + len(core):])
        return " ".join(out), 2

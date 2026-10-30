import re

NEGATORS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "cannot", "without"}

SCOPE_BREAKERS = {"but", "however", "although", "though", "yet"}

_CONTRACTIONS = [
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"\bshan['’]t\b"), "shall not"),
    (re.compile(r"\b(\w+)n['’]t\b"), r"\1 not"),
]

_TOKEN = re.compile(r"[a-z0-9_']+|[.,!?;:()\"]")
_PUNCT = set(".,!?;:()\"")


def expand_contractions(text):
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def mark_negation(text, window=None):
    text = expand_contractions(text)
    out, remaining = [], 0
    for tok in _TOKEN.findall(text):
        if tok in _PUNCT or tok in SCOPE_BREAKERS:
            remaining = 0
            out.append(tok)
        elif tok in NEGATORS:
            remaining = window if window else 10_000
            out.append(tok)
        elif remaining > 0:
            out.append("not_" + tok)
            remaining -= 1
        else:
            out.append(tok)
    return " ".join(out)


if __name__ == "__main__":
    for s in [
        "the film was not boring at all.",
        "i didn't like the ending, but the music was great.",
        "it was not a great movie.",
        "i would never recommend this movie to anyone.",
    ]:
        print("{}\n  -> {}\n  -> window=2: {}".format(s, mark_negation(s), mark_negation(s, window=2)))

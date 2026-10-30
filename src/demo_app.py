
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

from baseline import build_pipeline
from data_prep import PROJECT_ROOT, clean_text, prepare_data, training_pool
from negation import mark_negation

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
FINAL_C = 30
HOST, PORT = "127.0.0.1", 8000


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    print("No saved model found, training the final system on 25,000 reviews ...")
    _, val_df = prepare_data()
    full = pd.concat([training_pool(val_df), val_df], ignore_index=True)
    pipe = build_pipeline("LogReg", (1, 2), negation=True, strength=FINAL_C)
    pipe.fit(full["clean_text"], full["label"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


MODEL = None


def explain(review, n=8):
    text = clean_text(review)
    tfidf, clf = MODEL.named_steps["tfidf"], MODEL.named_steps["clf"]
    x = tfidf.transform([text])
    p_pos = float(clf.predict_proba(x)[0, 1])
    vocab = tfidf.get_feature_names_out()
    contrib = [(vocab[j], float(v * clf.coef_[0, j])) for j, v in zip(x.indices, x.data)]
    contrib.sort(key=lambda t: t[1])
    return {
        "label": "positive" if p_pos >= 0.5 else "negative",
        "confidence": max(p_pos, 1 - p_pos),
        "p_positive": p_pos,
        "marked": mark_negation(text),
        "negative_terms": [{"term": t, "score": round(s, 3)} for t, s in contrib[:n] if s < 0],
        "positive_terms": [{"term": t, "score": round(s, 3)} for t, s in contrib[::-1][:n] if s > 0],
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path in ("/", "/index.html"):
            self._send((Path(__file__).parent / "demo_page.html").read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/predict":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        review = json.loads(self.rfile.read(length) or b"{}").get("text", "")[:5000]
        self._send(json.dumps(explain(review)), "application/json")

    def log_message(self, *args):
        pass


def main():
    global MODEL
    MODEL = load_model()
    url = f"http://{HOST}:{PORT}"
    print("Demo running at {}  (Ctrl+C to stop)".format(url))
    try:
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()

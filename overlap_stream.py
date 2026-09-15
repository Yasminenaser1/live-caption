import json, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from wer import score

N = 50
# (window seconds, hop seconds) — hop sets how often new text appears
CONFIGS = [(3.0, 1.5), (3.0, 2.0), (5.0, 2.5), (5.0, 3.0)]

pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def stitch(prev, new, lookback=25):
    """Find where `new` best re-joins `prev` and append only what follows."""
    norm = lambda w: "".join(c for c in w.lower() if c.isalnum())
    p, n = prev.split(), new.split()
    if not p or not n:
        return " ".join(p + n)
    pn, nn = [norm(w) for w in p], [norm(w) for w in n]

    best_score, best_cut = 0, 0
    # try aligning new[0:] against each tail position of prev
    for start in range(max(0, len(p) - lookback), len(p)):
        span = len(p) - start
        if span > len(n):
            continue
        matches = sum(1 for a, b in zip(pn[start:], nn[:span]) if a == b and a)
        score_ = matches / span
        # need a solid majority to believe it's the same span
        if score_ >= 0.6 and matches > best_score:
            best_score, best_cut = matches, span
    return " ".join(p + n[best_cut:])

print(f"{'config':>22} {'WER':>8} {'new text every':>15} {'compute':>9}")
print(f"{'full utterance':>22} {'4.31%':>8} {'-':>15} {'1.0x':>9}")
print(f"{'fixed 3s (no overlap)':>22} {'11.46%':>8} {'3000ms':>15} {'1.0x':>9}")
print(f"{'VAD only':>22} {'3.79%':>8} {'variable':>15} {'1.0x':>9}")

for win, hop in CONFIGS:
    refs, hyps, lat = [], [], []
    for p in pairs:
        audio, sr = sf.read(p["audio"], dtype="float32")
        w, h = int(win * sr), int(hop * sr)
        text = ""
        for start in range(0, len(audio), h):
            piece = audio[start:start + w]
            if len(piece) < sr * 0.3:
                continue
            t0 = time.perf_counter()
            segs, _ = model.transcribe(piece, beam_size=1)
            chunk_text = " ".join(x.text for x in segs).strip()
            compute = (time.perf_counter() - t0) * 1000
            text = stitch(text, chunk_text) if text else chunk_text
            lat.append(hop * 1000 + compute)
        refs.append(p["reference"]); hyps.append(text)

    m = score(refs, hyps)
    print(f"{'win=' + str(win) + 's hop=' + str(hop) + 's':>22} {m['wer']*100:>7.2f}% "
          f"{np.percentile(lat,50):>13.0f}ms {win/hop:>8.1f}x")

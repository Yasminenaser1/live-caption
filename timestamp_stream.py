import json, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from wer import score

N = 50
CONFIGS = [(3.0, 1.5), (3.0, 2.0), (5.0, 2.5), (5.0, 3.0)]
# words whose start times fall within this many ms are treated as the same word
DEDUP_MS = 80   # duplicates land ~40ms apart; real words ~140ms

pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

print(f"{'config':>22} {'WER':>8} {'new text every':>15} {'compute':>9}")
print(f"{'full utterance':>22} {'4.31%':>8} {'-':>15} {'1.0x':>9}")
print(f"{'VAD only':>22} {'3.79%':>8} {'variable':>15} {'1.0x':>9}")
print(f"{'fixed 3s':>22} {'11.46%':>8} {'3000ms':>15} {'1.0x':>9}")

for win, hop in CONFIGS:
    refs, hyps, lat = [], [], []
    for p in pairs:
        audio, sr = sf.read(p["audio"], dtype="float32")
        w, h = int(win * sr), int(hop * sr)
        words = []   # (absolute_start_seconds, text)

        for start in range(0, len(audio), h):
            piece = audio[start:start + w]
            if len(piece) < sr * 0.3:
                continue
            offset = start / sr
            t0 = time.perf_counter()
            segs, _ = model.transcribe(piece, beam_size=1, word_timestamps=True)
            for s in segs:
                for wd in (s.words or []):
                    words.append((offset + wd.start, wd.word.strip()))
            lat.append(hop * 1000 + (time.perf_counter() - t0) * 1000)

        # keep the first occurrence of each time position
        words.sort(key=lambda x: x[0])
        norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
        kept = []
        for t, txt in words:
            dup = any(abs(t - pt) <= DEDUP_MS / 1000 and norm(txt) == norm(ptxt)
                      for pt, ptxt in kept[-4:])
            if not dup and norm(txt):
                kept.append((t, txt))
        kept = [txt for _, txt in kept]
        refs.append(p["reference"]); hyps.append(" ".join(kept))

    m = score(refs, hyps)
    print(f"{'win=' + str(win) + 's hop=' + str(hop) + 's':>22} {m['wer']*100:>7.2f}% "
          f"{np.percentile(lat,50):>13.0f}ms {win/hop:>8.1f}x")

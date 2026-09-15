import json, time
import numpy as np
from faster_whisper import WhisperModel
from wer import score

N = 100
pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

refs, hyps, rtfs = [], [], []
total_audio = total_compute = 0.0

for i, p in enumerate(pairs, 1):
    t0 = time.perf_counter()
    segs, info = model.transcribe(p["audio"], beam_size=1)
    text = " ".join(s.text for s in segs)
    dt = time.perf_counter() - t0

    refs.append(p["reference"]); hyps.append(text)
    rtfs.append(info.duration / dt)
    total_audio += info.duration; total_compute += dt
    if i % 25 == 0: print(f"  {i}/{N}")

m = score(refs, hyps)
print(f"\nbaseline — {N} utterances, {total_audio/60:.1f} min of audio")
print(f"  WER           {m['wer']*100:.2f}%")
print(f"  sub/del/ins   {m['substitutions']} / {m['deletions']} / {m['insertions']}")
print(f"  realtime      {total_audio/total_compute:.1f}x (median {np.median(rtfs):.1f}x)")

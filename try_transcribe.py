import json, time
from faster_whisper import WhisperModel

pairs = json.load(open("data/pairs.json"))
p = pairs[0]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

t0 = time.time()
segments, info = model.transcribe(p["audio"], beam_size=1)
text = " ".join(s.text for s in segments).strip().lower()
elapsed = time.time() - t0

print(f"audio {info.duration:.1f}s | transcribed in {elapsed:.1f}s "
      f"| {info.duration/elapsed:.1f}x realtime\n")
print("REF:", p["reference"][:120])
print("HYP:", text[:120])

import json
import soundfile as sf
from faster_whisper import WhisperModel

p = json.load(open("data/pairs.json"))[0]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
audio, sr = sf.read(p["audio"], dtype="float32")

print("REF:", p["reference"][:110], "\n")
for start in range(0, len(audio), int(1.5 * sr)):
    piece = audio[start:start + int(3.0 * sr)]
    if len(piece) < sr * 0.3:
        continue
    segs, _ = model.transcribe(piece, beam_size=1)
    print(f"[{start/sr:>4.1f}s] {' '.join(x.text for x in segs).strip()}")

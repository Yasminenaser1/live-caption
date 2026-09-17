import json
import soundfile as sf
from faster_whisper import WhisperModel

p = json.load(open("data/pairs.json"))[0]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
audio, sr = sf.read(p["audio"], dtype="float32")
w, h = int(3.0*sr), int(1.5*sr)

words = []
for start in range(0, len(audio), h):
    piece = audio[start:start+w]
    if len(piece) < sr*0.3: continue
    segs, _ = model.transcribe(piece, beam_size=1, word_timestamps=True)
    for s in segs:
        for wd in (s.words or []):
            words.append((start/sr + wd.start, wd.word.strip()))

words.sort(key=lambda x: x[0])
print("all words with absolute times:\n")
for t, txt in words[:30]:
    print(f"  {t:>5.2f}s  {txt}")

gaps = [words[i+1][0]-words[i][0] for i in range(len(words)-1)]
print(f"\nmedian gap between consecutive words: {sorted(gaps)[len(gaps)//2]*1000:.0f}ms")
print(f"REF: {p['reference'][:100]}")

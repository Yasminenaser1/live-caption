import json, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from wer import score

CHUNKS = [1.0, 2.0, 3.0, 5.0, 10.0]   # seconds of audio per transcription
N = 50
SR = 16000

pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

print(f"{'chunk':>7} {'WER':>8} {'latency p50':>13} {'realtime':>10}")
print(f"{'full':>7} {'4.31%':>8} {'(utterance)':>13} {'57.7x':>10}")

for chunk_s in CHUNKS:
    refs, hyps, lat = [], [], []
    for p in pairs:
        audio, sr = sf.read(p["audio"], dtype="float32")
        step = int(chunk_s * sr)
        parts = []
        for start in range(0, len(audio), step):
            piece = audio[start:start + step]
            if len(piece) < sr * 0.2:      # skip slivers under 200ms
                continue
            t0 = time.perf_counter()
            segs, _ = model.transcribe(piece, beam_size=1)
            parts.append(" ".join(s.text for s in segs))
            # latency a listener feels: wait for the chunk + time to transcribe it
            lat.append(chunk_s * 1000 + (time.perf_counter() - t0) * 1000)
        refs.append(p["reference"]); hyps.append(" ".join(parts))

    m = score(refs, hyps)
    print(f"{chunk_s:>6.1f}s {m['wer']*100:>7.2f}% {np.percentile(lat,50):>11.0f}ms "
          f"{chunk_s*1000/np.percentile(lat,50)*1:>9.1f}x")

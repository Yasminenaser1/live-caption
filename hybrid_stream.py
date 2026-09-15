import json, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from wer import score

N = 50
MAX_WAITS = [2.0, 3.0, 5.0]   # hard cap on how long we'll wait for a pause
MIN_SIL = 200

pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
opts = VadOptions(min_silence_duration_ms=MIN_SIL, speech_pad_ms=100)

def cut_points(audio, sr, max_wait):
    """Boundaries in samples. Prefer a VAD speech-end; force a cut at max_wait."""
    ends = [s["end"] for s in get_speech_timestamps(audio, opts)]
    cuts, pos = [], 0
    cap = int(max_wait * sr)
    while pos < len(audio):
        limit = pos + cap
        # latest VAD boundary that falls inside the budget
        candidates = [e for e in ends if pos < e <= limit]
        cut = max(candidates) if candidates else min(limit, len(audio))
        cuts.append((pos, cut))
        pos = cut
    return cuts

print(f"{'config':>24} {'WER':>8} {'p50':>9} {'p95':>9} {'max':>9} {'forced':>8}")
print(f"{'full utterance':>24} {'4.31%':>8} {'-':>9} {'-':>9} {'-':>9} {'-':>8}")
print(f"{'fixed 3s':>24} {'11.46%':>8} {'3104ms':>9} {'-':>9} {'-':>9} {'100%':>8}")
print(f"{'VAD only (200ms)':>24} {'3.79%':>8} {'4228ms':>9} {'11457ms':>9} {'-':>9} {'0%':>8}")

for max_wait in MAX_WAITS:
    refs, hyps, lat = [], [], []
    forced = total = 0

    for p in pairs:
        audio, sr = sf.read(p["audio"], dtype="float32")
        parts = []
        for start, end in cut_points(audio, sr, max_wait):
            piece = audio[start:end]
            if len(piece) < sr * 0.2:
                continue
            seg_s = len(piece) / sr
            t0 = time.perf_counter()
            segs, _ = model.transcribe(piece, beam_size=1)
            parts.append(" ".join(x.text for x in segs))
            lat.append(seg_s * 1000 + (time.perf_counter() - t0) * 1000)
            total += 1
            if seg_s >= max_wait - 0.01:     # hit the cap, no pause found
                forced += 1
        refs.append(p["reference"]); hyps.append(" ".join(parts))

    m = score(refs, hyps)
    print(f"{'hybrid cap=' + str(max_wait) + 's':>24} {m['wer']*100:>7.2f}% "
          f"{np.percentile(lat,50):>7.0f}ms {np.percentile(lat,95):>7.0f}ms "
          f"{max(lat):>7.0f}ms {forced/total*100:>7.0f}%")

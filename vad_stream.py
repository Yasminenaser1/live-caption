import json, time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps
from wer import score

N = 50
SR = 16000
CONFIGS = [200, 400, 800]   # min silence (ms) that counts as a boundary

pairs = json.load(open("data/pairs.json"))[:N]
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

print(f"{'config':>22} {'WER':>8} {'latency p50':>13} {'p95':>9} {'segs/utt':>9}")
print(f"{'full utterance':>22} {'4.31%':>8} {'-':>13} {'-':>9} {'1.0':>9}")
print(f"{'fixed 3s':>22} {'11.46%':>8} {'3104ms':>13} {'-':>9} {'-':>9}")

for min_sil in CONFIGS:
    refs, hyps, lat, nsegs = [], [], [], []
    opts = VadOptions(min_silence_duration_ms=min_sil, speech_pad_ms=100)

    for p in pairs:
        audio, sr = sf.read(p["audio"], dtype="float32")
        stamps = get_speech_timestamps(audio, opts)
        if not stamps:
            stamps = [{"start": 0, "end": len(audio)}]

        parts = []
        for s in stamps:
            piece = audio[s["start"]:s["end"]]
            seg_s = len(piece) / sr
            t0 = time.perf_counter()
            segs, _ = model.transcribe(piece, beam_size=1)
            parts.append(" ".join(x.text for x in segs))
            lat.append(seg_s * 1000 + (time.perf_counter() - t0) * 1000)

        refs.append(p["reference"]); hyps.append(" ".join(parts))
        nsegs.append(len(stamps))

    m = score(refs, hyps)
    print(f"{'VAD sil=' + str(min_sil) + 'ms':>22} {m['wer']*100:>7.2f}% "
          f"{np.percentile(lat,50):>11.0f}ms {np.percentile(lat,95):>7.0f}ms "
          f"{np.mean(nsegs):>8.1f}")

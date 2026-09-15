from faster_whisper import WhisperModel
import time

t0 = time.time()
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print(f"loaded in {time.time()-t0:.1f}s")
print("model ready — no audio yet")

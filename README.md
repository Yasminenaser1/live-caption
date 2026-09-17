# live-caption

Streaming speech-to-text with a measured latency/accuracy tradeoff. The
question isn't whether Whisper works — it's how much accuracy you give up to
show text sooner.

Run locally on CPU with `faster-whisper` (`tiny.en`, int8). No API keys, no
paid services. Scored against LibriSpeech test-clean, so the numbers are
comparable to published ones.

---

## The tradeoff

Streaming means transcribing before the speaker finishes. You choose how much
audio to wait for, and that choice costs accuracy.

| Method | WER | Latency p50 | Latency p95 | Notes |
|---|---|---|---|---|
| Full utterance (not streaming) | 4.31% | — | — | the ceiling |
| **VAD, 200ms silence** | **3.79%** | 4.2s | 11.5s | beats the ceiling; unbounded tail |
| Overlap win=5s hop=3s | 7.98% | 3.1s | — | 1.7× compute |
| Hybrid VAD + 5s cap | 7.78% | 3.3s | 5.1s | 28% of cuts forced |
| Fixed 3s chunks | 11.46% | 3.1s | 3.1s | simplest, predictable |
| Fixed 1s chunks | 35.21% | 1.1s | 1.1s | unusable |

Compute is never the constraint: `tiny.en` runs at **57.7× realtime** on an
M-series CPU. Latency here is almost entirely the wait for audio to accumulate,
not the time to process it. That reframes the whole problem — a faster model
wouldn't help.

---

## Findings

**1. Chunk size dominates everything.** Going from 10s chunks to 1s chunks
takes WER from 4.50% to 35.21% — an 8× increase in error for a 9-second
latency saving. The curve is steep below 3 seconds because words get cut
mid-utterance and the model has no context to recover them.

**2. Cutting at silences beats not streaming at all.** VAD-based chunking
scored 3.79%, *better* than handing the model whole files (4.31%). Trimming
leading and trailing silence appears to remove audio the model was
hallucinating on. This was the surprise of the project.

**3. VAD's cost is tail latency, not accuracy.** p50 of 4.2s hides a p95 of
11.5s — when a speaker doesn't pause, there is nowhere to cut and the caption
stalls. Fixed chunks are worse on average and better at the tail.

**4. Capping the wait mostly turns VAD back into fixed chunking.** A hybrid
that cuts at a VAD boundary or at a hard deadline, whichever comes first,
bounds the worst case exactly as designed (max 3.1s at cap=3s). But **48% of
its cuts were forced** by the deadline rather than found by VAD, and WER
(12.69%) came out slightly *worse* than plain fixed chunks. Good cut points
don't exist on demand.

**5. Overlapping windows made things worse, across three implementations.**
The intuition is that overlap should fix boundary truncation, and more overlap
should be better. Neither held:

| Stitching method | win=3 hop=1.5 (2×) | win=5 hop=3 (1.7×) |
|---|---|---|
| Exact suffix-prefix match | 46.06% | 20.27% |
| Fuzzy alignment scoring | 19.34% | 7.98% |
| Word-timestamp dedup | 40.74% | 18.73% |

In every case, *more* overlap produced *worse* results. The cleanest
comparison holds the window fixed: at win=5s, hop=2.5 (11.46%) is worse than
hop=3.0 (7.98%) — same window, only the number of merges differs. Each merge
is a chance to duplicate or drop words, and that cost exceeds the boundary
errors overlap was supposed to prevent.

The root cause is that Whisper does not transcribe identical audio identically
in different contexts. Text differs in punctuation and word choice; word
timestamps shift by tens of milliseconds and occasionally reorder. Neither
gives a stable key to join on.

---

## Diagnostics worth reading

The timestamp failure had a measurable cause. Dumping per-word absolute times
across overlapping windows showed duplicate words landing ~40ms apart while
genuinely consecutive words average **140ms** apart. The first dedup threshold
was 250ms, which deleted every second real word — visible as a suspiciously
flat 27.94% WER across two different hop sizes. Tightening to 80ms plus text
matching fixed that specific bug and still lost to text stitching.

---

## Limitations

- **LibriSpeech is read audiobook speech**, which has far fewer pauses than
  conversation. VAD would likely perform better on a real meeting, where people
  stop constantly — and the hybrid's 48% forced-cut rate would likely drop. The
  benchmark makes VAD look worse than it probably is in its intended setting.
- **50 utterances per configuration** (100 for the baseline). Small.
- **Only `tiny.en` was tested.** A larger model would be more accurate and
  might handle short chunks better, since more of the error may be context
  starvation rather than truncation. Untested.
- **Latency is modelled, not measured end to end.** It's chunk duration plus
  measured compute, which omits audio capture, buffering, and display. Real
  latency would be higher.
- **No real microphone input.** Everything replays files as if they were
  streaming. A live path would add jitter this doesn't capture.

---

## Recommendation

For captioning where occasional stalls are acceptable: **VAD at 200ms**. Best
accuracy of anything tested, including non-streaming.

Where latency must be bounded: **fixed 3s chunks**. 11.46% WER, perfectly
predictable, and the simplest thing to build. The hybrid adds complexity
without beating it on this corpus.

---

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install faster-whisper jiwer soundfile numpy

# LibriSpeech test-clean (~330MB)
mkdir -p data && cd data
curl -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf test-clean.tar.gz && cd ..

python load_data.py          # → data/pairs.json
```

Reproduce each result:

```bash
python baseline.py           # full-utterance ceiling
python stream.py             # fixed chunk sweep
python vad_stream.py         # VAD boundaries
python hybrid_stream.py      # VAD with a latency cap
python overlap_stream.py     # overlap + text stitching
python timestamp_stream.py   # overlap + timestamp stitching
python debug_ts.py           # word timing dump
```

WER is computed with `jiwer`, normalizing case and punctuation on both sides —
LibriSpeech references have neither, and Whisper adds both.

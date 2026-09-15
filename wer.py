import re
import jiwer

# LibriSpeech references are lowercase, no punctuation, numbers spelled out.
# Whisper adds punctuation and casing, so normalize both sides before scoring.
_norm = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def score(references, hypotheses):
    out = jiwer.process_words(references, hypotheses,
                              reference_transform=_norm, hypothesis_transform=_norm)
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }

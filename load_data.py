import glob, json, os

root = "data/LibriSpeech/test-clean"
pairs = []
for trans in glob.glob(f"{root}/*/*/*.trans.txt"):
    folder = os.path.dirname(trans)
    for line in open(trans):
        uid, text = line.strip().split(" ", 1)
        path = f"{folder}/{uid}.flac"
        if os.path.exists(path):
            pairs.append({"id": uid, "audio": path, "reference": text.lower()})

pairs.sort(key=lambda p: p["id"])
print(f"{len(pairs)} utterances")
json.dump(pairs, open("data/pairs.json", "w"), indent=2)
print(pairs[0]["reference"][:90])

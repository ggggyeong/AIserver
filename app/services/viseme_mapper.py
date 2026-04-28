import json

with open("app/data/phoneme_to_viseme_map.json", "r", encoding="utf-8") as f:
    PHONEME_TO_VISEME = json.load(f)


def map_phoneme_to_viseme(phoneme: str) -> str:
    return PHONEME_TO_VISEME.get(phoneme, "sil")
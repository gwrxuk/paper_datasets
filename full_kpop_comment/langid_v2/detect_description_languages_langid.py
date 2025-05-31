import csv
import os
import langid
from collections import defaultdict
import re

# Ensure output directory exists
os.makedirs('langid', exist_ok=True)

# Helper function to check for traditional Chinese characters
TRADITIONAL_CHINESE_RANGE = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)
TRADITIONAL_ONLY_CHARS = set('萬與專業叢東絲丟兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂乾亂爭於虧雲亞產畝親褻嚲億僅僕從侖倉儀們價眾優會傘偉傳傷倫偽佇佈體佔併來俠係俏保俞倆倉')

def contains_traditional_chinese(text):
    if any(char in TRADITIONAL_ONLY_CHARS for char in text):
        return True
    for char in text:
        code = ord(char)
        for start, end in TRADITIONAL_CHINESE_RANGE:
            if start <= code <= end:
                if char in TRADITIONAL_ONLY_CHARS:
                    return True
    return False

# Read video information
video_data = []
with open('video.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_data.append(row)

description_language_stats = []
all_description_languages = set()

chunk_size = 10
ngram_size = 2

for video in video_data:
    video_id = video['video_id']
    description = video['description']
    chunks = [description[i:i+chunk_size] for i in range(0, len(description), chunk_size)]
    lang_counts = defaultdict(int)
    for chunk in chunks:
        ngram = chunk
        if len(ngram) >= ngram_size:
            lang, prob = langid.classify(ngram)
            if lang == 'zh':
                if contains_traditional_chinese(ngram):
                    lang = 'zh-tw'
                else:
                    lang = 'zh-cn'
            lang_counts[lang] += 1
            all_description_languages.add(lang)
    row = {
        'youtube_id': video_id,
        'description': description,
        'total_languages': len(lang_counts),
    }
    row.update(lang_counts)
    description_language_stats.append(row)

all_description_languages = sorted(list(all_description_languages))

desc_fieldnames = ['youtube_id', 'description', 'total_languages'] + all_description_languages
with open('langid/video_description_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=desc_fieldnames)
    writer.writeheader()
    for stats in description_language_stats:
        row = {lang: stats.get(lang, 0) for lang in all_description_languages}
        row.update({
            'youtube_id': stats['youtube_id'],
            'description': stats['description'],
            'total_languages': stats['total_languages']
        })
        writer.writerow(row)

print(f"Successfully analyzed {len(video_data)} videos using langid (2-gram detection) on descriptions.")
print(f"Found {len(all_description_languages)} different languages in descriptions.")
print("Results saved to langid/video_description_language.csv") 
import csv
import os
import langid
from collections import defaultdict
import re

# Ensure output directory exists
os.makedirs('langid', exist_ok=True)

# Helper function to check for traditional Chinese characters
# Unicode range for traditional Chinese (commonly used characters)
TRADITIONAL_CHINESE_RANGE = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
)

# A small set of common traditional-only characters for quick check
TRADITIONAL_ONLY_CHARS = set('萬與專業叢東絲丟兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂乾亂爭於虧雲亞產畝親褻嚲億僅僕從侖倉儀們價眾優會傘偉傳傷倫偽佇佈體佔併來俠係俏保俞倆倉個們倫偉傳傷倫偽佇佈體佔併來俠係俏保俞倆倉個們倫偉傳傷倫偽佇佈體佔併來俠係俏保俞倆倉')

def contains_traditional_chinese(text):
    # Quick check for traditional-only chars
    if any(char in TRADITIONAL_ONLY_CHARS for char in text):
        return True
    # Unicode range check (not perfect, but helps)
    for char in text:
        code = ord(char)
        for start, end in TRADITIONAL_CHINESE_RANGE:
            if start <= code <= end:
                # Further check for traditional-only
                if char in TRADITIONAL_ONLY_CHARS:
                    return True
    return False

# Read video information
video_data = []
with open('video.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_data.append(row)

title_language_stats = []
all_title_languages = set()

# Parameters
chunk_size = 10  # Number of characters per chunk
ngram_size = 2  # n-gram size within each chunk

for video in video_data:
    video_id = video['video_id']
    title = video['title']
    # Split title into non-overlapping chunks of chunk_size
    chunks = [title[i:i+chunk_size] for i in range(0, len(title), chunk_size)]
    lang_counts = defaultdict(int)
    for chunk in chunks:
        # Generate n-grams within the chunk
        ngram = chunk
        if len(ngram) >= ngram_size:
            lang, prob = langid.classify(ngram)
            print(f"Ngram: {ngram}, Language: {lang}, Probability: {prob}")
            # Special handling for Chinese
            if lang == 'zh':
                if contains_traditional_chinese(ngram):
                    lang = 'zh-tw'
                else:
                    lang = 'zh-cn'
            lang_counts[lang] += 1
            all_title_languages.add(lang)
    # Save stats
    row = {
        'youtube_id': video_id,
        'title': title,
        'total_languages': len(lang_counts),
    }
    row.update(lang_counts)
    title_language_stats.append(row)

# Sort languages alphabetically
all_title_languages = sorted(list(all_title_languages))

# Write title results to CSV
prob_fieldnames = []
title_fieldnames = ['youtube_id', 'title', 'total_languages'] + all_title_languages
with open('langid/video_title_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=title_fieldnames)
    writer.writeheader()
    for stats in title_language_stats:
        row = {lang: stats.get(lang, 0) for lang in all_title_languages}
        row.update({
            'youtube_id': stats['youtube_id'],
            'title': stats['title'],
            'total_languages': stats['total_languages']
        })
        writer.writerow(row)

print(f"Successfully analyzed {len(video_data)} videos using langid (2-gram detection).")
print(f"Found {len(all_title_languages)} different languages in titles.")
print("Results saved to langid/video_title_language.csv") 
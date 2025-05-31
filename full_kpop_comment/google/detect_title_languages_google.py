import csv
import os
import requests
from collections import defaultdict

# Get your Google Cloud Translation API key from environment variable
API_KEY = os.environ.get('GOOGLE_TRANSLATE_API_KEY')
if not API_KEY:
    raise ValueError('Please set the GOOGLE_TRANSLATE_API_KEY environment variable.')

# Ensure output directory exists
os.makedirs('google', exist_ok=True)

# Read video information
video_data = []
with open('video.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        video_data.append(row)

title_language_stats = []
all_title_languages = set()

def detect_language(text):
    url = f'https://translation.googleapis.com/language/translate/v2/detect?key={API_KEY}'
    response = requests.post(url, data={'q': text})
    if response.status_code == 200:
        result = response.json()
        lang = result['data']['detections'][0][0]['language']
        return lang
    else:
        print(f"Error detecting language for: {text}")
        return 'und'

for video in video_data:
    video_id = video['video_id']
    title = video['title']
    lang = detect_language(title)
    all_title_languages.add(lang)
    # Save stats
    title_language_stats.append({
        'youtube_id': video_id,
        'title': title,
        'total_languages': 1,
        lang: 1
    })

# Sort languages alphabetically
all_title_languages = sorted(list(all_title_languages))

# Write title results to CSV
title_fieldnames = ['youtube_id', 'title', 'total_languages'] + all_title_languages
with open('google/video_title_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
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

print(f"Successfully analyzed {len(video_data)} videos using Google Translation API (API key method).")
print(f"Found {len(all_title_languages)} different languages in titles.")
print("Results saved to google/video_title_language.csv") 
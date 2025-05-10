import csv
from langdetect import detect, DetectorFactory
from collections import defaultdict

# Set seed for consistent results
DetectorFactory.seed = 0

def detect_languages_in_text(text):
    """Detect all languages present in a text."""
    # Split text into words and try to detect language for each word
    words = text.split()
    languages = set()
    
    for word in words:
        try:
            # Skip very short words and common symbols
            if len(word) < 2 or word.isdigit():
                continue
            lang = detect(word)
            languages.add(lang)
        except:
            continue
    
    return languages

def analyze_text(text):
    """Analyze text and return language statistics."""
    languages = detect_languages_in_text(text)
    lang_counts = defaultdict(int)
    for lang in languages:
        lang_counts[lang] += 1
    return languages, lang_counts

def main():
    # Read video information
    video_data = []
    with open('video.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_data.append(row)
    
    # Process each video and collect language statistics for both title and description
    all_title_languages = set()
    all_desc_languages = set()
    title_language_stats = []
    desc_language_stats = []
    
    for video in video_data:
        video_id = video['video_id']
        title = video['title']
        description = video['description']
        
        # Analyze title
        title_languages, title_lang_counts = analyze_text(title)
        all_title_languages.update(title_languages)
        
        # Analyze description
        desc_languages, desc_lang_counts = analyze_text(description)
        all_desc_languages.update(desc_languages)
        
        # Add title stats
        title_language_stats.append({
            'youtube_id': video_id,
            'title': title,
            'total_languages': len(title_languages),
            **title_lang_counts
        })
        
        # Add description stats
        desc_language_stats.append({
            'youtube_id': video_id,
            'description': description,
            'total_languages': len(desc_languages),
            **desc_lang_counts
        })
    
    # Sort languages alphabetically
    all_title_languages = sorted(list(all_title_languages))
    all_desc_languages = sorted(list(all_desc_languages))
    
    # Write title results to CSV
    title_fieldnames = ['youtube_id', 'title', 'total_languages'] + all_title_languages
    with open('video_title_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
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
    
    # Write description results to CSV
    desc_fieldnames = ['youtube_id', 'description', 'total_languages'] + all_desc_languages
    with open('video_description_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=desc_fieldnames)
        writer.writeheader()
        
        for stats in desc_language_stats:
            row = {lang: stats.get(lang, 0) for lang in all_desc_languages}
            row.update({
                'youtube_id': stats['youtube_id'],
                'description': stats['description'],
                'total_languages': stats['total_languages']
            })
            writer.writerow(row)
    
    print(f"Successfully analyzed {len(video_data)} videos")
    print(f"Found {len(all_title_languages)} different languages in titles")
    print(f"Found {len(all_desc_languages)} different languages in descriptions")
    print("Results saved to video_title_language.csv and video_description_language.csv")

if __name__ == "__main__":
    main() 
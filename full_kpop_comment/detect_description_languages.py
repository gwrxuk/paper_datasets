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

def main():
    # Read video information
    video_data = []
    with open('video.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_data.append(row)
    
    # Process each video and collect language statistics
    all_languages = set()
    video_language_stats = []
    
    for video in video_data:
        video_id = video['video_id']
        description = video['description']
        
        # Detect languages in description
        languages = detect_languages_in_text(description)
        
        # Update set of all languages found
        all_languages.update(languages)
        
        # Create language count dictionary
        lang_counts = defaultdict(int)
        for lang in languages:
            lang_counts[lang] += 1
        
        # Add video stats
        video_language_stats.append({
            'youtube_id': video_id,
            'description': description,
            'total_languages': len(languages),
            **lang_counts
        })
    
    # Sort languages alphabetically
    all_languages = sorted(list(all_languages))
    
    # Write results to CSV
    fieldnames = ['youtube_id', 'description', 'total_languages'] + all_languages
    
    with open('video_description_language.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for stats in video_language_stats:
            # Ensure all language columns exist in each row
            row = {lang: stats.get(lang, 0) for lang in all_languages}
            row.update({
                'youtube_id': stats['youtube_id'],
                'description': stats['description'],
                'total_languages': stats['total_languages']
            })
            writer.writerow(row)
    
    print(f"Successfully analyzed {len(video_data)} videos")
    print(f"Found {len(all_languages)} different languages in descriptions")
    print("Results saved to video_description_language.csv")

if __name__ == "__main__":
    main() 
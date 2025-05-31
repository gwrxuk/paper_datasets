import csv
import os

# Define all possible language codes
ALL_LANGUAGES = {
    'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es',
    'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hmn', 'hr', 'ht', 'hu', 'hy',
    'id', 'ig', 'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv',
    'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'ro',
    'ru', 'rw', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk',
    'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-cn', 'zh-tw', 'zu'
}

def save_results(results, output_file):
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(output_file)
    
    # Prepare fieldnames
    fieldnames = ['youtube_id', 'comment_count', 'total_languages', 'undetected_count'] + sorted(ALL_LANGUAGES)
    
    # Write results to CSV
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Process each result
        for row in results:
            # Get the language counts for this video
            lang_counts = row['language_counts']
            
            # Count only languages that have at least one comment
            detected_languages = sum(1 for lang, count in lang_counts.items() if count > 0 and lang in ALL_LANGUAGES)
            
            # Create output row
            output_row = {
                'youtube_id': row['youtube_id'],
                'comment_count': row['comment_count'],
                'total_languages': detected_languages,
                'undetected_count': row['undetected_count']
            }
            
            # Add individual language counts
            for lang in sorted(ALL_LANGUAGES):
                output_row[lang] = lang_counts.get(lang, 0)
            
            writer.writerow(output_row)
    
    print(f"Results saved to {output_file}") 
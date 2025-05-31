import csv
import os
from langid.langid import LanguageIdentifier, model
from collections import defaultdict
import re
import codecs
import json
import sys

# Initialize the language identifier
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Ensure output directory exists
os.makedirs('langid', exist_ok=True)

# Define all possible language codes
ALL_LANGUAGES = {
    'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es',
    'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hmn', 'hr', 'ht', 'hu', 'hy',
    'id', 'ig', 'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv',
    'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'ro',
    'ru', 'rw', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk',
    'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zh-cn', 'zh-tw', 'zu'
}

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

def load_processed_comments():
    """Load the set of already processed comments from JSON file"""
    if os.path.exists('langid/processed_comments.json'):
        with open('langid/processed_comments.json', 'r') as f:
            data = json.load(f)
            # Convert lists back to sets
            return {video_id: set(comments) for video_id, comments in data.items()}
    return {}

def save_processed_comments(processed_comments):
    """Save the set of processed comments to JSON file"""
    # Convert sets to lists for JSON serialization
    serializable_comments = {video_id: list(comments) for video_id, comments in processed_comments.items()}
    with open('langid/processed_comments.json', 'w') as f:
        json.dump(serializable_comments, f)

def save_results(results, is_final=False):
    """Save results to CSV file"""
    filename = 'langid/video_comment_language.csv'
    mode = 'w' if is_final else 'a'
    file_exists = os.path.exists(filename) and not is_final
    
    # Create a new results list with modified column names
    modified_results = []
    for row in results:
        new_row = {
            'youtube_id': row['youtube_id'],
            'total_comments': row['comment_count'],
            'total_languages': row['total_languages'],
            'undetected_languages': row['undetected_count']
        }
        # Add language counts without _count suffix
        for lang in sorted(ALL_LANGUAGES):
            new_row[lang] = row[f'{lang}_count']
        modified_results.append(new_row)
    
    with open(filename, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=modified_results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(modified_results)

def clean_text(text):
    """Clean text for language detection"""
    if not text:
        return ""
    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))
    # Remove excessive emojis (keep only one of each type)
    emoji_chars = set()
    cleaned_text = []
    for char in text:
        if ord(char) > 0xFFFF:  # Emoji range
            if char not in emoji_chars:
                emoji_chars.add(char)
                cleaned_text.append(char)
        else:
            cleaned_text.append(char)
    return ''.join(cleaned_text)

def main():
    # Create langid directory if it doesn't exist
    os.makedirs('langid', exist_ok=True)
    
    # Load already processed comments
    processed_comments = load_processed_comments()
    total_processed = sum(len(comments) for comments in processed_comments.values())
    print(f"Found {total_processed} already processed comments")
    
    # Get all comment files
    comment_files = [f for f in os.listdir('comments') if f.startswith('comments_') and f.endswith('.csv')]
    print(f"Found {len(comment_files)} comment files to process")
    
    # Process each file
    for file_idx, filename in enumerate(comment_files, 1):
        video_id = filename.replace('comments_', '').replace('.csv', '')
        print(f"\nProcessing video {file_idx}/{len(comment_files)}: {video_id}")
        
        # Initialize statistics for this video
        lang_counts = defaultdict(int)
        comment_count = 0
        undetected_count = 0  # Counter for undetected languages
        processed_video_comments = set()
        
        try:
            with codecs.open(os.path.join('comment_with_replies', filename), 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Get comment text and clean it
                        comment_text = row.get('text', '')
                        if not comment_text:
                            continue
                            
                        # Create a unique identifier for the comment
                        comment_id = f"{row.get('author', '')}_{comment_text[:50]}"
                        
                        # Skip if already processed
                        if video_id in processed_comments and comment_id in processed_comments[video_id]:
                            continue
                            
                        # Clean the text
                        cleaned_text = clean_text(comment_text)
                        if not cleaned_text:
                            continue
                            
                        # Skip if text is too short after cleaning
                        if len(cleaned_text) < 2:
                            continue
                            
                        # Create bigrams
                        bigrams = [cleaned_text] # [cleaned_text[i:i+2] for i in range(len(cleaned_text)-1)]
                        
                        language_detected = False  # Flag to check if language was detected for this comment
                        
                        # Process each bigram
                        for bigram in bigrams:
                            try:
                                # Get language classification for the bigram
                                lang, confidence = identifier.classify(bigram)
                                
                                # Check if the confidence is too low (could indicate undetected language)
                                if confidence < 0.1:  # Using a threshold of 0.1 for demonstration, adjust as needed
                                    undetected_count += 1
                                    language_detected = False
                                else:
                                    lang_counts[lang] += 1
                                    language_detected = True
                            except Exception as e:
                                print(f"Error processing bigram in {filename}: {str(e)}")
                                undetected_count += 1
                                continue
                        
                        # If no language was detected for this comment, increment the undetected counter
                        if not language_detected:
                            undetected_count += 1
                            
                        comment_count += 1
                        processed_video_comments.add(comment_id)
                        
                    except Exception as e:
                        print(f"Error processing comment in {filename}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            continue
            
        # Save results for this video
        if comment_count > 0:
            row = {
                'youtube_id': video_id,
                'comment_count': comment_count,
                'total_languages': len(lang_counts),
                'undetected_count': undetected_count
            }
            
            # Add all languages, setting count to 0 for languages not found
            for lang in sorted(ALL_LANGUAGES):
                row[f'{lang}_count'] = lang_counts.get(lang, 0)
            
            print(f"Saving results for video {video_id}...")
            save_results([row])
            
            # Update processed comments
            if video_id not in processed_comments:
                processed_comments[video_id] = set()
            processed_comments[video_id].update(processed_video_comments)
            save_processed_comments(processed_comments)
            
            print(f"Processed {comment_count} comments for video {video_id}")
            print(f"Found {len(lang_counts)} different languages")
            print(f"Comments with undetected language: {undetected_count}")
            
            # Save final results
            if file_idx == len(comment_files):
                print("\nSaving final results...")
                save_results([row], is_final=True)
    
    # Print final summary
    print(f"\nSuccessfully analyzed {len(comment_files)} videos' comments using langid (2-gram detection).")
    print(f"Found {len(ALL_LANGUAGES)} different languages in comments.")
    print("Results saved to langid/video_comment_language.csv")
    
    return len(comment_files)

if __name__ == "__main__":
    main()

# If we have a temporary file, merge it with the final results
if os.path.exists('langid/video_comment_language_temp.csv'):
    try:
        with open('langid/video_comment_language_temp.csv', 'r', encoding='utf-8') as temp_file:
            temp_reader = csv.DictReader(temp_file)
            temp_data = list(temp_reader)
        
        with open('langid/video_comment_language.csv', 'a', newline='', encoding='utf-8') as final_file:
            writer = csv.DictWriter(final_file, fieldnames=temp_reader.fieldnames)
            writer.writerows(temp_data)
        
        os.remove('langid/video_comment_language_temp.csv')
    except Exception as e:
        print(f"Error merging temporary results: {e}") 
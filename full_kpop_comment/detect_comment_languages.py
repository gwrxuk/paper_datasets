import csv
import os
import pandas as pd
from langdetect import detect, DetectorFactory
from collections import defaultdict
from langcodes import standardize_tag

# Set seed for consistent results
DetectorFactory.seed = 0

# Common language codes supported by langdetect
LANGUAGE_CODES = [
    'af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he',
    'hi', 'hr', 'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ne', 'nl', 'no', 'pa', 'pl',
    'pt', 'ro', 'ru', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-cn', 'zh-tw'
]

def get_processed_videos(output_file):
    """Get set of video IDs that have already been processed."""
    if not os.path.exists(output_file):
        return set()
    
    try:
        df = pd.read_csv(output_file)
        return set(df['youtube_id'].tolist())
    except:
        return set()

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
            if lang in LANGUAGE_CODES:  # Only count if it's in our predefined list
                languages.add(lang)
        except:
            continue
    
    return languages

def read_comments_safely(comment_file):
    """Read comments file safely with error handling."""
    try:
        # Try reading with pandas first with increased buffer size
        df = pd.read_csv(comment_file, 
                        encoding='utf-8', 
                        on_bad_lines='skip',
                        engine='python',  # Use Python engine instead of C
                        quoting=csv.QUOTE_NONE,  # Don't expect quotes
                        sep=',')
        return df
    except Exception as e:
        print(f"Error reading {comment_file} with pandas: {str(e)}")
        try:
            # Fallback to reading line by line with larger buffer
            comments = []
            with open(comment_file, 'r', encoding='utf-8', errors='replace', buffering=1024*1024) as f:
                header = next(f)  # Skip header
                for line in f:
                    try:
                        # Simple CSV parsing with error handling
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        parts = line.split(',', 3)  # Split into 4 parts max
                        if len(parts) >= 2:  # At least author and text
                            text = parts[1].strip()
                            if text:  # Only add non-empty comments
                                comments.append({'text': text})
                    except Exception as line_error:
                        print(f"Error parsing line in {comment_file}: {str(line_error)}")
                        continue
            if not comments:
                print(f"No valid comments found in {comment_file}")
                return pd.DataFrame()
            return pd.DataFrame(comments)
        except Exception as e:
            print(f"Error reading {comment_file} with fallback method: {str(e)}")
            return pd.DataFrame()

def process_video(video_id, video_data, output_file):
    """Process a single video's comments and save results."""
    comment_file = f'comments/comments_{video_id}.csv'
    if not os.path.exists(comment_file):
        return False
    
    try:
        # Read comments safely
        df = read_comments_safely(comment_file)
        if df.empty:
            print(f"No valid comments found in {comment_file}")
            return False
            
        all_comments = df['text'].dropna().tolist()
        
        # Combine all comments into one text
        combined_text = ' '.join(str(comment) for comment in all_comments)
        
        # Detect languages in comments
        languages = detect_languages_in_text(combined_text)
        
        # Create language count dictionary
        lang_counts = defaultdict(int)
        for lang in languages:
            lang_counts[lang] += 1
        
        # Create stats row
        stats = {
            'youtube_id': video_id,
            'title': video_data[video_id]['title'],
            'comment_count': len(df),
            'total_languages': len(languages)
        }
        
        # Add language counts
        for lang in LANGUAGE_CODES:
            stats[lang] = lang_counts.get(lang, 0)
        
        # Write to CSV
        file_exists = os.path.exists(output_file)
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['youtube_id', 'title', 'comment_count', 'total_languages'] + sorted(LANGUAGE_CODES)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
            
        return True
        
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        return False

def main():
    print(f"Using {len(LANGUAGE_CODES)} language codes")
    
    # Read video information
    video_data = {}
    with open('video.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_data[row['video_id']] = row
    
    # Initialize output file and get processed videos
    output_file = 'video_comment_language.csv'
    processed_videos = get_processed_videos(output_file)
    print(f"Found {len(processed_videos)} already processed videos")
    
    # Get all comment files
    comment_files = [f for f in os.listdir('comments') if f.startswith('comments_') and f.endswith('.csv')]
    total_videos = len(comment_files)
    remaining_videos = total_videos - len(processed_videos)
    
    if remaining_videos == 0:
        print("All videos have been processed!")
        return
    
    print(f"Processing {remaining_videos} remaining videos...")
    processed_count = 0
    
    for comment_file in comment_files:
        video_id = comment_file.replace('comments_', '').replace('.csv', '')
        if video_id in processed_videos or video_id not in video_data:
            continue
            
        # Process video
        success = process_video(video_id, video_data, output_file)
        
        if success:
            processed_count += 1
            
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{remaining_videos} new videos...")
    
    print(f"Successfully analyzed {processed_count} new videos")
    print(f"Total processed videos: {len(processed_videos) + processed_count}")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 
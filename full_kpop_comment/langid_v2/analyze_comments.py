import os
import csv
from langid.langid import LanguageIdentifier, model
from collections import defaultdict
import json
from process_comments import save_results, ALL_LANGUAGES

# Initialize the language identifier
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Ensure output directory exists
os.makedirs('langid_v2', exist_ok=True)

def process_comments():
    # Get list of comment files
    comment_files = [f for f in os.listdir('comments') if f.startswith('comments_') and f.endswith('.csv')]
    
    # Track processed comments to avoid duplicates
    processed_comments = {}
    
    # Process each video's comments
    for filename in comment_files:
        video_id = filename.replace('comments_', '').replace('.csv', '')
        print(f"\nProcessing video {video_id}...")
        
        try:
            # Initialize counters for this video
            comment_count = 0
            undetected_count = 0
            lang_counts = defaultdict(int)
            processed_video_comments = set()
            
            # Read comments
            with open(os.path.join('comments', filename), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    comment_id = row['comment_id']
                    
                    # Skip if already processed
                    if video_id in processed_comments and comment_id in processed_comments[video_id]:
                        continue
                    
                    try:
                        text = row['text']
                        if not text or len(text.strip()) == 0:
                            continue
                            
                        # Split text into chunks for better language detection
                        # Use larger chunks for better accuracy
                        chunk_size = 100
                        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                        
                        # Track language detections for this comment
                        chunk_languages = defaultdict(int)
                        chunk_confidences = defaultdict(float)
                        
                        # Process each chunk
                        for chunk in chunks:
                            try:
                                # Get language classification for the chunk
                                lang, confidence = identifier.classify(chunk)
                                
                                # Only count if confidence is high enough and language is in our set
                                if confidence >= 0.5 and lang in ALL_LANGUAGES:
                                    chunk_languages[lang] += 1
                                    chunk_confidences[lang] += confidence
                            except Exception as e:
                                print(f"Error processing chunk in {filename}: {str(e)}")
                                continue
                        
                        # If we have any language detections, use the most common one
                        if chunk_languages:
                            # Get the language with highest confidence
                            best_lang = max(chunk_languages.items(), key=lambda x: (x[1], chunk_confidences[x[0]]))[0]
                            lang_counts[best_lang] += 1
                        else:
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
                'language_counts': lang_counts,
                'undetected_count': undetected_count
            }
            
            print(f"Saving results for video {video_id}...")
            save_results([row], 'langid_v2/video_comment_language.csv')
            
            # Update processed comments
            if video_id not in processed_comments:
                processed_comments[video_id] = set()
            processed_comments[video_id].update(processed_video_comments)
            
            print(f"Processed {comment_count} comments for video {video_id}")
            print(f"Found {len(lang_counts)} different languages")
            print(f"Comments with undetected language: {undetected_count}")
    
    # Print final summary
    print(f"\nSuccessfully analyzed {len(comment_files)} videos' comments using langid.")
    print(f"Found {len(ALL_LANGUAGES)} different languages in comments.")
    print("Results saved to langid_v2/video_comment_language.csv")

if __name__ == "__main__":
    process_comments() 
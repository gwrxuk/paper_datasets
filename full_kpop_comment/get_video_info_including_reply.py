import os
import csv
import googleapiclient.discovery
from googleapiclient.errors import HttpError
import time
import json

# YouTube Data API setup
API_KEY = "AIzaSyBIznwATjgsaiLjtkQbpXX22AL548aUmng"  # YouTube Data API key
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)

def extract_video_ids():
    """Extract video IDs from comment filenames."""
    video_ids = []
    for filename in os.listdir('comments'):
        if filename.startswith('comments_') and filename.endswith('.csv'):
            video_id = filename.replace('comments_', '').replace('.csv', '')
            video_ids.append(video_id)
    return video_ids

def load_processed_state():
    """Load the state of processed comments and pagination tokens."""
    state_file = 'comment_with_replies/processing_state.json'
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {'processed_videos': [], 'next_page_tokens': {}}

def save_processed_state(state):
    """Save the state of processed comments and pagination tokens."""
    os.makedirs('comment_with_replies', exist_ok=True)
    with open('comment_with_replies/processing_state.json', 'w') as f:
        json.dump(state, f)

def save_failed_video(video_id, error_message):
    """Save failed video information to CSV."""
    failed_file = 'comment_with_replies/failed_comments.csv'
    file_exists = os.path.exists(failed_file)
    
    with open(failed_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video_id', 'error_message', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'video_id': video_id,
            'error_message': error_message,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })

def save_comment_to_csv(comment, output_file, fieldnames):
    """Save a single comment to CSV file."""
    file_exists = os.path.exists(output_file)
    mode = 'a' if file_exists else 'w'
    
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(comment)

def get_comments_with_replies(video_id, next_page_token=None):
    """Fetch comments and their replies using YouTube Data API."""
    comments = []
    try:
        # Get top-level comments
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        )
        
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comment_id = item['snippet']['topLevelComment']['id']
            
            # Add original comment
            comments.append({
                'comment_id': comment_id,
                'parent_id': None,  # No parent for original comments
                'is_reply': False,
                'author': comment['authorDisplayName'],
                'text': comment['textDisplay'],
                'like_count': comment['likeCount'],
                'published_at': comment['publishedAt'],
                'updated_at': comment['updatedAt']
            })
            
            # Get replies if they exist
            if 'replies' in item:
                replies = item['replies']['comments']
                for reply in replies:
                    reply_snippet = reply['snippet']
                    comments.append({
                        'comment_id': reply['id'],
                        'parent_id': comment_id,  # Link to parent comment
                        'is_reply': True,
                        'author': reply_snippet['authorDisplayName'],
                        'text': reply_snippet['textDisplay'],
                        'like_count': reply_snippet['likeCount'],
                        'published_at': reply_snippet['publishedAt'],
                        'updated_at': reply_snippet['updatedAt']
                    })
        
        # Return comments and next page token
        return comments, response.get('nextPageToken')
            
    except HttpError as e:
        error_message = str(e)
        print(f"An HTTP error occurred for video {video_id}: {error_message}")
        # Save to failed videos if it's a 400 error or no comments
        if e.resp.status == 400 or 'commentsDisabled' in error_message:
            save_failed_video(video_id, error_message)
        return [], None

def main():
    # Create output directory if it doesn't exist
    os.makedirs('comment_with_replies', exist_ok=True)
    
    # Load processing state
    state = load_processed_state()
    
    # Get all video IDs
    video_ids = extract_video_ids()
    print(f"Found {len(video_ids)} video IDs")
    
    # Process each video
    for i, video_id in enumerate(video_ids, 1):
        # Skip already processed videos
        if video_id in state['processed_videos']:
            print(f"Skipping already processed video {i}/{len(video_ids)}: {video_id}")
            continue
            
        print(f"Processing video {i}/{len(video_ids)}: {video_id}")
        
        # Get next page token for this video
        next_page_token = state['next_page_tokens'].get(video_id)
        
        # Initialize output file
        output_file = f'comment_with_replies/comments_{video_id}.csv'
        fieldnames = ['comment_id', 'parent_id', 'is_reply', 'author', 
                     'text', 'like_count', 'published_at', 'updated_at']
        
        try:
            total_comments = 0
            while True:
                # Get comments for current page
                comments, next_page_token = get_comments_with_replies(video_id, next_page_token)
                
                # If no comments and no next page, mark as failed
                if not comments and not next_page_token and total_comments == 0:
                    save_failed_video(video_id, "No comments found")
                    break
                
                # Save comments incrementally
                for comment in comments:
                    save_comment_to_csv(comment, output_file, fieldnames)
                
                total_comments += len(comments)
                print(f"Saved {len(comments)} comments for video {video_id} (Total: {total_comments})")
                
                # Update state with next page token
                if next_page_token:
                    state['next_page_tokens'][video_id] = next_page_token
                    save_processed_state(state)
                else:
                    # No more pages, mark video as processed
                    state['processed_videos'].append(video_id)
                    if video_id in state['next_page_tokens']:
                        del state['next_page_tokens'][video_id]
                    save_processed_state(state)
                    break
                
                # Add delay to respect API quota
                time.sleep(0.5)
                
        except Exception as e:
            error_message = str(e)
            print(f"Error processing video {video_id}: {error_message}")
            save_failed_video(video_id, error_message)
            # Save state before exiting
            save_processed_state(state)
            continue
        
        # Add delay between videos to respect API quota
        time.sleep(1)

if __name__ == "__main__":
    main() 
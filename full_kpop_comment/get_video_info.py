import os
import csv
import googleapiclient.discovery
from googleapiclient.errors import HttpError

# YouTube Data API setup
API_KEY = "AIzaSyD8enbIC6BEs-KgNG32zTyXI6v_O82wb5E"  # YouTube Data API key
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)

def extract_video_ids():
    """Extract video IDs from comment filenames."""
    video_ids = []
    for filename in os.listdir('comments'):
        if filename.startswith('comments_') and filename.endswith('.csv'):
            video_id = filename.replace('comments_', '').replace('.csv', '')
            video_ids.append(video_id)
    return video_ids

def get_video_info(video_id):
    """Fetch video information using YouTube Data API."""
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        
        if response['items']:
            video = response['items'][0]
            return {
                'video_id': video_id,
                'title': video['snippet']['title'],
                'description': video['snippet']['description'],
                'published_at': video['snippet']['publishedAt'],
                'channel_title': video['snippet']['channelTitle'],
                'view_count': video['statistics'].get('viewCount', '0'),
                'like_count': video['statistics'].get('likeCount', '0'),
                'comment_count': video['statistics'].get('commentCount', '0')
            }
        return None
    except HttpError as e:
        print(f"An HTTP error occurred for video {video_id}: {e}")
        return None

def main():
    # Get all video IDs
    video_ids = extract_video_ids()
    print(f"Found {len(video_ids)} video IDs")
    
    # Fetch video information
    video_info_list = []
    for i, video_id in enumerate(video_ids, 1):
        print(f"Processing video {i}/{len(video_ids)}: {video_id}")
        video_info = get_video_info(video_id)
        if video_info:
            video_info_list.append(video_info)
    
    # Save to CSV
    if video_info_list:
        fieldnames = ['video_id', 'title', 'description', 'published_at', 
                     'channel_title', 'view_count', 'like_count', 'comment_count']
        
        with open('video.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_info_list)
        
        print(f"Successfully saved {len(video_info_list)} video records to video.csv")
    else:
        print("No video information was retrieved")

if __name__ == "__main__":
    main() 
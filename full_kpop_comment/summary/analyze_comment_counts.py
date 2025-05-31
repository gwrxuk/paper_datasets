import pandas as pd
import os
from datetime import datetime
from collections import Counter

def generate_html_report(df, summary, top_10, bottom_10, zero_comments, video_info, language_stats):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Comment Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #007bff;
            }}
            .stat-card h3 {{
                margin: 0;
                color: #666;
                font-size: 0.9em;
            }}
            .stat-card p {{
                margin: 5px 0 0;
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
            .zero-comments {{
                background-color: #fff3cd;
            }}
            .video-title {{
                max-width: 400px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .coverage {{
                margin: 20px 0;
                padding: 15px;
                background-color: #e9ecef;
                border-radius: 5px;
            }}
            .stats-section {{
                margin: 20px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            .language-stats {{
                margin: 20px 0;
                padding: 15px;
                background-color: #e9ecef;
                border-radius: 5px;
            }}
            .language-table {{
                width: 100%;
                margin-top: 10px;
            }}
            .language-table th, .language-table td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .language-table th {{
                background-color: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YouTube Comment Analysis Report</h1>
            
            <div class="coverage">
                <h3>Video Coverage</h3>
                <p>Total videos in dataset: {summary['total_videos_in_dataset']:,}</p>
                <p>Videos with comments: {summary['videos_with_comments']:,} ({summary['coverage_percentage']:.1f}%)</p>
                <p>Videos without comments: {summary['videos_without_comments']:,} ({100 - summary['coverage_percentage']:.1f}%)</p>
            </div>

            <div class="language-stats">
                <h2>Language Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Unique Languages in Titles</h3>
                        <p>{language_stats['title_languages_count']:,}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Unique Languages in Descriptions</h3>
                        <p>{language_stats['description_languages_count']:,}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Unique Languages in Comments</h3>
                        <p>{language_stats['comment_languages_count']:,}</p>
                    </div>
                </div>

                <h3>Top 10 Languages in Titles</h3>
                <table class="language-table">
                    <tr>
                        <th>Language</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {''.join(f"""
                    <tr>
                        <td>{lang}</td>
                        <td>{count:,}</td>
                        <td>{(count/language_stats['total_videos']*100):.1f}%</td>
                    </tr>
                    """ for lang, count in language_stats['title_languages'].most_common(10))}
                </table>

                <h3>Top 10 Languages in Descriptions</h3>
                <table class="language-table">
                    <tr>
                        <th>Language</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {''.join(f"""
                    <tr>
                        <td>{lang}</td>
                        <td>{count:,}</td>
                        <td>{(count/language_stats['total_videos']*100):.1f}%</td>
                    </tr>
                    """ for lang, count in language_stats['description_languages'].most_common(10))}
                </table>

                <h3>Top 10 Languages in Comments</h3>
                <table class="language-table">
                    <tr>
                        <th>Language</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {''.join(f"""
                    <tr>
                        <td>{lang}</td>
                        <td>{count:,}</td>
                        <td>{(count/language_stats['total_comments']*100):.1f}%</td>
                    </tr>
                    """ for lang, count in language_stats['comment_languages'].most_common(10))}
                </table>
            </div>

            <div class="stats-section">
                <h2>Comment Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Comments</h3>
                        <p>{summary['total_comments']:,}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Average Comments per Video</h3>
                        <p>{summary['avg_comments']:,.2f}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Median Comments per Video</h3>
                        <p>{summary['median_comments']:,.2f}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Max Comments per Video</h3>
                        <p>{summary['max_comments']:,}</p>
                    </div>
                </div>
            </div>

            <div class="stats-section">
                <h2>Like Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>Total Likes</h3>
                        <p>{summary['total_likes']:,}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Average Likes per Comment</h3>
                        <p>{summary['avg_likes_per_comment']:,.2f}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Median Likes per Comment</h3>
                        <p>{summary['median_likes_per_comment']:,.2f}</p>
                    </div>
                    <div class="stat-card">
                        <h3>Max Likes per Comment</h3>
                        <p>{summary['max_likes_per_comment']:,}</p>
                    </div>
                </div>
            </div>

            <h2>Top 10 Most Commented Videos</h2>
            <table>
                <tr>
                    <th>Video ID</th>
                    <th>Title</th>
                    <th>Comment Count</th>
                    <th>Total Likes</th>
                    <th>Avg Likes/Comment</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{row['video_id']}</td>
                    <td class="video-title" title="{video_info.get(row['video_id'], {}).get('title', 'N/A')}">{video_info.get(row['video_id'], {}).get('title', 'N/A')}</td>
                    <td>{row['comment_count']:,}</td>
                    <td>{row['total_likes']:,}</td>
                    <td>{row['avg_likes_per_comment']:,.2f}</td>
                </tr>
                """ for _, row in top_10.iterrows())}
            </table>

            <h2>Bottom 10 Least Commented Videos</h2>
            <table>
                <tr>
                    <th>Video ID</th>
                    <th>Title</th>
                    <th>Comment Count</th>
                    <th>Total Likes</th>
                    <th>Avg Likes/Comment</th>
                </tr>
                {''.join(f"""
                <tr>
                    <td>{row['video_id']}</td>
                    <td class="video-title" title="{video_info.get(row['video_id'], {}).get('title', 'N/A')}">{video_info.get(row['video_id'], {}).get('title', 'N/A')}</td>
                    <td>{row['comment_count']:,}</td>
                    <td>{row['total_likes']:,}</td>
                    <td>{row['avg_likes_per_comment']:,.2f}</td>
                </tr>
                """ for _, row in bottom_10.iterrows())}
            </table>

            <h2>Videos with No Comments</h2>
            <table>
                <tr>
                    <th>Video ID</th>
                    <th>Title</th>
                </tr>
                {''.join(f"""
                <tr class="zero-comments">
                    <td>{video_id}</td>
                    <td class="video-title" title="{video_info.get(video_id, {}).get('title', 'N/A')}">{video_info.get(video_id, {}).get('title', 'N/A')}</td>
                </tr>
                """ for video_id in zero_comments)}
            </table>

            <div class="timestamp">
                Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def analyze_comment_counts():
    # Get all comment files
    comment_dir = 'comment_with_replies'
    if not os.path.exists(comment_dir):
        print(f"Error: Could not find {comment_dir} directory")
        return

    # Read video information
    video_info_file = 'video_title_language.csv'
    if not os.path.exists(video_info_file):
        print(f"Error: Could not find {video_info_file}")
        return

    try:
        video_info_df = pd.read_csv(video_info_file)
        total_videos_in_dataset = len(video_info_df)
        video_info = {}
        
        # Initialize language counters
        title_languages = Counter()
        description_languages = Counter()
        
        for _, row in video_info_df.iterrows():
            video_info[row['youtube_id']] = {
                'title': row['title'],
                'language': row.get('language', 'N/A')
            }
            # Count languages in titles and descriptions
            title_languages[row.get('title_language', 'unknown')] += 1
            description_languages[row.get('description_language', 'unknown')] += 1
            
    except Exception as e:
        print(f"Error reading video information: {str(e)}")
        return

    # Initialize list to store video comment counts and likes
    video_stats = []
    zero_comment_videos = []
    videos_with_comments = set()
    total_likes = 0
    total_comments = 0
    comment_languages = Counter()

    # Process each comment file
    for filename in os.listdir(comment_dir):
        if filename.startswith('comments_') and filename.endswith('.csv'):
            video_id = filename.replace('comments_', '').replace('.csv', '')
            file_path = os.path.join(comment_dir, filename)
            
            try:
                # Read comment data
                df = pd.read_csv(file_path)
                comment_count = len(df)
                total_likes_for_video = df['like_count'].sum()
                
                # Count comment languages
                if 'language' in df.columns:
                    comment_languages.update(df['language'].fillna('unknown'))
                
                if comment_count == 0:
                    zero_comment_videos.append(video_id)
                else:
                    videos_with_comments.add(video_id)
                    total_likes += total_likes_for_video
                    total_comments += comment_count
                
                video_stats.append({
                    'video_id': video_id,
                    'comment_count': comment_count,
                    'total_likes': total_likes_for_video,
                    'avg_likes_per_comment': total_likes_for_video / comment_count if comment_count > 0 else 0
                })
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Convert to DataFrame and sort
    df = pd.DataFrame(video_stats)
    df_sorted = df.sort_values('comment_count', ascending=False)
    
    # Get top and bottom 10 videos
    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)
    
    # Calculate coverage statistics
    videos_with_comments_count = len(videos_with_comments)
    videos_without_comments = total_videos_in_dataset - videos_with_comments_count
    coverage_percentage = (videos_with_comments_count / total_videos_in_dataset) * 100
    
    # Create language statistics
    language_stats = {
        'title_languages': title_languages,
        'description_languages': description_languages,
        'comment_languages': comment_languages,
        'title_languages_count': len(title_languages),
        'description_languages_count': len(description_languages),
        'comment_languages_count': len(comment_languages),
        'total_videos': total_videos_in_dataset,
        'total_comments': total_comments
    }
    
    # Create summary statistics
    summary = {
        'total_videos_in_dataset': total_videos_in_dataset,
        'videos_with_comments': videos_with_comments_count,
        'videos_without_comments': videos_without_comments,
        'coverage_percentage': coverage_percentage,
        'total_videos': len(df),
        'total_comments': total_comments,
        'avg_comments': df['comment_count'].mean(),
        'median_comments': df['comment_count'].median(),
        'max_comments': df['comment_count'].max(),
        'min_comments': df['comment_count'].min(),
        'zero_comment_videos': len(zero_comment_videos),
        'total_likes': total_likes,
        'avg_likes_per_comment': total_likes / total_comments if total_comments > 0 else 0,
        'median_likes_per_comment': df['avg_likes_per_comment'].median(),
        'max_likes_per_comment': df['avg_likes_per_comment'].max()
    }
    
    # Save results
    output_dir = 'summary'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save HTML report
    html_content = generate_html_report(df, summary, top_10, bottom_10, zero_comment_videos, video_info, language_stats)
    with open(f'{output_dir}/comment_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Also save the raw data with titles
    top_10_with_titles = top_10.copy()
    top_10_with_titles['title'] = top_10_with_titles['video_id'].map(lambda x: video_info.get(x, {}).get('title', 'N/A'))
    top_10_with_titles.to_csv(f'{output_dir}/top_10_commented_videos.csv', index=False)
    
    bottom_10_with_titles = bottom_10.copy()
    bottom_10_with_titles['title'] = bottom_10_with_titles['video_id'].map(lambda x: video_info.get(x, {}).get('title', 'N/A'))
    bottom_10_with_titles.to_csv(f'{output_dir}/bottom_10_commented_videos.csv', index=False)
    
    zero_comments_with_titles = pd.DataFrame({
        'video_id': zero_comment_videos,
        'title': [video_info.get(vid, {}).get('title', 'N/A') for vid in zero_comment_videos]
    })
    zero_comments_with_titles.to_csv(f'{output_dir}/zero_comment_videos.csv', index=False)
    
    # Save language statistics
    pd.DataFrame({
        'language': [lang for lang, _ in title_languages.most_common()],
        'count': [count for _, count in title_languages.most_common()],
        'percentage': [count/total_videos_in_dataset*100 for _, count in title_languages.most_common()]
    }).to_csv(f'{output_dir}/title_languages.csv', index=False)
    
    pd.DataFrame({
        'language': [lang for lang, _ in description_languages.most_common()],
        'count': [count for _, count in description_languages.most_common()],
        'percentage': [count/total_videos_in_dataset*100 for _, count in description_languages.most_common()]
    }).to_csv(f'{output_dir}/description_languages.csv', index=False)
    
    pd.DataFrame({
        'language': [lang for lang, _ in comment_languages.most_common()],
        'count': [count for _, count in comment_languages.most_common()],
        'percentage': [count/total_comments*100 for _, count in comment_languages.most_common()]
    }).to_csv(f'{output_dir}/comment_languages.csv', index=False)
    
    print(f"\nAnalysis complete! Report has been saved to {output_dir}/comment_analysis_report.html")
    print("You can open the HTML file in your web browser to view the results.")

if __name__ == "__main__":
    analyze_comment_counts() 
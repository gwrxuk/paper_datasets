import csv
import os
from collections import Counter
from datetime import datetime

def read_csv_to_dict(file_path):
    """Read CSV file into a list of dictionaries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def generate_html_report(summary, language_stats, video_language_counts, top_video_metadata, top_comment_videos_languages):
    """Generate HTML report with analysis results"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Comment Analysis Report v2</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .stat-box {{ 
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 15px;
                min-width: 200px;
            }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #0d6efd; }}
            .stat-label {{ color: #6c757d; }}
        </style>
    </head>
    <body>
        <h1>YouTube Comment Analysis Report v2</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Comment Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{summary['total_videos']:,}</div>
                    <div class="stat-label">Total Videos</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{summary['total_comments']:,}</div>
                    <div class="stat-label">Total Comments</div>
                </div>
            </div>
            
            <h3>Comment Count Statistics per Video</h3>
            <table class="stat-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Comments</td>
                    <td>{summary['avg_comments']:.1f}</td>
                </tr>
                <tr>
                    <td>Median Comments</td>
                    <td>{summary['median_comments']:.1f}</td>
                </tr>
                <tr>
                    <td>Maximum Comments</td>
                    <td>{summary['max_comments']:,}</td>
                </tr>
                <tr>
                    <td>Minimum Comments</td>
                    <td>{summary['min_comments']:,}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{summary['std_comments']:.1f}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Like Count Statistics</h2>
            <table class="stat-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Average Likes</td>
                    <td>{summary['avg_likes']:.1f}</td>
                </tr>
                <tr>
                    <td>Median Likes</td>
                    <td>{summary['median_likes']:.1f}</td>
                </tr>
                <tr>
                    <td>Maximum Likes</td>
                    <td>{summary['max_likes']:,}</td>
                </tr>
                <tr>
                    <td>Minimum Likes</td>
                    <td>{summary['min_likes']:,}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{summary['std_likes']:.1f}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Language Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{language_stats['comment']['unique_languages']}</div>
                    <div class="stat-label">Global Unique Comment Languages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{language_stats['comment']['max_per_video_languages']}</div>
                    <div class="stat-label">Maximum Languages in a Single Video</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Top 10 Languages in Video Titles</h2>
            <table class="stat-table">
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {''.join(f'<tr><td>{lang}</td><td>{count}</td><td>{(count/summary["total_videos"]*100):.1f}%</td></tr>' for lang, count in language_stats['title']['top_10'])}
            </table>
        </div>

        <div class="section">
            <h2>Top 10 Languages in Video Descriptions</h2>
            <table class="stat-table">
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {''.join(f'<tr><td>{lang}</td><td>{count}</td><td>{(count/summary["total_videos"]*100):.1f}%</td></tr>' for lang, count in language_stats['description']['top_10'])}
            </table>
        </div>

        <div class="section">
            <h2>Top 10 Languages in Comments</h2>
            <table class="stat-table">
                <tr>
                    <th>Language</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {''.join(f'<tr><td>{lang}</td><td>{count}</td><td>{(count/summary["total_comments"]*100):.1f}%</td></tr>' for lang, count in language_stats['comment']['top_10'])}
            </table>
        </div>

        <div class="section">
            <h2>Top 10 Videos with the Most Languages Used in Comments</h2>
            <table class="stat-table">
                <tr>
                    <th>Video ID</th>
                    <th>Languages</th>
                </tr>
                {''.join(f'<tr><td>{video_id}</td><td>{lang_count}</td></tr>' for video_id, lang_count in sorted(video_language_counts.items(), key=lambda x: x[1], reverse=True)[:10])}
            </table>
        </div>

        <div class="section">
            <h2>Metadata for Top 10 Videos with Most Languages in Comments</h2>
            <table class="stat-table">
                <tr>
                    <th>Video ID</th>
                    <th>Title</th>
                    <th>Like Count</th>
                    <th>View Count</th>
                    <th>Comment Count</th>
                </tr>
                {''.join(f'<tr><td>{meta["video_id"]}</td><td>{meta["title"]}</td><td>{meta["like_count"]}</td><td>{meta["view_count"]}</td><td>{meta["comment_count"]}</td></tr>' for meta in top_video_metadata)}
            </table>
        </div>

        <div class="section">
            <h2>Languages Used in Top 10 Videos with Most Comments</h2>
            <table class="stat-table">
                <tr>
                    <th>Video ID</th>
                    <th>Languages</th>
                </tr>
                {''.join(f'<tr><td>{video_id}</td><td>{", ".join(languages)}</td></tr>' for video_id, languages in top_comment_videos_languages)}
            </table>
        </div>

        <div class="section">
            <h2>Top 10 Languages Used in Titles, Comments, and Descriptions</h2>
            <table class="stat-table">
                <tr>
                    <th>Category</th>
                    <th>Languages</th>
                </tr>
                <tr>
                    <td>Titles</td>
                    <td>{", ".join(lang for lang, _ in language_stats['title']['top_10'])}</td>
                </tr>
                <tr>
                    <td>Comments</td>
                    <td>{", ".join(lang for lang, _ in language_stats['comment']['top_10'])}</td>
                </tr>
                <tr>
                    <td>Descriptions</td>
                    <td>{", ".join(lang for lang, _ in language_stats['description']['top_10'])}</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """
    return html_content

def analyze_comment_counts():
    """Analyze comment counts and generate report"""
    # Read video information
    video_info, _ = read_csv_to_dict('langid_v2/video_title_language.csv')
    
    # Read video metadata for like and view counts
    video_metadata, _ = read_csv_to_dict('video.csv')
    video_metadata_dict = {row['video_id']: row for row in video_metadata}
    
    # Read comment language data and get all possible language columns
    comment_data, comment_fieldnames = read_csv_to_dict('langid_v2/video_comment_language.csv')
    # Identify language columns (exclude metadata columns)
    meta_cols = {'youtube_id', 'total_comments', 'total_languages', 'undetected_languages'}
    language_columns = [col for col in comment_fieldnames if col not in meta_cols]
    
    # Initialize counters for title and description languages
    title_languages = Counter()
    description_languages = Counter()
    
    # Count languages in titles and descriptions
    for row in video_info:
        # Count title languages
        for key, value in row.items():
            if key not in ['youtube_id', 'title', 'total_languages'] and value.isdigit() and int(value) > 0:
                title_languages[key] += 1
        
        # Count description languages
        desc_file = 'langid_v2/video_description_language.csv'
        if os.path.exists(desc_file):
            desc_info, _ = read_csv_to_dict(desc_file)
            desc_row = next((r for r in desc_info if r['youtube_id'] == row['youtube_id']), None)
            if desc_row:
                for key, value in desc_row.items():
                    if key not in ['youtube_id', 'description', 'total_languages'] and value.isdigit() and int(value) > 0:
                        description_languages[key] += 1
    
    # Initialize counter for comment languages (total comments per language)
    comment_languages = Counter()
    total_undetected = 0
    total_comments = 0
    per_video_language_counts = []
    video_language_counts = {}  # Dictionary to store language counts per video
    
    # Count languages in comments
    for row in comment_data:
        total_comments += int(row['total_comments'])
        total_undetected += int(row['undetected_languages'])
        # Count number of languages with at least one comment for this video
        lang_count = 0
        for lang in language_columns:
            count = int(row.get(lang, 0))
            if count > 0:
                comment_languages[lang] += count
                lang_count += 1
        per_video_language_counts.append(lang_count)
        video_language_counts[row['youtube_id']] = lang_count
    
    # Global unique languages: languages with at least one comment in any video
    unique_languages = [lang for lang in language_columns if comment_languages[lang] > 0]
    max_per_video_languages = max(per_video_language_counts) if per_video_language_counts else 0
    
    # Print verification
    print(f"Global unique comment languages: {len(unique_languages)}")
    print(f"Maximum languages in a single video's comments: {max_per_video_languages}")
    
    # Create language statistics
    language_stats = {
        'title': {
            'unique_languages': len(title_languages),
            'avg_languages_per_video': sum(title_languages.values()) / len(video_info),
            'top_10': title_languages.most_common(10)
        },
        'description': {
            'unique_languages': len(description_languages),
            'avg_languages_per_video': sum(description_languages.values()) / len(video_info),
            'top_10': description_languages.most_common(10)
        },
        'comment': {
            'unique_languages': len(unique_languages),
            'max_per_video_languages': max_per_video_languages,
            'avg_languages_per_video': sum(per_video_language_counts) / len(per_video_language_counts) if per_video_language_counts else 0,
            'undetected_count': total_undetected,
            'top_10': comment_languages.most_common(10)
        }
    }
    
    # Calculate summary statistics
    comment_counts = [int(row['total_comments']) for row in comment_data]
    like_counts = [int(video_metadata_dict.get(row['youtube_id'], {}).get('like_count', 0)) for row in video_info]
    view_counts = [int(video_metadata_dict.get(row['youtube_id'], {}).get('view_count', 0)) for row in video_info]
    
    summary = {
        'total_videos': len(video_info),
        'total_comments': total_comments,
        'avg_comments': sum(comment_counts) / len(comment_counts) if comment_counts else 0,
        'median_comments': sorted(comment_counts)[len(comment_counts)//2] if comment_counts else 0,
        'max_comments': max(comment_counts) if comment_counts else 0,
        'min_comments': min(comment_counts) if comment_counts else 0,
        'std_comments': (sum((x - (sum(comment_counts)/len(comment_counts)))**2 for x in comment_counts)/len(comment_counts))**0.5 if comment_counts else 0,
        'videos_with_comments': len(comment_data),
        'videos_without_comments': len(video_info) - len(comment_data),
        'coverage_percentage': (len(comment_data) / len(video_info)) * 100 if video_info else 0,
        'total_likes': sum(like_counts),
        'avg_likes': sum(like_counts) / len(like_counts) if like_counts else 0,
        'median_likes': sorted(like_counts)[len(like_counts)//2] if like_counts else 0,
        'max_likes': max(like_counts) if like_counts else 0,
        'min_likes': min(like_counts) if like_counts else 0,
        'std_likes': (sum((x - (sum(like_counts)/len(like_counts)))**2 for x in like_counts)/len(like_counts))**0.5 if like_counts else 0,
        'total_views': sum(view_counts),
        'avg_views': sum(view_counts) / len(view_counts) if view_counts else 0,
        'median_views': sorted(view_counts)[len(view_counts)//2] if view_counts else 0,
        'max_views': max(view_counts) if view_counts else 0,
        'min_views': min(view_counts) if view_counts else 0,
        'std_views': (sum((x - (sum(view_counts)/len(view_counts)))**2 for x in view_counts)/len(view_counts))**0.5 if view_counts else 0
    }
    
    # Print results
    print(f"Total videos analyzed: {summary['total_videos']}")
    print(f"Total comments: {summary['total_comments']}")
    print(f"Average comments per video: {summary['avg_comments']:.1f}")
    print(f"Median comments per video: {summary['median_comments']:.1f}")
    print(f"Maximum comments: {summary['max_comments']}")
    print(f"Minimum comments: {summary['min_comments']}")
    print(f"Standard deviation: {summary['std_comments']:.1f}")
    print(f"\nVideos with comments: {summary['videos_with_comments']}")
    print(f"Videos without comments: {summary['videos_without_comments']}")
    print(f"Coverage percentage: {summary['coverage_percentage']:.1f}%")
    
    print("\nTop 10 languages by comment count:")
    for lang, count in language_stats['comment']['top_10']:
        percentage = (count / total_comments) * 100
        print(f"{lang}: {count} comments ({percentage:.1f}% of total)")
    
    print("\nTop 10 languages in video titles:")
    for lang, count in language_stats['title']['top_10']:
        percentage = (count / len(video_info)) * 100
        print(f"{lang}: {count} videos ({percentage:.1f}% of total)")
    
    print("\nTop 10 languages in video descriptions:")
    for lang, count in language_stats['description']['top_10']:
        percentage = (count / len(video_info)) * 100
        print(f"{lang}: {count} videos ({percentage:.1f}% of total)")
    
    # Print top 10 videos with the most languages used in comments
    print("\nTop 10 videos with the most languages used in comments:")
    top_videos = sorted(video_language_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for video_id, lang_count in top_videos:
        print(f"Video ID: {video_id}, Languages: {lang_count}")

    # Gather metadata for top 10 videos
    top_video_metadata = []
    for video_id, lang_count in top_videos:
        meta = video_metadata_dict.get(video_id, {})
        title = meta.get('title', '')
        like_count = meta.get('like_count', '')
        view_count = meta.get('view_count', '')
        comment_count = meta.get('comment_count', '')
        top_video_metadata.append({
            'video_id': video_id,
            'title': title,
            'like_count': like_count,
            'view_count': view_count,
            'comment_count': comment_count
        })

    # Gather languages used in top 10 videos with most comments
    top_comment_videos = sorted(comment_data, key=lambda x: int(x['total_comments']), reverse=True)[:10]
    top_comment_videos_languages = []
    for video in top_comment_videos:
        video_id = video['youtube_id']
        languages = [lang for lang in language_columns if int(video.get(lang, 0)) > 0]
        top_comment_videos_languages.append((video_id, languages))

    # Generate and save HTML report
    html_content = generate_html_report(summary, language_stats, video_language_counts, top_video_metadata, top_comment_videos_languages)
    with open('summary/comment_analysis_report_v2.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Analysis complete. Report saved to summary/comment_analysis_report_v2.html")

if __name__ == "__main__":
    analyze_comment_counts() 
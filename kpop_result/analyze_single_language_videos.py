import pandas as pd
import numpy as np

# Read the CSV files
comment_lang_df = pd.read_csv('video_comment_language.csv')
video_df = pd.read_csv('video.csv')

def analyze_single_language_videos():
    # Merge video metadata with language data
    merged_df = pd.merge(comment_lang_df, video_df[['youtube_id', 'title', 'view_count', 'like_count']], on='youtube_id')
    
    # Get language columns (excluding metadata columns)
    exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages', 'view_count', 'like_count']
    lang_cols = [col for col in merged_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(merged_df[col])]
    
    # Find videos with only one language
    single_lang_videos = merged_df[merged_df['total_languages'] == 1].copy()
    
    # For each video, identify which language it uses
    single_lang_videos['language'] = single_lang_videos.apply(
        lambda row: next((lang for lang in lang_cols if row[lang] > 0), 'unknown'),
        axis=1
    )
    
    # Sort by view count to see most popular single-language videos
    single_lang_videos = single_lang_videos.sort_values('view_count', ascending=False)
    
    # Count videos per language
    language_counts = single_lang_videos['language'].value_counts()
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K-pop Videos with Single Language Comments</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            h1, h2 {{
                color: #2c3e50;
                text-align: center;
            }}
            .section {{
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .table-container {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px 0;
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
                background-color: #2c3e50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #e9ecef;
            }}
            .video-title {{
                max-width: 300px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 10px;
            }}
            .stat-value {{
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .summary {{
                background-color: #fff3cd;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #ffc107;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>K-pop Videos with Single Language Comments</h1>
            
            <div class="summary">
                <p>Analysis of K-pop videos where all comments are in a single language. 
                This shows the most linguistically homogeneous video discussions in the dataset.</p>
            </div>

            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{len(single_lang_videos)}</div>
                        <div>Total Single-Language Videos</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(language_counts)}</div>
                        <div>Unique Languages</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(single_lang_videos) / len(merged_df) * 100:.1f}%</div>
                        <div>Percentage of Total Videos</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Language Distribution</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Language</th>
                                <th>Number of Videos</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add language distribution rows
    for lang, count in language_counts.items():
        percentage = (count / len(single_lang_videos)) * 100
        html_content += f"""
                            <tr>
                                <td>{lang}</td>
                                <td>{count:,}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>"""
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="section">
                <h2>Top Single-Language Videos by Views</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Video Title</th>
                                <th>Language</th>
                                <th>Views</th>
                                <th>Likes</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add top videos by views
    for rank, (_, row) in enumerate(single_lang_videos.head(10).iterrows(), 1):
        html_content += f"""
                            <tr>
                                <td>{rank}</td>
                                <td class="video-title" title="{row['title']}">{row['title']}</td>
                                <td>{row['language']}</td>
                                <td>{row['view_count']:,}</td>
                                <td>{row['like_count']:,}</td>
                            </tr>"""
    
    html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open('single_language_videos.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_single_language_videos() 
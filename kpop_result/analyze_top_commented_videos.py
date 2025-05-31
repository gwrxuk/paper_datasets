import pandas as pd
import numpy as np

# Read the CSV files
comment_lang_df = pd.read_csv('video_comment_language.csv')
video_df = pd.read_csv('video.csv')

def analyze_top_commented_videos():
    # Merge video metadata with language data
    merged_df = pd.merge(comment_lang_df, video_df[['youtube_id', 'title', 'view_count', 'like_count']], on='youtube_id')
    
    # Sort by total comments and get top 10
    top_videos = merged_df.sort_values('total_comments', ascending=False).head(10)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top 10 Most Commented K-pop Videos</title>
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
                max-width: 400px;
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
            .highlight {{
                color: #e74c3c;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Top 10 Most Commented K-pop Videos</h1>
            
            <div class="summary">
                <p>Analysis of the most engaging K-pop videos based on comment count, showing the linguistic diversity 
                and viewer engagement across different videos.</p>
            </div>

            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{top_videos['total_comments'].mean():,.0f}</div>
                        <div>Average Comments per Video</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{top_videos['total_languages'].mean():.1f}</div>
                        <div>Average Languages per Video</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{top_videos['view_count'].mean():,.0f}</div>
                        <div>Average Views per Video</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Top 10 Most Commented Videos</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Video Title</th>
                                <th>Comments</th>
                                <th>Languages</th>
                                <th>Views</th>
                                <th>Likes</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows for top 10 videos
    for rank, (_, row) in enumerate(top_videos.iterrows(), 1):
        html_content += f"""
                            <tr>
                                <td>{rank}</td>
                                <td class="video-title" title="{row['title']}">{row['title']}</td>
                                <td>{row['total_comments']:,}</td>
                                <td>{row['total_languages']}</td>
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
    with open('top_commented_videos.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_top_commented_videos() 
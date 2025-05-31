import pandas as pd
import numpy as np

# Read the CSV files
comment_lang_df = pd.read_csv('video_comment_language.csv')
video_df = pd.read_csv('video.csv')

def analyze_least_diverse_videos():
    # Merge video metadata with language data
    merged_df = pd.merge(comment_lang_df, video_df[['youtube_id', 'title', 'view_count', 'like_count']], on='youtube_id')
    
    # Sort by total_languages and get bottom 10
    least_diverse = merged_df.sort_values('total_languages').head(10)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Least Linguistically Diverse K-pop Videos</title>
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
            .highlight {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .summary {{
                background-color: #fff3cd;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #ffc107;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Least Linguistically Diverse K-pop Videos</h1>
            
            <div class="summary">
                <p>Analysis of videos with the least language diversity in their comments, 
                showing the minimum number of languages detected in K-pop video discussions.</p>
            </div>

            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{least_diverse['total_languages'].min()}</div>
                    <div>Minimum Languages</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{least_diverse['total_languages'].mean():.1f}</div>
                    <div>Average Languages (Bottom 10)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{least_diverse['total_languages'].max()}</div>
                    <div>Maximum Languages (Bottom 10)</div>
                </div>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Video Title</th>
                            <th>Languages</th>
                            <th>Views</th>
                            <th>Likes</th>
                            <th>Languages Used</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add table rows for least diverse videos
    for rank, (_, row) in enumerate(least_diverse.iterrows(), 1):
        # Get languages used in this video
        lang_cols = [col for col in row.index if col not in ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages', 'view_count', 'like_count']]
        languages_used = [lang for lang in lang_cols if row[lang] > 0]
        
        html_content += f"""
                        <tr>
                            <td>{rank}</td>
                            <td class="video-title" title="{row['title']}">{row['title']}</td>
                            <td>{row['total_languages']}</td>
                            <td>{row['view_count']:,}</td>
                            <td>{row['like_count']:,}</td>
                            <td>{', '.join(languages_used)}</td>
                        </tr>"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open('least_diverse_videos.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_least_diverse_videos() 
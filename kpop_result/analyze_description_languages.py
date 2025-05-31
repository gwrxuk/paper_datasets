import pandas as pd
import numpy as np

# Read the CSV files
desc_lang_df = pd.read_csv('video_description_language.csv')
video_df = pd.read_csv('video.csv')

def analyze_description_languages():
    # Get language columns (excluding metadata columns)
    exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages']
    lang_cols = [col for col in desc_lang_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(desc_lang_df[col])]
    
    # Calculate total videos
    total_videos = len(desc_lang_df)
    
    # Calculate language statistics
    lang_stats = {}
    for col in lang_cols:
        videos_with_lang = (desc_lang_df[col] > 0).sum()
        percentage = (videos_with_lang / total_videos) * 100
        lang_stats[col] = {
            'videos': videos_with_lang,
            'percentage': percentage
        }
    
    # Get top 10 languages
    top_languages = sorted(lang_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)[:10]
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top 10 Languages in K-pop Video Descriptions</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            .container {{
                max-width: 800px;
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
            <h1>Top 10 Languages in K-pop Video Descriptions</h1>
            
            <div class="summary">
                <p>Analysis of {total_videos:,} K-pop videos shows the most prevalent languages used in video descriptions, 
                demonstrating the linguistic diversity in K-pop content presentation.</p>
            </div>

            <div class="section">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{desc_lang_df['total_languages'].mean():.1f}</div>
                        <div>Average Languages per Description</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{desc_lang_df['total_languages'].median()}</div>
                        <div>Median Languages per Description</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{desc_lang_df['total_languages'].max()}</div>
                        <div>Maximum Languages in a Description</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Top 10 Languages in Video Descriptions</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Language</th>
                                <th>Number of Videos</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
    """
    
    # Add table rows for top 10 languages
    for rank, (lang, stats) in enumerate(top_languages, 1):
        highlight_class = 'highlight' if stats['percentage'] > 50 else ''  # Highlight languages used in >50% of videos
        html_content += f"""
                            <tr>
                                <td>{rank}</td>
                                <td class="{highlight_class}">{lang}</td>
                                <td>{stats['videos']:,}</td>
                                <td>{stats['percentage']:.1f}%</td>
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
    with open('description_languages_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_description_languages() 
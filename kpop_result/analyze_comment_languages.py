import pandas as pd
import numpy as np

# Read the CSV files
comment_lang_df = pd.read_csv('video_comment_language.csv')

def analyze_comment_languages():
    # Calculate total number of distinct languages
    exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages']
    lang_cols = [col for col in comment_lang_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(comment_lang_df[col])]
    distinct_languages = len(lang_cols)
    
    # Calculate statistics for languages per video
    avg_languages = comment_lang_df['total_languages'].mean()
    median_languages = comment_lang_df['total_languages'].median()
    min_languages = comment_lang_df['total_languages'].min()
    max_languages = comment_lang_df['total_languages'].max()
    
    # Calculate percentage of videos containing each language
    total_videos = len(comment_lang_df)
    lang_percentages = {}
    for col in lang_cols:
        videos_with_lang = (comment_lang_df[col] > 0).sum()
        percentage = (videos_with_lang / total_videos) * 100
        lang_percentages[col] = percentage
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K-pop Video Comment Language Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
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
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .language-list {{
                columns: 2;
                column-gap: 40px;
            }}
            .language-item {{
                margin-bottom: 10px;
                padding: 5px;
                background-color: white;
                border-radius: 4px;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>K-pop Video Comment Language Analysis</h1>
            
            <div class="summary">
                <h2>Key Findings</h2>
                <p>Analysis of {total_videos} K-pop videos reveals the global nature of K-pop fandom through comment language diversity.</p>
            </div>

            <div class="section">
                <h2>Overall Language Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{distinct_languages}</div>
                        <div>Distinct Languages Detected</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{avg_languages:.1f}</div>
                        <div>Average Languages per Video</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{median_languages}</div>
                        <div>Median Languages per Video</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{min_languages} - {max_languages}</div>
                        <div>Range of Languages per Video</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Language Distribution in Comments</h2>
                <div class="language-list">
    """
    
    # Add language percentages
    for lang, percentage in sorted(lang_percentages.items(), key=lambda x: x[1], reverse=True):
        highlight_class = 'highlight' if percentage > 95 else ''  # Highlight languages present in >95% of videos
        html_content += f"""
                    <div class="language-item">
                        <span class="{highlight_class}">{lang}</span>: {percentage:.1f}%
                    </div>"""
    
    html_content += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open('comment_language_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_comment_languages() 
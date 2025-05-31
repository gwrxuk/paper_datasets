import pandas as pd
import numpy as np

# Read the CSV files
video_df = pd.read_csv('video.csv')
title_lang_df = pd.read_csv('video_title_language.csv')
desc_lang_df = pd.read_csv('video_description_language.csv')

def calculate_language_stats():
    # Calculate average number of languages
    avg_title_langs = title_lang_df['total_languages'].mean()
    avg_desc_langs = desc_lang_df['total_languages'].mean()
    
    # Calculate percentage of videos containing each language
    def get_language_percentages(df):
        # Exclude metadata columns
        exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages']
        lang_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # Calculate percentage of videos containing each language
        total_videos = len(df)
        lang_percentages = {}
        for col in lang_cols:
            videos_with_lang = (df[col] > 0).sum()
            percentage = (videos_with_lang / total_videos) * 100
            lang_percentages[col] = percentage
        
        return lang_percentages
    
    title_percentages = get_language_percentages(title_lang_df)
    desc_percentages = get_language_percentages(desc_lang_df)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K-pop Video Language Distribution Analysis</title>
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
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }}
            .section {{
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
            }}
            .stat-box {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 10px;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>K-pop Video Language Distribution Analysis</h1>
            
            <div class="section">
                <h2>Average Languages per Video</h2>
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{avg_title_langs:.1f}</div>
                        <div>Languages in Titles</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{avg_desc_langs:.1f}</div>
                        <div>Languages in Descriptions</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Language Distribution in Titles</h2>
                <div class="language-list">
    """
    
    # Add title language percentages
    for lang, percentage in sorted(title_percentages.items(), key=lambda x: x[1], reverse=True):
        if percentage > 1:
            highlight_class = 'highlight' if lang in ['en', 'ko'] else ''
            html_content += f"""
                    <div class="language-item">
                        <span class="{highlight_class}">{lang}</span>: {percentage:.1f}%
                    </div>"""
    
    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>Language Distribution in Descriptions</h2>
                <div class="language-list">
    """
    
    # Add description language percentages
    for lang, percentage in sorted(desc_percentages.items(), key=lambda x: x[1], reverse=True):
        if percentage > 1:
            highlight_class = 'highlight' if lang in ['en', 'ko', 'es', 'de', 'ur'] else ''
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
    with open('language_distribution_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    calculate_language_stats() 
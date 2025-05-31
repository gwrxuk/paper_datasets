import pandas as pd
import numpy as np

# Read the CSV files
comment_lang_df = pd.read_csv('video_comment_language.csv')

def analyze_top_languages():
    # Calculate total number of videos
    total_videos = len(comment_lang_df)
    
    # Get language columns (excluding metadata columns)
    exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages']
    lang_cols = [col for col in comment_lang_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(comment_lang_df[col])]
    
    # Calculate video coverage for each language
    lang_coverage = {}
    for col in lang_cols:
        videos_with_lang = (comment_lang_df[col] > 0).sum()
        percentage = (videos_with_lang / total_videos) * 100
        lang_coverage[col] = {
            'videos': videos_with_lang,
            'percentage': percentage
        }
    
    # Sort languages by coverage and get top 10
    top_languages = sorted(lang_coverage.items(), key=lambda x: x[1]['percentage'], reverse=True)[:10]
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Top 10 Languages in K-pop YouTube Comments</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Top 10 Languages in K-pop YouTube Comments</h1>
            
            <div class="summary">
                <p>Analysis of {total_videos} K-pop videos shows the most prevalent languages in YouTube comments, 
                demonstrating the global reach of K-pop fandom.</p>
            </div>

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
        highlight_class = 'highlight' if stats['percentage'] > 95 else ''
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
    </body>
    </html>
    """
    
    # Write to file
    with open('top_languages_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_top_languages() 
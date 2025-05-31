import pandas as pd
import os
import numpy as np

def get_language_columns(df):
    """
    Get all language columns from the DataFrame.
    Excludes metadata columns like youtube_id, total_comments, total_languages, undetected_languages.
    """
    return [col for col in df.columns if col not in ['youtube_id', 'total_comments', 'total_languages', 'undetected_languages']]

def calculate_title_language_stats(df, language_cols):
    """
    Calculate statistics for language usage in video titles.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    language_cols (list): List of language column names.
    
    Returns:
    dict: A dictionary containing the calculated statistics.
    """
    # Calculate number of videos with each language in title
    videos_with_language = (df[language_cols] > 0).sum()
    
    stats = {
        'total_languages': len(language_cols),
        'total_videos': len(df),
        'max_videos_per_language': float(videos_with_language.max()),
        'min_videos_per_language': float(videos_with_language.min()),
        'median_videos_per_language': float(videos_with_language.median()),
        'average_videos_per_language': float(videos_with_language.mean()),
        'top_languages_by_video_count': videos_with_language.nlargest(10).to_dict(),
        'top_languages_by_percentage': (videos_with_language / len(df) * 100).nlargest(10).to_dict()
    }
    return stats

def generate_html_report(stats, output_path):
    """
    Generate an HTML report from the language statistics.
    
    Parameters:
    stats (dict): Statistics for language usage
    output_path (str): Path to save the HTML file
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video Title Language Statistics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .section {{ margin-bottom: 40px; }}
        .metric {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Video Title Language Statistics Report</h1>
    
    <div class="section">
        <h2>Overall Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td class="metric">Total Number of Languages</td><td>{stats['total_languages']:,.0f}</td></tr>
            <tr><td class="metric">Total Number of Videos</td><td>{stats['total_videos']:,.0f}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Videos per Language Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td class="metric">Maximum Videos per Language</td><td>{stats['max_videos_per_language']:,.0f}</td></tr>
            <tr><td class="metric">Minimum Videos per Language</td><td>{stats['min_videos_per_language']:,.0f}</td></tr>
            <tr><td class="metric">Median Videos per Language</td><td>{stats['median_videos_per_language']:,.1f}</td></tr>
            <tr><td class="metric">Average Videos per Language</td><td>{stats['average_videos_per_language']:,.2f}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Top 10 Languages by Video Count</h2>
        <table>
            <tr>
                <th>Language</th>
                <th>Number of Videos</th>
                <th>Percentage</th>
            </tr>
"""
    
    # Add top 10 languages by video count
    for lang, count in stats['top_languages_by_video_count'].items():
        percentage = (count / stats['total_videos']) * 100
        html_content += f"            <tr><td>{lang}</td><td>{count:,.0f}</td><td>{percentage:.2f}%</td></tr>\n"
    
    html_content += """        </table>
    </div>
</body>
</html>"""
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report generated at: {output_path}")

# Example usage
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv('langid_v2/video_comment_language.csv')
    
    # Get language columns
    language_cols = get_language_columns(df)
    
    # Calculate stats for video titles
    stats = calculate_title_language_stats(df, language_cols)
    print("\nVideo Title Language Statistics:")
    print(f"Total number of languages: {stats['total_languages']}")
    print(f"Total number of videos: {stats['total_videos']:,.0f}")
    print("\nTop 10 languages by video count:")
    for lang, count in stats['top_languages_by_video_count'].items():
        percentage = (count / stats['total_videos']) * 100
        print(f"{lang}: {count:,.0f} videos ({percentage:.2f}%)")
    
    # Generate HTML report
    generate_html_report(stats, 'summary_v2/title_language_stats_report.html') 
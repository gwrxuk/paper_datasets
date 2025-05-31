import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('langid_v2/video_comment_language.csv')

# Get language columns (excluding metadata columns)
language_columns = [col for col in df.columns if col not in ['youtube_id', 'total_comments', 'total_languages', 'undetected_languages']]

# Create a detailed analysis per video
video_language_analysis = []

for _, row in df.iterrows():
    video_id = row['youtube_id']
    languages_present = []
    
    # Check each language column
    for lang in language_columns:
        if row[lang] > 0:  # If the language has any comments
            percentage = (row[lang] / row['total_comments'] * 100)
            languages_present.append({
                'language': lang,
                'count': int(row[lang]),
                'percentage': round(percentage, 2)
            })
    
    # Sort languages by count
    languages_present.sort(key=lambda x: x['count'], reverse=True)
    
    video_language_analysis.append({
        'video_id': video_id,
        'total_comments': int(row['total_comments']),
        'total_languages': int(row['total_languages']),
        'undetected_languages': int(row['undetected_languages']),
        'languages': languages_present
    })

# Save detailed analysis to CSV
detailed_analysis = []
for video in video_language_analysis:
    for lang in video['languages']:
        detailed_analysis.append({
            'video_id': video['video_id'],
            'language': lang['language'],
            'comment_count': lang['count'],
            'percentage': lang['percentage'],
            'total_comments': video['total_comments'],
            'total_languages': video['total_languages']
        })

detailed_df = pd.DataFrame(detailed_analysis)
detailed_df.to_csv('detailed_language_analysis.csv', index=False)

# Generate HTML output
html_content = ['<html><head><meta charset="UTF-8"><title>Detailed Language Analysis per Video</title>',
                '<style>body{font-family:sans-serif;} table{border-collapse:collapse;margin-bottom:30px;} th,td{border:1px solid #ccc;padding:4px 8px;} th{background:#eee;} h2{margin-top:40px;}</style>',
                '</head><body>']
html_content.append('<h1>Detailed Language Analysis per Video</h1>')

for video in video_language_analysis:
    html_content.append(f'<h2>Video ID: {video["video_id"]}</h2>')
    html_content.append(f'<p><b>Total Comments:</b> {video["total_comments"]} | <b>Number of Languages:</b> {video["total_languages"]} | <b>Undetected Languages:</b> {video["undetected_languages"]}</p>')
    html_content.append('<table>')
    html_content.append('<tr><th>Language</th><th>Comment Count</th><th>Percentage (%)</th></tr>')
    for lang in video['languages']:
        html_content.append(f'<tr><td>{lang["language"]}</td><td>{lang["count"]}</td><td>{lang["percentage"]}</td></tr>')
    html_content.append('</table>')

html_content.append('</body></html>')

with open('detailed_language_analysis.html', 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_content))

print("\nAnalysis complete. Detailed results saved to 'detailed_language_analysis.csv' and 'detailed_language_analysis.html'") 
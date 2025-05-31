import pandas as pd

# Read the CSV file
df = pd.read_csv('video_title_language.csv')

# Get language columns (excluding metadata columns)
language_columns = [col for col in df.columns if col not in ['youtube_id', 'title', 'total_languages']]

# Create a detailed analysis per video
video_language_analysis = []

for _, row in df.iterrows():
    video_id = row['youtube_id']
    title = row['title']
    languages_present = []
    
    # Check each language column
    for lang in language_columns:
        if row[lang] > 0:  # If the language has any presence
            languages_present.append({
                'language': lang,
                'detected': True
            })
    
    video_language_analysis.append({
        'video_id': video_id,
        'title': title,
        'total_languages': int(row['total_languages']),
        'languages': languages_present
    })

# Generate HTML output
html_content = ['<html><head><meta charset="UTF-8"><title>Video Title Language Analysis</title>',
                '<style>body{font-family:sans-serif;} table{border-collapse:collapse;margin-bottom:30px;} th,td{border:1px solid #ccc;padding:4px 8px;} th{background:#eee;} h2{margin-top:40px;} .title{font-style:italic;}</style>',
                '</head><body>']
html_content.append('<h1>Video Title Language Analysis</h1>')

# Add summary statistics
total_videos = len(video_language_analysis)
total_languages = sum(v['total_languages'] for v in video_language_analysis)
avg_languages = total_languages / total_videos

html_content.append(f'<h2>Summary Statistics</h2>')
html_content.append(f'<p>Total Videos: {total_videos}</p>')
html_content.append(f'<p>Average Languages per Title: {avg_languages:.2f}</p>')

# Add detailed analysis for each video
for video in video_language_analysis:
    html_content.append(f'<h2>Video ID: {video["video_id"]}</h2>')
    html_content.append(f'<p class="title"><b>Title:</b> {video["title"]}</p>')
    html_content.append(f'<p><b>Number of Languages:</b> {video["total_languages"]}</p>')
    html_content.append('<table>')
    html_content.append('<tr><th>Detected Languages</th></tr>')
    for lang in video['languages']:
        html_content.append(f'<tr><td>{lang["language"]}</td></tr>')
    html_content.append('</table>')

html_content.append('</body></html>')

with open('video_title_language_analysis.html', 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_content))

print("\nAnalysis complete. Results saved to 'video_title_language_analysis.html'") 
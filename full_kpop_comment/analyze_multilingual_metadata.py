import pandas as pd
import numpy as np

# Read the data
video_lang_df = pd.read_csv('langid_v2/video_title_language.csv')
desc_lang_df = pd.read_csv('langid_v2/video_description_language.csv')

# Calculate average number of languages
avg_title_langs = video_lang_df['total_languages'].mean()
avg_desc_langs = desc_lang_df['total_languages'].mean()

# Calculate language percentages for titles
title_lang_percentages = {}
for col in video_lang_df.columns:
    if col.startswith('lang_'):
        lang = col.replace('lang_', '')
        percentage = (video_lang_df[col] > 0).mean() * 100
        title_lang_percentages[lang] = percentage

# Calculate language percentages for descriptions
desc_lang_percentages = {}
for col in desc_lang_df.columns:
    if col.startswith('lang_'):
        lang = col.replace('lang_', '')
        percentage = (desc_lang_df[col] > 0).mean() * 100
        desc_lang_percentages[lang] = percentage

# Prepare HTML content
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Multilingual Metadata Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ccc; padding: 8px 12px; }}
        th {{ background: #f4f4f4; }}
    </style>
</head>
<body>
    <h1>Multilingual Metadata Analysis</h1>
    <h2>Key Statistics</h2>
    <ul>
        <li>Average number of languages in video titles: <b>{avg_title_langs:.1f}</b></li>
        <li>Average number of languages in video descriptions: <b>{avg_desc_langs:.1f}</b></li>
    </ul>
    <h2>Language Prevalence in Video Titles</h2>
    <table>
        <tr><th>Language</th><th>Percentage of Videos</th></tr>
        {''.join(f'<tr><td>{lang}</td><td>{percentage:.1f}%</td></tr>' for lang, percentage in sorted(title_lang_percentages.items(), key=lambda x: x[1], reverse=True))}
    </table>
    <h2>Language Prevalence in Video Descriptions</h2>
    <table>
        <tr><th>Language</th><th>Percentage of Videos</th></tr>
        {''.join(f'<tr><td>{lang}</td><td>{percentage:.1f}%</td></tr>' for lang, percentage in sorted(desc_lang_percentages.items(), key=lambda x: x[1], reverse=True))}
    </table>
    <h2>Verification of Specific Statistics</h2>
    <ul>
        <li>English in titles: <b>{title_lang_percentages.get('en', 0):.1f}%</b></li>
        <li>Korean in titles: <b>{title_lang_percentages.get('ko', 0):.1f}%</b></li>
        <li>English in descriptions: <b>{desc_lang_percentages.get('en', 0):.1f}%</b></li>
        <li>Korean in descriptions: <b>{desc_lang_percentages.get('ko', 0):.1f}%</b></li>
        <li>Spanish in descriptions: <b>{desc_lang_percentages.get('es', 0):.1f}%</b></li>
        <li>German in descriptions: <b>{desc_lang_percentages.get('de', 0):.1f}%</b></li>
        <li>Urdu in descriptions: <b>{desc_lang_percentages.get('ur', 0):.1f}%</b></li>
    </ul>
</body>
</html>
'''

# Save results to an HTML file
with open('result_v2/multilingual_metadata_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("HTML report saved to result_v2/multilingual_metadata_analysis.html") 
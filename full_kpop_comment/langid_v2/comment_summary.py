import csv
import os

csv_file = 'langid/video_comment_language.csv'
html_file = 'langid/comment.html'

comment_rows = []
language_usage = {}

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_columns = reader.fieldnames[3:]
    for lang in all_columns:
        language_usage[lang] = 0
    for row in reader:
        lang_count = sum(int(row[lang]) for lang in all_columns)
        comment_rows.append({
            'youtube_id': row['youtube_id'],
            'comment': row['comment'],
            'lang_count': lang_count,
            'langs': {lang: int(row[lang]) for lang in all_columns}
        })
        for lang in all_columns:
            if int(row[lang]) > 0:
                language_usage[lang] += 1

num_comments = len(comment_rows)
num_languages = len(all_columns)

# Most/least used languages
max_lang_comment_count = max(language_usage.values())
min_lang_comment_count = min(language_usage.values())
most_used_langs = [lang for lang, count in language_usage.items() if count == max_lang_comment_count]
least_used_langs = [lang for lang, count in language_usage.items() if count == min_lang_comment_count]

# Comments with most/least languages
max_lang_count = max(row['lang_count'] for row in comment_rows)
min_lang_count = min(row['lang_count'] for row in comment_rows)
max_lang_comments = [row for row in comment_rows if row['lang_count'] == max_lang_count]
min_lang_comments = [row for row in comment_rows if row['lang_count'] == min_lang_count]

# Prepare bar chart data
max_bar = max(language_usage.values())
bar_chart_html = '<h2>Language Usage Bar Chart</h2><div style="font-family:monospace">'
for lang in sorted(language_usage, key=language_usage.get, reverse=True):
    count = language_usage[lang]
    bar = '█' * int(50 * count / max_bar)
    bar_chart_html += f'<div><span style="display:inline-block;width:60px">{lang}</span> {bar} {count}</div>'
bar_chart_html += '</div>'

with open(html_file, 'w', encoding='utf-8') as f:
    f.write('<html><head><title>LangID Comment Language Summary</title></head><body>')
    f.write('<h1>LangID Comment Language Summary</h1>')
    f.write(f'<p><strong>Number of comments:</strong> {num_comments}</p>')
    all_langs_str = ', '.join(all_columns)
    f.write(f'<p><strong>Number of unique languages in comments:</strong> {num_languages} ({all_langs_str})</p>')
    f.write(f'<p><strong>Most used language(s) ({max_lang_comment_count} comments):</strong> {", ".join(most_used_langs)}</p>')
    f.write(f'<p><strong>Least used language(s) ({min_lang_comment_count} comments):</strong> {", ".join(least_used_langs)}</p>')
    f.write(f'<p><strong>Most languages used in a comment:</strong> {max_lang_count}</p>')
    f.write('<ul>')
    for row in max_lang_comments:
        langs_in_comment = [lang for lang in all_columns if row['langs'].get(lang, 0) > 0]
        langs_str = ', '.join(langs_in_comment)
        f.write(f'<li>{row["youtube_id"]}: <br><em>Languages: {langs_str}</em></li>')
    f.write('</ul>')
    f.write(f'<p><strong>Least languages used in a comment:</strong> {min_lang_count}</p>')
    f.write('<ul>')
    for row in min_lang_comments:
        f.write(f'<li>{row["youtube_id"]}</li>')
    f.write('</ul>')
    f.write(bar_chart_html)
    f.write('</body></html>')

print(f"Summary and visualization saved to {html_file}") 
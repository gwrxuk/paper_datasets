import csv
import os

csv_file = 'langid/video_title_language.csv'
html_file = 'langid/title.html'

title_rows = []
language_usage = {}
language_prob_sums = {}

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    # Separate count and probability columns
    all_columns = reader.fieldnames[3:]
    count_columns = [col for col in all_columns if not col.endswith('_prob')]
    prob_columns = [col for col in all_columns if col.endswith('_prob')]
    for lang in count_columns:
        language_usage[lang] = 0
        language_prob_sums[lang] = 0.0
    for row in reader:
        lang_count = sum(int(row[lang]) for lang in count_columns)
        title_rows.append({
            'youtube_id': row['youtube_id'],
            'title': row['title'],
            'lang_count': lang_count,
            'langs': {lang: int(row[lang]) for lang in count_columns},
            'probs': {lang: float(row.get(f'{lang}_prob', 0.0)) for lang in count_columns}
        })
        for lang in count_columns:
            if int(row[lang]) > 0:
                language_usage[lang] += 1
            language_prob_sums[lang] += float(row.get(f'{lang}_prob', 0.0))

num_videos = len(title_rows)
num_languages = len(count_columns)

# Most/least used languages
max_lang_video_count = max(language_usage.values())
min_lang_video_count = min(language_usage.values())
most_used_langs = [lang for lang, count in language_usage.items() if count == max_lang_video_count]
least_used_langs = [lang for lang, count in language_usage.items() if count == min_lang_video_count]

# Videos with most/least languages
max_lang_count = max(row['lang_count'] for row in title_rows)
min_lang_count = min(row['lang_count'] for row in title_rows)
max_lang_videos = [row for row in title_rows if row['lang_count'] == max_lang_count]
min_lang_videos = [row for row in title_rows if row['lang_count'] == min_lang_count]

# Prepare bar chart data
max_bar = max(language_usage.values())
bar_chart_html = '<h2>Language Usage Bar Chart</h2><div style="font-family:monospace">'
for lang in sorted(language_usage, key=language_usage.get, reverse=True):
    count = language_usage[lang]
    bar = '█' * int(50 * count / max_bar)
    bar_chart_html += f'<div><span style="display:inline-block;width:60px">{lang}</span> {bar} {count}</div>'
bar_chart_html += '</div>'

# Prepare probability summary section
prob_html = '<h2>Summed Language Probabilities</h2><div style="font-family:monospace">'
max_prob = max(abs(p) for p in language_prob_sums.values()) or 1
for lang in sorted(language_prob_sums, key=language_prob_sums.get, reverse=True):
    prob = language_prob_sums[lang]
    bar = '█' * int(50 * abs(prob) / max_prob)
    prob_html += f'<div><span style="display:inline-block;width:60px">{lang}</span> {bar} {prob:.2f}</div>'
prob_html += '</div>'

with open(html_file, 'w', encoding='utf-8') as f:
    f.write('<html><head><title>LangID Title Language Summary</title></head><body>')
    f.write('<h1>LangID Title Language Summary</h1>')
    f.write(f'<p><strong>Number of videos:</strong> {num_videos}</p>')
    f.write(f'<p><strong>Number of unique languages in title:</strong> {num_languages}</p>')
    f.write(f'<p><strong>Most used language(s) ({max_lang_video_count} videos):</strong> {", ".join(most_used_langs)}</p>')
    f.write(f'<p><strong>Least used language(s) ({min_lang_video_count} videos):</strong> {", ".join(least_used_langs)}</p>')
    f.write(f'<p><strong>Most languages used in a title:</strong> {max_lang_count}</p>')
    f.write('<ul>')
    for row in max_lang_videos:
        langs_in_title = [lang for lang in count_columns if row['langs'].get(lang, 0) > 0]
        langs_str = ', '.join(langs_in_title)
        # Show probabilities for these languages
        prob_str = ', '.join(f"{lang}: {row['probs'][lang]:.2f}" for lang in langs_in_title)
        f.write(f'<li>{row["youtube_id"]}: {row["title"]} <br><em>Languages: {langs_str}</em><br><em>Probabilities: {prob_str}</em></li>')
    f.write('</ul>')
    # New section: Languages used in the most language-used video(s)
    f.write('<p><strong>Languages used in the most language-used video(s):</strong></p>')
    # Collect all languages used in any of the max_lang_videos
    langs_in_max = set()
    for row in max_lang_videos:
        for lang in count_columns:
            if int(row['langs'].get(lang, 0)) > 0:
                langs_in_max.add(lang)
    f.write('<ul>')
    for lang in sorted(langs_in_max):
        f.write(f'<li>{lang}</li>')
    f.write('</ul>')
    f.write(f'<p><strong>Least languages used in a title:</strong> {min_lang_count}</p>')
    f.write('<ul>')
    for row in min_lang_videos:
        f.write(f'<li>{row["youtube_id"]}: {row["title"]}</li>')
    f.write('</ul>')
    f.write(bar_chart_html)
    f.write(prob_html)
    f.write('</body></html>')

print(f"Summary and visualization saved to {html_file}") 
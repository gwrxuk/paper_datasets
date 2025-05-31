import csv

# File path
csv_file = 'video_title_language.csv'
html_file = 'video_title_summary.html'

# Initialize variables
max_lang_count = -1
min_lang_count = float('inf')
max_titles = []
min_titles = []
max_title_langs = []
min_title_langs = []

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    language_columns = header[3:]  # language columns start from index 3
    num_languages = len(language_columns)
    rows = list(reader)
    num_videos = len(rows)

    # For language usage count
    lang_usage = {lang: 0 for lang in language_columns}

    for row in rows:
        title = row[1]
        lang_flags = [int(x) for x in row[3:]]
        lang_count = sum(lang_flags)
        # Find languages used in this title
        langs_in_title = [lang for lang, flag in zip(language_columns, lang_flags) if flag > 0]
        if lang_count > max_lang_count:
            max_lang_count = lang_count
            max_titles = [title]
            max_title_langs = [langs_in_title]
        elif lang_count == max_lang_count:
            max_titles.append(title)
            max_title_langs.append(langs_in_title)
        if lang_count < min_lang_count:
            min_lang_count = lang_count
            min_titles = [title]
            min_title_langs = [langs_in_title]
        elif lang_count == min_lang_count:
            min_titles.append(title)
            min_title_langs.append(langs_in_title)
        # Count language usage
        for i, lang in enumerate(language_columns):
            if lang_flags[i] > 0:
                lang_usage[lang] += 1

# Find most and least used languages
max_lang_video_count = max(lang_usage.values())
min_lang_video_count = min(lang_usage.values())
most_used_langs = [lang for lang, count in lang_usage.items() if count == max_lang_video_count]
least_used_langs = [lang for lang, count in lang_usage.items() if count == min_lang_video_count]

# Write to HTML file
with open(html_file, 'w', encoding='utf-8') as f:
    f.write('<html><head><title>Video Title Language Summary</title></head><body>')
    f.write('<h1>Video Title Language Summary</h1>')
    f.write(f'<p><strong>Number of videos:</strong> {num_videos}</p>')
    f.write(f'<p><strong>Number of unique languages in title:</strong> {num_languages}</p>')
    f.write(f'<p><strong>Most languages used in a title:</strong> {max_lang_count}</p>')
    f.write('<ul>')
    for t, langs in zip(max_titles, max_title_langs):
        f.write(f'<li>{t}<br><em>Languages used: {", ".join(langs)}</em></li>')
    f.write('</ul>')
    f.write(f'<p><strong>Least languages used in a title:</strong> {min_lang_count}</p>')
    f.write('<ul>')
    for t, langs in zip(min_titles, min_title_langs):
        f.write(f'<li>{t}<br><em>Languages used: {", ".join(langs) if langs else "None"}</em></li>')
    f.write('</ul>')
    f.write(f'<p><strong>Language(s) in the most videos ({max_lang_video_count} videos):</strong></p>')
    f.write('<ul>')
    for lang in most_used_langs:
        f.write(f'<li>{lang}</li>')
    f.write('</ul>')
    f.write(f'<p><strong>Language(s) in the least videos ({min_lang_video_count} videos):</strong></p>')
    f.write('<ul>')
    for lang in least_used_langs:
        f.write(f'<li>{lang}</li>')
    f.write('</ul>')
    f.write('</body></html>') 
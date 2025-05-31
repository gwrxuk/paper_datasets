import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import re
from scipy import stats
import jinja2
import os

# Read the CSV file
df = pd.read_csv('langid_v2/video_title_language.csv')

# 1. Basic Language Distribution
language_columns = [col for col in df.columns if col not in ['youtube_id', 'title', 'total_languages']]
language_stats = df[language_columns].sum().sort_values(ascending=False)

# 2. Language Diversity Analysis
df['language_diversity'] = df[language_columns].apply(lambda x: (x > 0).sum(), axis=1)
diversity_stats = df['language_diversity'].value_counts().sort_index()

# 3. Language Pair Analysis
language_pairs = []
for _, row in df.iterrows():
    present_langs = [lang for lang in language_columns if row[lang] > 0]
    for i in range(len(present_langs)):
        for j in range(i+1, len(present_langs)):
            language_pairs.append(tuple(sorted([present_langs[i], present_langs[j]])))
pair_counts = Counter(language_pairs)

# 4. Title Length Analysis
df['title_length'] = df['title'].str.len()
df['word_count'] = df['title'].str.split().str.len()

# 5. Language Dominance Analysis
df['dominant_language'] = df[language_columns].idxmax(axis=1)
dominant_lang_stats = df['dominant_language'].value_counts()

# 6. Language Score Distribution
language_scores = df[language_columns].melt()
score_stats = language_scores.groupby('variable')['value'].agg(['mean', 'std', 'max'])

# 7. Language Transition Analysis
df['has_transition'] = df['title'].str.contains(r'[가-힣].*[a-zA-Z]|[a-zA-Z].*[가-힣]', regex=True)
transition_stats = df['has_transition'].value_counts()

# 8. Special Character Analysis
df['special_chars'] = df['title'].str.count(r'[#@!$%^&*()_+\-=\[\]{};\'"\\|,.<>/?]')
special_char_stats = df['special_chars'].describe()

# 9. Language Confidence Analysis
df['max_confidence'] = df[language_columns].max(axis=1)
confidence_stats = df['max_confidence'].describe()

# 10. Language Mixing Patterns
df['language_mix_type'] = df.apply(lambda x: ' '.join(sorted([lang for lang in language_columns if x[lang] > 0])), axis=1)
mix_patterns = df['language_mix_type'].value_counts().head(10)

# Create visualizations
fig = make_subplots(
    rows=5, cols=4,
    subplot_titles=(
        '1. Top 20 Languages Distribution',
        '2. Language Diversity Distribution',
        '3. Most Common Language Pairs',
        '4. Title Length Distribution',
        '5. Dominant Language Distribution',
        '6. Language Score Statistics',
        '7. Language Transition Analysis',
        '8. Special Character Distribution',
        '9. Language Confidence Distribution',
        '10. Language Mixing Patterns',
        '11. Title Length vs Language Count',
        '12. Language Score Correlation',
        '13. Special Characters vs Languages',
        '14. Word Count Distribution',
        '15. Language Confidence vs Diversity',
        '16. Title Length vs Word Count',
        '17. Language Pair Heatmap',
        '18. Language Score Box Plot',
        '19. Language Transition Patterns',
        '20. Language Mixing Network'
    ),
    vertical_spacing=0.05,
    horizontal_spacing=0.05
)

# 1. Top 20 Languages
fig.add_trace(
    go.Bar(x=language_stats.head(20).index, y=language_stats.head(20).values),
    row=1, col=1
)

# 2. Language Diversity
fig.add_trace(
    go.Bar(x=diversity_stats.index, y=diversity_stats.values),
    row=1, col=2
)

# 3. Language Pairs
top_pairs = dict(pair_counts.most_common(10))
fig.add_trace(
    go.Bar(x=list(top_pairs.keys()), y=list(top_pairs.values())),
    row=1, col=3
)

# 4. Title Length
fig.add_trace(
    go.Histogram(x=df['title_length'], nbinsx=50),
    row=1, col=4
)

# 5. Dominant Language
fig.add_trace(
    go.Bar(x=dominant_lang_stats.index, y=dominant_lang_stats.values),
    row=2, col=1
)

# 6. Language Score Statistics
fig.add_trace(
    go.Box(y=language_scores['value'], x=language_scores['variable']),
    row=2, col=2
)

# 7. Language Transition
fig.add_trace(
    go.Bar(x=transition_stats.index, y=transition_stats.values),
    row=2, col=3
)

# 8. Special Characters
fig.add_trace(
    go.Histogram(x=df['special_chars'], nbinsx=20),
    row=2, col=4
)

# 9. Language Confidence
fig.add_trace(
    go.Histogram(x=df['max_confidence'], nbinsx=30),
    row=3, col=1
)

# 10. Language Mixing
fig.add_trace(
    go.Bar(x=mix_patterns.index, y=mix_patterns.values),
    row=3, col=2
)

# 11. Title Length vs Language Count
fig.add_trace(
    go.Scatter(x=df['title_length'], y=df['language_diversity'], mode='markers'),
    row=3, col=3
)

# 12. Language Score Correlation
correlation_matrix = df[language_columns].corr()
fig.add_trace(
    go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index),
    row=3, col=4
)

# 13. Special Characters vs Languages
fig.add_trace(
    go.Box(x=df['special_chars'], y=df['language_diversity']),
    row=4, col=1
)

# 14. Word Count Distribution
fig.add_trace(
    go.Histogram(x=df['word_count'], nbinsx=30),
    row=4, col=2
)

# 15. Language Confidence vs Diversity
fig.add_trace(
    go.Scatter(x=df['max_confidence'], y=df['language_diversity'], mode='markers'),
    row=4, col=3
)

# 16. Title Length vs Word Count
fig.add_trace(
    go.Scatter(x=df['title_length'], y=df['word_count'], mode='markers'),
    row=4, col=4
)

# 17. Language Pair Heatmap
pair_matrix = pd.DataFrame(0, index=language_columns, columns=language_columns)
for pair, count in pair_counts.items():
    pair_matrix.loc[pair[0], pair[1]] = count
    pair_matrix.loc[pair[1], pair[0]] = count
fig.add_trace(
    go.Heatmap(z=pair_matrix.values, x=pair_matrix.columns, y=pair_matrix.index),
    row=5, col=1
)

# 18. Language Score Box Plot
fig.add_trace(
    go.Box(y=df[language_columns].melt()['value'], x=df[language_columns].melt()['variable']),
    row=5, col=2
)

# 19. Language Transition Patterns
transition_by_lang = df.groupby('dominant_language')['has_transition'].mean()
fig.add_trace(
    go.Bar(x=transition_by_lang.index, y=transition_by_lang.values),
    row=5, col=3
)

# 20. Language Mixing Network
mix_network = pd.DataFrame(0, index=language_columns, columns=language_columns)
for mix in df['language_mix_type']:
    langs = mix.split()
    for i in range(len(langs)):
        for j in range(i+1, len(langs)):
            mix_network.loc[langs[i], langs[j]] += 1
            mix_network.loc[langs[j], langs[i]] += 1
fig.add_trace(
    go.Heatmap(z=mix_network.values, x=mix_network.columns, y=mix_network.index),
    row=5, col=4
)

# Generate analysis text
analysis_text = {
    'overview': f"""
    <h2>Overview</h2>
    <p>This analysis examines the language distribution in {len(df):,} K-pop video titles. The dataset reveals interesting patterns in how different languages are used in video titles, particularly focusing on the mixing of Korean and other languages.</p>
    """,
    
    'language_distribution': f"""
    <h2>Language Distribution Analysis</h2>
    <p>The most common languages found in video titles are:</p>
    <ul>
        <li>English (en): {language_stats.get('en', 0):,} occurrences</li>
        <li>Korean (ko): {language_stats.get('ko', 0):,} occurrences</li>
        <li>Japanese (ja): {language_stats.get('ja', 0):,} occurrences</li>
    </ul>
    <p>On average, each title contains {df['language_diversity'].mean():.2f} different languages, indicating a high degree of language mixing in K-pop video titles.</p>
    """,
    
    'languages_per_video': f"""
    <h2>Languages per Video Analysis</h2>
    <p>Distribution of languages used in video titles:</p>
    <ul>
        {''.join(f'<li>{count} languages: {diversity_stats.get(count, 0):,} videos ({diversity_stats.get(count, 0)/len(df)*100:.1f}%)</li>' for count in range(1, diversity_stats.index.max() + 1))}
    </ul>
    <p>Key observations:</p>
    <ul>
        <li>Most common number of languages: {diversity_stats.idxmax()} languages ({diversity_stats.max():,} videos)</li>
        <li>Single language titles: {diversity_stats.get(1, 0):,} videos ({diversity_stats.get(1, 0)/len(df)*100:.1f}%)</li>
        <li>Maximum languages in a single title: {diversity_stats.index.max()} languages</li>
        <li>Videos with 3 or more languages: {diversity_stats[diversity_stats.index >= 3].sum():,} videos ({diversity_stats[diversity_stats.index >= 3].sum()/len(df)*100:.1f}%)</li>
    </ul>
    """,
    
    'language_mixing': f"""
    <h2>Language Mixing Patterns</h2>
    <p>Language mixing is a common phenomenon in K-pop video titles:</p>
    <ul>
        <li>{transition_stats[True]/len(df)*100:.1f}% of titles contain language transitions (mixing of Korean and English characters)</li>
        <li>The most common language pair is English-Korean, appearing in {pair_counts.get(('en', 'ko'), 0):,} titles</li>
        <li>The most frequent language mix pattern is "{mix_patterns.index[0]}"</li>
    </ul>
    """,
    
    'title_characteristics': f"""
    <h2>Title Characteristics</h2>
    <p>Analysis of title structure reveals:</p>
    <ul>
        <li>Average title length: {df['title_length'].mean():.1f} characters</li>
        <li>Average word count: {df['word_count'].mean():.1f} words</li>
        <li>Average number of special characters: {df['special_chars'].mean():.1f}</li>
    </ul>
    """,
    
    'language_confidence': f"""
    <h2>Language Detection Confidence</h2>
    <p>The language detection system shows:</p>
    <ul>
        <li>Average confidence score: {df['max_confidence'].mean():.2f}</li>
        <li>Most confident language: {dominant_lang_stats.index[0]} (appears in {dominant_lang_stats.iloc[0]:,} titles)</li>
    </ul>
    """,
    
    'correlation_analysis': f"""
    <h2>Correlation Analysis</h2>
    <p>Key correlations found:</p>
    <ul>
        <li>Title length and language count: {df['title_length'].corr(df['language_diversity']):.2f}</li>
        <li>Word count and language count: {df['word_count'].corr(df['language_diversity']):.2f}</li>
        <li>Special characters and language count: {df['special_chars'].corr(df['language_diversity']):.2f}</li>
    </ul>
    """
}

# Create HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>K-pop Video Title Language Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .visualization {
            margin: 20px 0;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .analysis-section {
            background-color: #f9f9f9;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        ul {
            list-style-type: none;
            padding-left: 20px;
        }
        li {
            margin: 10px 0;
        }
        .highlight {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Analysis of Language Distribution in K-pop Video Titles</h1>
        
        {{ overview }}
        
        <div class="visualization">
            {{ plot_div|safe }}
        </div>
        
        {{ language_distribution }}
        
        {{ languages_per_video }}
        
        {{ language_mixing }}
        
        {{ title_characteristics }}
        
        {{ language_confidence }}
        
        {{ correlation_analysis }}
    </div>
</body>
</html>
"""

# Create the HTML report
template = jinja2.Template(html_template)
html_content = template.render(
    plot_div=fig.to_html(full_html=False),
    **analysis_text
)

# Ensure the result_v2 directory exists
os.makedirs('result_v2', exist_ok=True)

# Save the HTML report
with open('result_v2/language_distribution_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of videos: {len(df)}")
print(f"Average number of languages per title: {df['language_diversity'].mean():.2f}")
print(f"Most common language pair: {max(pair_counts.items(), key=lambda x: x[1])[0]}")
print(f"Average title length: {df['title_length'].mean():.2f} characters")
print(f"Most common dominant language: {dominant_lang_stats.index[0]}")
print(f"Percentage of titles with language transitions: {transition_stats[True]/len(df)*100:.2f}%")
print(f"Average number of special characters: {df['special_chars'].mean():.2f}")
print(f"Average language confidence: {df['max_confidence'].mean():.2f}")
print(f"Most common language mix: {mix_patterns.index[0]}")
print(f"Average word count: {df['word_count'].mean():.2f}") 
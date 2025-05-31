import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import jinja2
import os

# Read the data
video_lang_df = pd.read_csv('langid_v2/video_title_language.csv')
comments_df = pd.read_csv('langid_v2/video_comment_language.csv')

# Calculate video-level statistics
video_stats = pd.read_csv('langid_v2/video_comment_language.csv')

# Merge with video titles
video_stats = video_stats.merge(video_lang_df[['youtube_id', 'title']], on='youtube_id', how='left')

# Calculate overall statistics
total_comments = video_stats['total_comments'].sum()
total_videos = len(video_stats)
videos_with_comments = (video_stats['total_comments'] > 0).sum()
avg_comments = video_stats['total_comments'].mean()
median_comments = video_stats['total_comments'].median()
std_comments = video_stats['total_comments'].std()

# Find top videos
top_videos = video_stats.nlargest(10, 'total_comments')
top_videos_percentage = (top_videos['total_comments'].sum() / total_comments) * 100

# Find the video with most languages
most_languages_video = video_stats.loc[video_stats['total_languages'].idxmax()]

# Generate analysis text
analysis_text = {
    'overview': f"""
    <h2>Video Engagement Analysis</h2>
    <p>This study's dataset comprises {total_comments:,} comments across {total_videos:,} K-pop YouTube videos, with {videos_with_comments/total_videos*100:.1f}% of the videos receiving at least one comment. On average, each video garnered about {avg_comments:,.0f} comments (median ~{median_comments:,.0f}), but the distribution is highly skewed (standard deviation ≈{std_comments:,.0f}).</p>
    """,
    
    'top_videos': f"""
    <h2>Top Videos Analysis</h2>
    <p>The top 10 most-commented videos ({len(top_videos)/total_videos*100:.1f}% of the sample) account for roughly {top_videos_percentage:.1f}% of all comments, underscoring the concentration of audience activity in a few mega-popular uploads.</p>
    <h3>Top 10 Most Commented Videos:</h3>
    <ul>
        {''.join(f'<li>{row["title"]}: {row["total_comments"]:,} comments, {row["total_languages"]} languages</li>' for _, row in top_videos.iterrows())}
    </ul>
    """,
    
    'language_diversity': f"""
    <h2>Language Diversity Analysis</h2>
    <p>Language diversity across videos:</p>
    <ul>
        <li>Average number of languages per video: {video_stats['total_languages'].mean():.1f}</li>
        <li>Maximum number of languages in a single video: {video_stats['total_languages'].max()}</li>
        <li>Most language-diverse video: {most_languages_video['title']} with {most_languages_video['total_languages']} languages</li>
    </ul>
    """
}

# Create visualizations
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Distribution of Comments per Video',
        'Top 10 Most Commented Videos',
        'Distribution of Languages per Video',
        'Comments vs Languages Scatter Plot'
    )
)

# 1. Comments distribution
fig.add_trace(
    go.Histogram(x=video_stats['total_comments'], nbinsx=50, name='Comments per Video'),
    row=1, col=1
)

# 2. Top 10 videos
fig.add_trace(
    go.Bar(
        x=top_videos['title'],
        y=top_videos['total_comments'],
        name='Comment Count'
    ),
    row=1, col=2
)

# 3. Languages distribution
fig.add_trace(
    go.Histogram(x=video_stats['total_languages'], nbinsx=30, name='Languages per Video'),
    row=2, col=1
)

# 4. Comments vs Languages scatter
fig.add_trace(
    go.Scatter(
        x=video_stats['total_comments'],
        y=video_stats['total_languages'],
        mode='markers',
        name='Comments vs Languages'
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=1000,
    width=1200,
    title_text='K-pop Video Engagement Analysis',
    showlegend=False
)

# Create HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>K-pop Video Engagement Analysis</title>
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
        <h1>K-pop Video Engagement Analysis</h1>
        
        {{ overview }}
        
        <div class="visualization">
            {{ plot_div|safe }}
        </div>
        
        {{ top_videos }}
        
        {{ language_diversity }}
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
with open('result_v2/video_engagement_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total comments: {total_comments:,}")
print(f"Total videos: {total_videos:,}")
print(f"Videos with comments: {videos_with_comments:,} ({videos_with_comments/total_videos*100:.1f}%)")
print(f"Average comments per video: {avg_comments:,.0f}")
print(f"Median comments per video: {median_comments:,.0f}")
print(f"Standard deviation: {std_comments:,.0f}")
print(f"Top 10 videos comment percentage: {top_videos_percentage:.1f}%")
print(f"Average languages per video: {video_stats['total_languages'].mean():.1f}")
print(f"Maximum languages in a video: {video_stats['total_languages'].max()}") 
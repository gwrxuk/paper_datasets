import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Read the CSV files
video_df = pd.read_csv('video.csv')
title_lang_df = pd.read_csv('video_title_language.csv')
desc_lang_df = pd.read_csv('video_description_language.csv')
comment_lang_df = pd.read_csv('video_comment_language.csv')

# Function to get top languages
# Only sum numeric columns (language counts)
def get_top_languages(df, n=10):
    # Exclude metadata columns
    exclude_cols = ['youtube_id', 'title', 'description', 'total_comments', 'total_languages', 'undetected_languages']
    lang_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    lang_totals = df[lang_cols].sum(numeric_only=True)
    top_langs = lang_totals.sort_values(ascending=False).head(n)
    return top_langs

# Get top languages for each type
top_title_langs = get_top_languages(title_lang_df)
top_desc_langs = get_top_languages(desc_lang_df)
top_comment_langs = get_top_languages(comment_lang_df)

# Create visualizations
def create_language_plots():
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Top Languages in Titles', 'Top Languages in Descriptions', 'Top Languages in Comments'),
        vertical_spacing=0.1
    )

    # Add title language bar chart
    fig.add_trace(
        go.Bar(x=top_title_langs.index, y=top_title_langs.values, name='Titles'),
        row=1, col=1
    )

    # Add description language bar chart
    fig.add_trace(
        go.Bar(x=top_desc_langs.index, y=top_desc_langs.values, name='Descriptions'),
        row=2, col=1
    )

    # Add comment language bar chart
    fig.add_trace(
        go.Bar(x=top_comment_langs.index, y=top_comment_langs.values, name='Comments'),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text='Language Distribution Analysis',
        title_x=0.5
    )

    return fig

# Create statistics
def generate_statistics():
    stats = {
        'Total Videos': len(video_df),
        'Average Languages per Title': title_lang_df['total_languages'].mean(),
        'Average Languages per Description': desc_lang_df['total_languages'].mean(),
        'Average Languages per Comment Section': comment_lang_df['total_languages'].mean(),
        'Most Common Title Language': top_title_langs.index[0],
        'Most Common Description Language': top_desc_langs.index[0],
        'Most Common Comment Language': top_comment_langs.index[0]
    }
    return stats

# Generate HTML report
def generate_html_report():
    # Create visualizations
    fig = create_language_plots()
    
    # Get statistics
    stats = generate_statistics()
    
    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <title>Language Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .stats-container {{ margin: 20px 0; padding: 20px; background-color: #f5f5f5; border-radius: 5px; }}
            .stat-item {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Language Analysis Report</h1>
        
        <div class="stats-container">
            <h2>Key Statistics</h2>
            <div class="stat-item">Total Videos: {stats['Total Videos']}</div>
            <div class="stat-item">Average Languages per Title: {stats['Average Languages per Title']:.2f}</div>
            <div class="stat-item">Average Languages per Description: {stats['Average Languages per Description']:.2f}</div>
            <div class="stat-item">Average Languages per Comment Section: {stats['Average Languages per Comment Section']:.2f}</div>
            <div class="stat-item">Most Common Title Language: {stats['Most Common Title Language']}</div>
            <div class="stat-item">Most Common Description Language: {stats['Most Common Description Language']}</div>
            <div class="stat-item">Most Common Comment Language: {stats['Most Common Comment Language']}</div>
        </div>
        
        <div id="visualization">
            {fig.to_html(full_html=False)}
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open('language_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

# Generate the report
generate_html_report() 
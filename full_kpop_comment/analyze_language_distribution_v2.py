import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from datetime import datetime

def load_and_prepare_data(file_path):
    """Load data from CSV and prepare it for analysis."""
    # Load the language analysis data
    df = pd.read_csv(file_path)
    
    # Load the original video data to get total video count
    video_df = pd.read_csv('video.csv')
    total_videos = len(video_df)
    
    # Pivot the data to get one row per video
    df_pivot = df.pivot(index='video_id', columns='language', values='comment_count').fillna(0)
    
    # Add total comments and total languages columns
    df_pivot['total_comments'] = df_pivot.sum(axis=1)
    df_pivot['total_languages'] = (df_pivot > 0).sum(axis=1)
    
    return df_pivot, total_videos

def create_language_distribution_plot(df, output_dir):
    """Create a bar chart showing the distribution of languages."""
    # Calculate total comments per language
    language_totals = df.drop(['total_comments', 'total_languages'], axis=1).sum()
    language_totals = language_totals.sort_values(ascending=False)
    
    # Create the plot
    fig = px.bar(
        x=language_totals.index,
        y=language_totals.values,
        title='Distribution of Languages in Comments',
        labels={'x': 'Language', 'y': 'Total Comments'},
        color=language_totals.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Language',
        yaxis_title='Total Comments',
        showlegend=False
    )
    
    # Save the plot
    fig.write_html(os.path.join(output_dir, 'language_distribution.html'))

def create_top_languages_plot(df, output_dir):
    """Create a pie chart showing the top languages."""
    # Calculate total comments per language
    language_totals = df.drop(['total_comments', 'total_languages'], axis=1).sum()
    language_totals = language_totals.sort_values(ascending=False)
    
    # Take top 10 languages and combine others
    top_10 = language_totals.head(10)
    others = pd.Series({'Others': language_totals[10:].sum()})
    combined = pd.concat([top_10, others])
    
    # Create the plot
    fig = px.pie(
        values=combined.values,
        names=combined.index,
        title='Top Languages in Comments',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Save the plot
    fig.write_html(os.path.join(output_dir, 'top_languages.html'))

def create_language_correlation_plot(df, output_dir):
    """Create a heatmap showing correlations between languages."""
    # Calculate correlations between languages
    corr_matrix = df.drop(['total_comments', 'total_languages'], axis=1).corr()
    
    # Create the plot
    fig = px.imshow(
        corr_matrix,
        title='Language Correlations in Comments',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    fig.update_layout(
        xaxis_title='Language',
        yaxis_title='Language'
    )
    
    # Save the plot
    fig.write_html(os.path.join(output_dir, 'language_correlation.html'))

def create_language_prevalence_plot(df, output_dir):
    """Create a bar chart showing language prevalence in videos."""
    # Calculate how many videos have each language
    language_prevalence = (df.drop(['total_comments', 'total_languages'], axis=1) > 0).sum()
    language_prevalence = language_prevalence.sort_values(ascending=False)
    
    # Create the plot
    fig = px.bar(
        x=language_prevalence.index,
        y=language_prevalence.values,
        title='Language Prevalence in Videos',
        labels={'x': 'Language', 'y': 'Number of Videos'},
        color=language_prevalence.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Language',
        yaxis_title='Number of Videos',
        showlegend=False
    )
    
    # Save the plot
    fig.write_html(os.path.join(output_dir, 'language_prevalence.html'))

def create_language_combination_plot(df, output_dir):
    """Create a histogram showing the number of languages per video."""
    # Create the plot
    fig = px.histogram(
        df,
        x='total_languages',
        title='Distribution of Languages per Video',
        labels={'x': 'Number of Languages', 'y': 'Number of Videos'},
        nbins=20
    )
    
    fig.update_layout(
        xaxis_title='Number of Languages',
        yaxis_title='Number of Videos'
    )
    
    # Save the plot
    fig.write_html(os.path.join(output_dir, 'language_combinations.html'))

def generate_html_report(df, total_videos, output_dir):
    """Generate a comprehensive HTML report with all visualizations and statistics."""
    # Calculate statistics
    total_comments = df['total_comments'].sum()
    avg_comments = df['total_comments'].mean()
    median_comments = df['total_comments'].median()
    avg_languages = df['total_languages'].mean()
    median_languages = df['total_languages'].median()
    
    # Get top 5 languages by total comments
    language_totals = df.drop(['total_comments', 'total_languages'], axis=1).sum()
    top_5_languages = language_totals.nlargest(5)
    
    # Get notable correlations
    corr_matrix = df.drop(['total_comments', 'total_languages'], axis=1).corr()
    notable_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.3:  # Only show strong correlations
                notable_correlations.append({
                    'lang1': corr_matrix.columns[i],
                    'lang2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i,j]
                })
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K-pop Video Language Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 30px; }}
            h1, h2 {{ color: #333; }}
            .stat {{ margin: 10px 0; }}
            .visualization {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>K-pop Video Language Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="stat">Total Videos in Dataset: {total_videos:,}</div>
            <div class="stat">Videos with Language Analysis: {len(df):,} ({(len(df)/total_videos*100):.1f}%)</div>
            <div class="stat">Total Comments Analyzed: {total_comments:,}</div>
            <div class="stat">Average Comments per Video: {avg_comments:.1f}</div>
            <div class="stat">Median Comments per Video: {median_comments:.1f}</div>
            <div class="stat">Average Languages per Video: {avg_languages:.1f}</div>
            <div class="stat">Median Languages per Video: {median_languages:.1f}</div>
        </div>
        
        <div class="section">
            <h2>Top Languages</h2>
            <div class="visualization">
                <iframe src="top_languages.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            <h3>Top 5 Languages by Total Comments:</h3>
            <ul>
                {''.join(f'<li>{lang}: {count:,} comments</li>' for lang, count in top_5_languages.items())}
            </ul>
        </div>
        
        <div class="section">
            <h2>Language Distribution</h2>
            <div class="visualization">
                <iframe src="language_distribution.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        </div>
        
        <div class="section">
            <h2>Language Prevalence in Videos</h2>
            <div class="visualization">
                <iframe src="language_prevalence.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        </div>
        
        <div class="section">
            <h2>Language Combinations</h2>
            <div class="visualization">
                <iframe src="language_combinations.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        </div>
        
        <div class="section">
            <h2>Language Correlations</h2>
            <div class="visualization">
                <iframe src="language_correlation.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            <h3>Notable Language Correlations:</h3>
            <ul>
                {''.join(f'<li>{corr["lang1"]} and {corr["lang2"]}: {corr["correlation"]:.2f}</li>' for corr in notable_correlations)}
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save the HTML report
    with open(os.path.join(output_dir, 'language_analysis_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    # Create output directory
    output_dir = 'result_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    df, total_videos = load_and_prepare_data('detailed_language_analysis.csv')
    
    # Create visualizations
    create_language_distribution_plot(df, output_dir)
    create_top_languages_plot(df, output_dir)
    create_language_correlation_plot(df, output_dir)
    create_language_prevalence_plot(df, output_dir)
    create_language_combination_plot(df, output_dir)
    
    # Generate HTML report
    generate_html_report(df, total_videos, output_dir)
    
    print(f"Analysis complete. Results saved in {output_dir}/")

if __name__ == "__main__":
    main() 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('langid_v2/video_comment_language.csv')

# Calculate total comments across all videos
total_comments = df['total_comments'].sum()
print(f"Total comments across all videos: {total_comments:,}")

# Calculate total undetected languages
total_undetected = df['undetected_languages'].sum()
print(f"Total undetected languages: {total_undetected:,}")
print(f"Percentage of undetected languages: {(total_undetected/total_comments)*100:.2f}%")

# Get language columns (excluding metadata columns)
language_cols = [col for col in df.columns if col not in ['youtube_id', 'total_comments', 'total_languages', 'undetected_languages']]

# Calculate total comments per language
language_totals = df[language_cols].sum().sort_values(ascending=False)

# Get top 10 languages
top_10_languages = language_totals.head(10)
print("\nTop 10 languages by comment count:")
for lang, count in top_10_languages.items():
    percentage = (count/total_comments)*100
    print(f"{lang}: {count:,} comments ({percentage:.2f}%)")

# Create a bar plot of top 10 languages
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_languages.index, y=top_10_languages.values)
plt.title('Top 10 Languages in Video Comments')
plt.xlabel('Language')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_10_languages.png')

# Calculate average number of languages per video
avg_languages = df['total_languages'].mean()
print(f"\nAverage number of languages per video: {avg_languages:.2f}")

# Calculate percentage of videos with each language
video_percentages = (df[language_cols] > 0).mean() * 100
top_10_percentages = video_percentages.sort_values(ascending=False).head(10)
print("\nTop 10 languages by video coverage:")
for lang, percentage in top_10_percentages.items():
    print(f"{lang}: {percentage:.2f}% of videos")

# Save detailed language statistics to CSV
language_stats = pd.DataFrame({
    'total_comments': language_totals,
    'percentage_of_total': (language_totals/total_comments)*100,
    'percentage_of_videos': video_percentages
})
language_stats.to_csv('language_statistics.csv')

# Plot top 30 languages
plt.figure(figsize=(14, 7))
top_30_languages = language_totals.head(30)
sns.barplot(x=top_30_languages.index, y=top_30_languages.values)
plt.title('Top 30 Languages in Video Comments')
plt.xlabel('Language')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_30_languages.png')

# Plot histogram of all languages
plt.figure(figsize=(10, 6))
sns.histplot(language_totals.values, bins=30, kde=True)
plt.title('Histogram of Comment Counts for All Languages')
plt.xlabel('Number of Comments')
plt.ylabel('Number of Languages')
plt.tight_layout()
plt.savefig('language_histogram.png')

# Example: Analyze a specific language (e.g., 'ko')
specific_language = 'ko'
if specific_language in language_totals:
    lang_count = language_totals[specific_language]
    lang_percentage = (lang_count / total_comments) * 100
    lang_video_coverage = video_percentages[specific_language]
    print(f"\nStats for language '{specific_language}':")
    print(f"Total comments: {lang_count:,} ({lang_percentage:.2f}% of all comments)")
    print(f"Present in {lang_video_coverage:.2f}% of videos")
else:
    print(f"\nLanguage '{specific_language}' not found in data.")

# Example: Analyze a specific video by youtube_id
youtube_id = df['youtube_id'].iloc[0]  # Change as needed
video_row = df[df['youtube_id'] == youtube_id]
if not video_row.empty:
    print(f"\nStats for video '{youtube_id}':")
    print(video_row[language_cols + ['total_comments', 'total_languages', 'undetected_languages']].T)
else:
    print(f"\nVideo '{youtube_id}' not found in data.")

# Analyze trends over time if timestamp is available
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    # Example: plot total comments per month
    monthly_comments = df['total_comments'].resample('M').sum()
    plt.figure(figsize=(12, 6))
    monthly_comments.plot()
    plt.title('Total Comments per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig('comments_trend_over_time.png')
else:
    print("\nNo timestamp column found. Skipping trend analysis over time.")

# Analyze undetected languages per video
plt.figure(figsize=(10, 6))
sns.histplot(df['undetected_languages'], bins=30, kde=True)
plt.title('Histogram of Undetected Languages per Video')
plt.xlabel('Number of Undetected Comments')
plt.ylabel('Number of Videos')
plt.tight_layout()
plt.savefig('undetected_languages_histogram.png')

# Print videos with high undetected counts
high_undetected = df[df['undetected_languages'] > df['undetected_languages'].quantile(0.95)]
print(f"\nTop 5% videos by undetected language count:")
print(high_undetected[['youtube_id', 'undetected_languages', 'total_comments']])

# Generate HTML report
html_report = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Language Usage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #2c3e50; }
        table { border-collapse: collapse; width: 80%; margin-bottom: 30px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: right; }
        th { background: #f4f4f4; }
        img { max-width: 700px; margin-bottom: 30px; border: 1px solid #ccc; }
        .section { margin-bottom: 40px; }
    </style>
</head>
<body>
    <h1>Language Usage Report</h1>
    <div class="section">
        <h2>Key Statistics</h2>
        <ul>
            <li><b>Total comments across all videos:</b> {total_comments:,}</li>
            <li><b>Total undetected languages:</b> {total_undetected:,}</li>
            <li><b>Percentage of undetected languages:</b> {undetected_percentage:.2f}%</li>
            <li><b>Average number of languages per video:</b> {avg_languages:.2f}</li>
        </ul>
    </div>
    <div class="section">
        <h2>Top 10 Languages by Comment Count</h2>
        <table>
            <tr><th>Language</th><th>Comments</th><th>Percentage</th></tr>
            {top_10_languages_html}
        </table>
        <img src="../top_10_languages.png" alt="Top 10 Languages">
    </div>
    <div class="section">
        <h2>Top 30 Languages by Comment Count</h2>
        <img src="../top_30_languages.png" alt="Top 30 Languages">
    </div>
    <div class="section">
        <h2>Histogram of All Languages' Comment Counts</h2>
        <img src="../language_histogram.png" alt="Histogram of All Languages">
    </div>
    <div class="section">
        <h2>Top 10 Languages by Video Coverage</h2>
        <table>
            <tr><th>Language</th><th>Coverage (%)</th></tr>
            {top_10_coverage_html}
        </table>
    </div>
    <div class="section">
        <h2>Stats for Language 'ko'</h2>
        <ul>
            <li><b>Total comments:</b> {ko_comments:,} ({ko_percentage:.2f}% of all comments)</li>
            <li><b>Present in:</b> {ko_coverage:.2f}% of videos</li>
        </ul>
    </div>
    <div class="section">
        <h2>Example Video Breakdown (First Video in Dataset)</h2>
        <pre>
Video ID: {example_video_id}
Total comments: {example_video_comments}
Total languages: {example_video_languages}
Undetected languages: {example_video_undetected}
        </pre>
    </div>
    <div class="section">
        <h2>Undetected Languages Analysis</h2>
        <img src="../undetected_languages_histogram.png" alt="Undetected Languages Histogram">
        <h3>Top 5% Videos by Undetected Language Count</h3>
        <pre>
{high_undetected_html}
        </pre>
    </div>
    <div class="section">
        <h2>Trends Over Time</h2>
        <p>No timestamp column found. Skipping trend analysis over time.</p>
    </div>
</body>
</html>
"""

# Prepare HTML content
top_10_languages_html = ""
for lang, count in top_10_languages.items():
    percentage = (count / total_comments) * 100
    top_10_languages_html += f"<tr><td>{lang}</td><td>{count:,}</td><td>{percentage:.2f}%</td></tr>"

top_10_coverage_html = ""
for lang, percentage in top_10_percentages.items():
    top_10_coverage_html += f"<tr><td>{lang}</td><td>{percentage:.2f}</td></tr>"

ko_comments = language_totals.get('ko', 0)
ko_percentage = (ko_comments / total_comments) * 100
ko_coverage = video_percentages.get('ko', 0)

example_video_id = df['youtube_id'].iloc[0]
example_video_comments = df.loc[df['youtube_id'] == example_video_id, 'total_comments'].iloc[0]
example_video_languages = df.loc[df['youtube_id'] == example_video_id, 'total_languages'].iloc[0]
example_video_undetected = df.loc[df['youtube_id'] == example_video_id, 'undetected_languages'].iloc[0]

high_undetected_html = ""
for _, row in high_undetected.iterrows():
    high_undetected_html += f"{row['youtube_id']}: {row['undetected_languages']:,} undetected ({row['total_comments']:,} comments)\n"

# Fill in the HTML template
html_report = html_report.format(
    total_comments=total_comments,
    total_undetected=total_undetected,
    undetected_percentage=(total_undetected / total_comments) * 100,
    avg_languages=avg_languages,
    top_10_languages_html=top_10_languages_html,
    top_10_coverage_html=top_10_coverage_html,
    ko_comments=ko_comments,
    ko_percentage=ko_percentage,
    ko_coverage=ko_coverage,
    example_video_id=example_video_id,
    example_video_comments=example_video_comments,
    example_video_languages=example_video_languages,
    example_video_undetected=example_video_undetected,
    high_undetected_html=high_undetected_html
)

# Save the HTML report
with open('summary_v2/language_usage_report.html', 'w') as f:
    f.write(html_report)

print("HTML report generated at summary_v2/language_usage_report.html") 
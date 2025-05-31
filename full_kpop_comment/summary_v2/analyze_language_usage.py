import csv
import os
import json
from collections import defaultdict
import langid
import re
from datetime import datetime

def load_language_data():
    """Load language data from CSV files"""
    # Load comment language data
    comment_languages = {}
    with open('langid_v2/video_comment_language.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['youtube_id']
            comment_languages[video_id] = {
                'total_comments': int(row['total_comments']),
                'languages': {lang: int(row[lang]) for lang in row.keys() 
                            if lang not in ['youtube_id', 'total_comments', 'total_languages', 'undetected_languages']}
            }
    
    # Load title language data
    title_languages = {}
    with open('langid_v2/video_title_language.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['youtube_id']
            title_languages[video_id] = {
                'languages': {lang: int(row[lang]) for lang in row.keys() 
                            if lang not in ['youtube_id', 'undetected', 'title', 'total_languages'] and row[lang].isdigit()}
            }
    
    # Load description language data
    desc_languages = {}
    with open('langid_v2/video_description_language.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['youtube_id']
            desc_languages[video_id] = {
                'languages': {lang: int(row[lang]) for lang in row.keys() 
                            if lang not in ['youtube_id', 'undetected', 'description', 'total_languages'] and row[lang].isdigit()}
            }
    
    return comment_languages, title_languages, desc_languages

def calculate_language_statistics(comment_languages, title_languages, desc_languages):
    """Calculate statistics for language usage"""
    stats = {
        'comments': defaultdict(int),
        'titles': defaultdict(int),
        'descriptions': defaultdict(int),
        'top_videos': []
    }
    
    # Process comment languages
    for video_id, video_data in comment_languages.items():
        for lang, count in video_data['languages'].items():
            stats['comments'][lang] += count
    
    # Process title languages
    for video_id, video_data in title_languages.items():
        for lang, count in video_data['languages'].items():
            if count > 0:  # Only count if language is present
                stats['titles'][lang] += 1
    
    # Process description languages
    for video_id, video_data in desc_languages.items():
        for lang, count in video_data['languages'].items():
            if count > 0:  # Only count if language is present
                stats['descriptions'][lang] += 1
    
    # Calculate total unique languages per video
    for video_id in set(comment_languages.keys()) | set(title_languages.keys()) | set(desc_languages.keys()):
        comment_langs = set(comment_languages.get(video_id, {}).get('languages', {}).keys())
        title_langs = set(title_languages.get(video_id, {}).get('languages', {}).keys())
        desc_langs = set(desc_languages.get(video_id, {}).get('languages', {}).keys())
        
        total_langs = len(comment_langs | title_langs | desc_langs)
        stats['top_videos'].append({
            'video_id': video_id,
            'total_languages': total_langs,
            'comment_languages': len(comment_langs),
            'title_languages': len(title_langs),
            'description_languages': len(desc_langs)
        })
    
    # Sort videos by total languages
    stats['top_videos'].sort(key=lambda x: x['total_languages'], reverse=True)
    
    return stats

def generate_html_report(stats):
    """Generate HTML report with language usage statistics"""
    total_comments = sum(stats['comments'].values())
    total_videos = sum(stats['titles'].values())  # total videos with at least one title language detected
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"UTF-8\">
        <title>Language Usage Analysis in K-pop Videos</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin: 30px 0; }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Language Usage Analysis in K-pop Videos</h1>
        <p class=\"timestamp\">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class=\"section\">
            <h2>Top 10 Videos with Most Languages Used</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Video ID</th>
                    <th>Total Languages</th>
                    <th>Comment Languages</th>
                    <th>Title Languages</th>
                    <th>Description Languages</th>
                </tr>
    """
    # Add top videos statistics
    for i, video in enumerate(stats['top_videos'][:10], 1):
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{video['video_id']}</td>
                    <td>{video['total_languages']}</td>
                    <td>{video['comment_languages']}</td>
                    <td>{video['title_languages']}</td>
                    <td>{video['description_languages']}</td>
                </tr>"""
    
    html_content += """
            </table>
        </div>
        
        <div class=\"section\">
            <h2>Top 20 Languages in Comments</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Language</th>
                    <th>Comment Count</th>
                    <th>Percentage</th>
                </tr>
    """
    # Add comment language statistics
    for i, (lang, count) in enumerate(sorted(stats['comments'].items(), key=lambda x: x[1], reverse=True)[:20], 1):
        percentage = (count / total_comments) * 100 if total_comments else 0
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{lang}</td>
                    <td>{count:,}</td>
                    <td>{percentage:.1f}%</td>
                </tr>"""
    html_content += """
            </table>
        </div>
        
        <div class=\"section\">
            <h2>Top 20 Languages in Video Titles</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Language</th>
                    <th>Video Count</th>
                    <th>Percentage</th>
                </tr>
    """
    for i, (lang, count) in enumerate(sorted(stats['titles'].items(), key=lambda x: x[1], reverse=True)[:20], 1):
        percentage = (count / total_videos) * 100 if total_videos else 0
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{lang}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>"""
    html_content += """
            </table>
        </div>
        
        <div class=\"section\">
            <h2>Top 20 Languages in Video Descriptions</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Language</th>
                    <th>Video Count</th>
                    <th>Percentage</th>
                </tr>
    """
    for i, (lang, count) in enumerate(sorted(stats['descriptions'].items(), key=lambda x: x[1], reverse=True)[:20], 1):
        percentage = (count / total_videos) * 100 if total_videos else 0
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{lang}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>"""
    html_content += f"""
            </table>
        </div>
        
        <div class=\"section\">
            <h2>Summary Statistics</h2>
            <ul>
                <li>Total number of videos analyzed: {total_videos:,}</li>
                <li>Total number of comments analyzed: {total_comments:,}</li>
                <li>Number of unique languages in comments: {len(stats['comments'])}</li>
                <li>Number of unique languages in titles: {len(stats['titles'])}</li>
                <li>Number of unique languages in descriptions: {len(stats['descriptions'])}</li>
            </ul>
        </div>
    </body>
    </html>
    """
    # Save the HTML report
    with open('summary_v2/language_usage_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    # Load language data
    print("Loading language data...")
    comment_languages, title_languages, desc_languages = load_language_data()
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_language_statistics(comment_languages, title_languages, desc_languages)
    
    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(stats)
    
    print("Analysis complete. Report saved to summary_v2/language_usage_report.html")

if __name__ == "__main__":
    main() 
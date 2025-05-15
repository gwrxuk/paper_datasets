import pandas as pd
import os

# If needed, install required packages:
# !pip install jieba textblob

# Import libraries for text processing and analysis
import jieba
from textblob import TextBlob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Set up output directory
output_dir = 'comparison_v2'
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('gpt_gemini_responses.csv')

# Drop rows where either model's response is missing
df = df.dropna(subset=['gpt', 'gemini'])
# Perform Chinese word segmentation on both GPT-4 and Gemini responses
df['GPT4_segmented'] = df['gpt'].apply(lambda text: ' '.join(jieba.cut(str(text))))
df['Gemini_segmented'] = df['gemini'].apply(lambda text: ' '.join(jieba.cut(str(text))))

# Save the original and segmented responses to a CSV file
segmented_csv_path = os.path.join(output_dir, 'segmented_responses.csv')
df.to_csv(segmented_csv_path, columns=['gpt', 'GPT4_segmented', 'gemini', 'Gemini_segmented'], index=False)
print(f"Segmented responses saved to {segmented_csv_path}")

# Calculate word counts for each segmented response
df['GPT4_word_count'] = df['GPT4_segmented'].str.split().apply(len)
df['Gemini_word_count'] = df['Gemini_segmented'].str.split().apply(len)

# Compute summary statistics for word counts
gpt_mean = df['GPT4_word_count'].mean()
gpt_median = df['GPT4_word_count'].median()
gpt_std = df['GPT4_word_count'].std()
gem_mean = df['Gemini_word_count'].mean()
gem_median = df['Gemini_word_count'].median()
gem_std = df['Gemini_word_count'].std()

# Save mean, median, std to CSV
wordcount_stats = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'StdDev'],
    'GPT4_word_count': [gpt_mean, gpt_median, gpt_std],
    'Gemini_word_count': [gem_mean, gem_median, gem_std]
})
wordcount_csv_path = os.path.join(output_dir, 'word_count_stats.csv')
wordcount_stats.to_csv(wordcount_csv_path, index=False)
print(f"Word count statistics saved to {wordcount_csv_path}")

# Plot a grouped bar chart for word count statistics
metrics = ['Mean', 'Median', 'StdDev']
gpt_vals = [gpt_mean, gpt_median, gpt_std]
gem_vals = [gem_mean, gem_median, gem_std]
x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, gpt_vals, width, label='GPT-4')
ax.bar(x + width/2, gem_vals, width, label='Gemini')
# Style adjustments for APA-like appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Word Count')
ax.set_title('Word Count Comparison (GPT-4 vs Gemini)')
ax.legend()
# Save plot to file
wordcount_plot_path = os.path.join(output_dir, 'word_count_comparison.png')
plt.tight_layout()
plt.savefig(wordcount_plot_path)
plt.close(fig)
print(f"Word count bar chart saved to {wordcount_plot_path}")

# Compute sentiment polarity for each response using TextBlob
df['GPT4_polarity'] = df['GPT4_segmented'].apply(lambda text: TextBlob(text).sentiment.polarity)
df['Gemini_polarity'] = df['Gemini_segmented'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Classify polarity into sentiment categories
def classify_sentiment(p):
    if p > 0: 
        return 'Positive'
    elif p < 0: 
        return 'Negative'
    else: 
        return 'Neutral'

df['GPT4_sentiment'] = df['GPT4_polarity'].apply(classify_sentiment)
df['Gemini_sentiment'] = df['Gemini_polarity'].apply(classify_sentiment)

# Save sentiment scores and labels to CSV
sentiment_csv_path = os.path.join(output_dir, 'sentiment_analysis.csv')
df.to_csv(sentiment_csv_path, columns=['GPT4_polarity','GPT4_sentiment','Gemini_polarity','Gemini_sentiment'], index=False)
print(f"Sentiment analysis results saved to {sentiment_csv_path}")

# Plot a stacked bar chart for sentiment distribution
sentiment_counts_gpt = df['GPT4_sentiment'].value_counts()
sentiment_counts_gem = df['Gemini_sentiment'].value_counts()
categories = ['Positive', 'Neutral', 'Negative']
gpt_counts = [sentiment_counts_gpt.get(cat, 0) for cat in categories]
gem_counts = [sentiment_counts_gem.get(cat, 0) for cat in categories]

fig, ax = plt.subplots()
# GPT-4 stacked bar
ax.bar('GPT-4', gpt_counts[0], label='Positive', color='#2ca02c')
ax.bar('GPT-4', gpt_counts[1], bottom=gpt_counts[0], label='Neutral', color='#c7c7c7')
ax.bar('GPT-4', gpt_counts[2], bottom=gpt_counts[0]+gpt_counts[1], label='Negative', color='#d62728')
# Gemini stacked bar
ax.bar('Gemini', gem_counts[0], color='#2ca02c')
ax.bar('Gemini', gem_counts[1], bottom=gem_counts[0], color='#c7c7c7')
ax.bar('Gemini', gem_counts[2], bottom=gem_counts[0]+gem_counts[1], color='#d62728')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Number of Responses')
ax.set_title('Sentiment Distribution (GPT-4 vs Gemini)')
ax.legend(title='Sentiment')
# Save plot
sentiment_plot_path = os.path.join(output_dir, 'sentiment_distribution.png')
plt.tight_layout()
plt.savefig(sentiment_plot_path)
plt.close(fig)
print(f"Sentiment distribution chart saved to {sentiment_plot_path}")

# Calculate type-token ratio (TTR) for each response
df['GPT4_TTR'] = df['GPT4_segmented'].str.split().apply(lambda tokens: len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0)
df['Gemini_TTR'] = df['Gemini_segmented'].str.split().apply(lambda tokens: len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0)

# Compute average TTR for each model
avg_ttr_gpt = df['GPT4_TTR'].mean()
avg_ttr_gem = df['Gemini_TTR'].mean()

# Save TTR of each response and the average to CSV
lexical_csv_path = os.path.join(output_dir, 'lexical_diversity.csv')
df.to_csv(lexical_csv_path, columns=['GPT4_TTR', 'Gemini_TTR'], index=False)
# Append a summary row with the averages
with open(lexical_csv_path, 'a') as f:
    f.write(f'Average,{avg_ttr_gpt:.4f},{avg_ttr_gem:.4f}\n')
print(f"Lexical diversity (TTR) results saved to {lexical_csv_path}")

# Plot a bar chart for average lexical diversity
fig, ax = plt.subplots()
ax.bar(['GPT-4', 'Gemini'], [avg_ttr_gpt, avg_ttr_gem], color=['#1f77b4', '#ff7f0e'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Type-Token Ratio')
ax.set_title('Average Lexical Diversity (TTR)')
# Annotate bars with the value
for i, v in enumerate([avg_ttr_gpt, avg_ttr_gem]):
    ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontstyle='italic')
lexical_plot_path = os.path.join(output_dir, 'lexical_diversity.png')
plt.tight_layout()
plt.savefig(lexical_plot_path)
plt.close(fig)
print(f"Lexical diversity bar chart saved to {lexical_plot_path}")
# Define Chinese and English stopwords
chinese_stopwords = set([
    '的','了','和','是','在','也','有','就','不','而','及','與','但','如果',
    '因此','此外','另外','然而','而且','不過','所以','當然','例如',
    '這','那','這個','那個','以及','嗎','啊','呢','吧','嗯',
    '我','你','他','她','它','我們','你們','他們','她們',
    '請問','請','謝謝','感謝','抱歉','對不起',
    '說','表示','認為','覺得','知道','看到'
])
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
english_stopwords = ENGLISH_STOP_WORDS

# Flatten all tokens from each model's responses into a list
gpt_tokens_all = [tok for text in df['GPT4_segmented'] for tok in text.split()]
gem_tokens_all = [tok for text in df['Gemini_segmented'] for tok in text.split()]

# Filter out stopwords and non-informative tokens
def is_informative(token):
    t = token.lower()
    if t.isdigit():
        return False
    if t in english_stopwords or t in chinese_stopwords:
        return False
    if len(t) == 1:  # filter single-character tokens (often not meaningful)
        return False
    return True

gpt_tokens_filtered = [tok for tok in gpt_tokens_all if is_informative(tok)]
gem_tokens_filtered = [tok for tok in gem_tokens_all if is_informative(tok)]

# Get top 10 most common tokens for each model
gpt_freq = Counter(gpt_tokens_filtered)
gem_freq = Counter(gem_tokens_filtered)
top10_gpt = gpt_freq.most_common(10)
top10_gem = gem_freq.most_common(10)

# Save the top 10 keywords and their frequencies for each model to CSV
keywords_csv_path = os.path.join(output_dir, 'top10_keywords.csv')
with open(keywords_csv_path, 'w', encoding='utf-8') as f:
    f.write('GPT4_keyword,GPT4_count,Gemini_keyword,Gemini_count\n')
    for i in range(10):
        gpt_word, gpt_count = top10_gpt[i] if i < len(top10_gpt) else ("", "")
        gem_word, gem_count = top10_gem[i] if i < len(top10_gem) else ("", "")
        f.write(f"{gpt_word},{gpt_count},{gem_word},{gem_count}\n")
print(f"Top 10 keywords for each model saved to {keywords_csv_path}")
# Define keywords indicating humor or formal tone
humor_keywords = ["哈哈", "笑", "lol", "LOL", "XD", "xd", "😂", "🤣", "搞笑", "有趣"]
formal_keywords = ["尊敬", "貴", "先生", "小姐", "您好", "尊敬的", "此致", "敬禮", "Sir", "Madam", "Dear", "Sincerely"]

# Function to detect tone based on presence of keywords
def detect_tone(text):
    text = str(text)
    for kw in humor_keywords:
        if kw in text:
            return "Humor"
    for kw in formal_keywords:
        if kw in text:
            return "Formal"
    return "Neutral"

# Classify each response for tone
df['GPT4_tone'] = df['gpt'].apply(detect_tone)
df['Gemini_tone'] = df['gemini'].apply(detect_tone)

# Save tone results to CSV
tone_csv_path = os.path.join(output_dir, 'tone_detection.csv')
df.to_csv(tone_csv_path, columns=['GPT4_tone', 'Gemini_tone'], index=False)
print(f"Tone detection results saved to {tone_csv_path}")

# Count the tone categories for each model
tone_counts_gpt = df['GPT4_tone'].value_counts()
tone_counts_gem = df['Gemini_tone'].value_counts()
tone_categories = ['Humor', 'Formal', 'Neutral']
gpt_tone_vals = [tone_counts_gpt.get(cat, 0) for cat in tone_categories]
gem_tone_vals = [tone_counts_gem.get(cat, 0) for cat in tone_categories]

# Plot a grouped bar chart for tone distribution
x = np.arange(len(tone_categories))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, gpt_tone_vals, width, label='GPT-4')
ax.bar(x + width/2, gem_tone_vals, width, label='Gemini')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(x)
ax.set_xticklabels(tone_categories)
ax.set_ylabel('Number of Responses')
ax.set_title('Tone Classification (GPT-4 vs Gemini)')
ax.legend(title='Tone')
tone_plot_path = os.path.join(output_dir, 'tone_distribution.png')
plt.tight_layout()
plt.savefig(tone_plot_path)
plt.close(fig)
print(f"Tone distribution bar chart saved to {tone_plot_path}")

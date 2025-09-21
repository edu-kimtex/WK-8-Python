import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("data/metadata.csv")

# =============================
# Part 1: Data Loading & Exploration
# =============================
print("Shape:", df.shape)
print(df.info())
print(df.head())

# Missing values
print("Missing values per column:\n", df.isnull().sum())

# =============================
# Part 2: Data Cleaning
# =============================
# Convert date column
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["year"] = df["publish_time"].dt.year

# Create new feature: abstract word count
df["abstract_word_count"] = df["abstract"].fillna("").apply(lambda x: len(x.split()))

# Drop rows with no title or publish_time
df = df.dropna(subset=["title", "publish_time"])

# =============================
# Part 3: Analysis
# =============================
# Papers per year
year_counts = df["year"].value_counts().sort_index()

# Top journals
top_journals = df["journal"].value_counts().head(10)

# Word frequency in titles (word cloud)
text = " ".join(title for title in df["title"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# =============================
# Part 3: Visualizations
# =============================
# Publications over time
plt.figure(figsize=(8,5))
sns.barplot(x=year_counts.index, y=year_counts.values, color="skyblue")
plt.title("Publications by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.savefig("plots/publications_by_year.png")
plt.show()

# Top journals
plt.figure(figsize=(10,6))
sns.barplot(y=top_journals.index, x=top_journals.values, palette="viridis")
plt.title("Top 10 Journals Publishing COVID-19 Papers")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.savefig("plots/top_journals.png")
plt.show()

# Word cloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Paper Titles")
plt.savefig("plots/wordcloud_titles.png")
plt.show()

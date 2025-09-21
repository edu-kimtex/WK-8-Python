import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/metadata.csv")
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["year"] = df["publish_time"].dt.year
    return df

df = load_data()

# =============================
# Streamlit Layout
# =============================
st.title("📊 CORD-19 Data Explorer")
st.write("Exploring COVID-19 research papers metadata")

# Sidebar filter
years = st.slider("Select year range", 2015, 2023, (2019, 2021))
filtered = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]

st.subheader("Sample Data")
st.dataframe(filtered.head())

# Publications by year
st.subheader("Publications by Year")
year_counts = filtered["year"].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=year_counts.index, y=year_counts.values, ax=ax, color="skyblue")
plt.title("Publications by Year")
st.pyplot(fig)

# Top journals
st.subheader("Top Journals")
top_journals = filtered["journal"].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(y=top_journals.index, x=top_journals.values, ax=ax, palette="viridis")
plt.title("Top 10 Journals")
st.pyplot(fig)

# Word cloud for titles
st.subheader("Word Cloud of Titles")
text = " ".join(title for title in filtered["title"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

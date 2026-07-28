# 🎬 Telugu Movie Recommender

Ever wanted to find movies like the ones you love?  
This is a content-based recommendation system that helps you discover **similar Telugu movies** by analyzing their plot and genre.

Built with simple but powerful tools like **TF-IDF** and **cosine similarity**, it's a fun project that combines machine learning with movie love 🎥✨

---

## 🧠 What It Does

You enter a Telugu movie title.  
It looks at the **movie’s overview and genre**, compares it with all other movies in the dataset, and suggests the **top 5 most similar ones**.

It’s a content-based system — no user ratings or fancy deep learning — just good old NLP!

---

## 🛠️ How It Works

- **TF-IDF Vectorizer** turns text into numerical data
- **Cosine Similarity** measures how close the movies are in meaning
- You get a list of recommendations with **genre** and **release year**

---

## 🗂️ Project Files

| File | What it does |
|------|---------------|
| `telugu movie recommender.ipynb` | Main logic for cleaning, vectorizing, and recommending |
| `TeluguMovies_dataset.csv` | The dataset with movie titles, overviews, genres, and years |


 Example Usage
 
from recommender import recommend
recommend("U Turn")

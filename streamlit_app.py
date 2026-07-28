import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Page config ----
st.set_page_config(page_title="Telugu Movie Recommender", page_icon="🎬", layout="wide")

YEAR_DECAY = 12.0          # years; controls how quickly year-distance reduces score
DEFAULT_PLOT_WEIGHT = 0.50   # plot similarity weight
DEFAULT_GENRE_WEIGHT = 0.35  # genre overlap weight
DEFAULT_YEAR_WEIGHT = 0.15   # year proximity weight


# ---- Load & prepare data (cached) ----
@st.cache_data
def load_data():
    df = pd.read_csv("TeluguMovies_dataset.csv")

    # Clean whitespace and missing values
    for col in ["Movie", "Overview", "Genre", "Certificate"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    # Drop rows with no overview & reset index for strict positional matching
    df = df[df["Overview"] != ""].reset_index(drop=True)

    # Standardize column types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if "No.of.Ratings" in df.columns:
        df["No.of.Ratings"] = pd.to_numeric(df["No.of.Ratings"], errors="coerce")

    # --- 1. Plot Similarity (Sentence Transformers with TF-IDF fallback) ---
    plot_sim = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df["Overview"].tolist(), show_progress_bar=False)
        plot_sim = cosine_similarity(embeddings, embeddings)
    except Exception:
        # Fallback: TF-IDF with min_df=1 to preserve unique movie terms
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.85)
        matrix = tfidf.fit_transform(df["Overview"])
        plot_sim = cosine_similarity(matrix, matrix)

    # --- 2. Genre Similarity (Exact token matching) ---
    # Custom tokenizer for comma-separated genre lists
    def parse_genres(g_str):
        return [g.strip().lower() for g in g_str.split(",") if g.strip()]

    cv = CountVectorizer(tokenizer=parse_genres, lowercase=False)
    genre_matrix = cv.fit_transform(df["Genre"])
    # Normalize by cosine similarity for jaccard-like overlap
    genre_sim = cosine_similarity(genre_matrix, genre_matrix)

    return df, plot_sim, genre_sim


df, plot_sim, genre_sim = load_data()


# ---- Helper to check shared genres ----
def shares_genre(genres1_str, genres2_str):
    g1 = set(g.strip().lower() for g in genres1_str.split(",") if g.strip())
    g2 = set(g.strip().lower() for g in genres2_str.split(",") if g.strip())
    return len(g1.intersection(g2)) > 0


# ---- Recommendation logic ----
def recommend(title, top_n=5, strict_genre=False, plot_w=0.50, genre_w=0.35, year_w=0.15):
    matches = df[df["Movie"].str.lower() == title.lower()]
    if matches.empty:
        return None

    idx = matches.index[0]
    target_year = df.loc[idx, "Year"]
    target_genres = df.loc[idx, "Genre"]

    # Hybrid content + genre scoring
    scores = plot_w * plot_sim[idx] + genre_w * genre_sim[idx]

    # Incorporate year proximity factor
    if pd.notna(target_year):
        years = df["Year"].values
        year_diff = np.abs(years - target_year)
        year_diff = np.nan_to_num(year_diff, nan=YEAR_DECAY * 3)
        year_factor = np.exp(-year_diff / YEAR_DECAY)
        scores += year_w * year_factor
    else:
        # Renormalize weights when target movie year is missing
        total_w = plot_w + genre_w
        if total_w > 0:
            scores = scores / total_w

    # Create score pairs
    sim_scores = list(enumerate(scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude target movie itself
    filtered_results = []
    for item_idx, score in sim_scores:
        if item_idx == idx:
            continue
        
        # Apply strict genre filter if enabled
        if strict_genre:
            cand_genres = df.loc[item_idx, "Genre"]
            if not shares_genre(target_genres, cand_genres):
                continue
                
        filtered_results.append((item_idx, score))
        if len(filtered_results) >= top_n:
            break

    if not filtered_results:
        return pd.DataFrame()

    res_indices = [r[0] for r in filtered_results]
    res_scores = [r[1] for r in filtered_results]

    result_df = df.iloc[res_indices].copy()
    # Normalize score to percentage (capped at 100%)
    result_df["Match_Score"] = [min(round(s * 100, 1), 99.9) for s in res_scores]
    return result_df


# ---- UI Layout ----
st.title("🎬 Enhanced Telugu Movie Recommender")
st.write(
    f"Content-based hybrid recommender covering **{len(df):,}** Telugu movies with plot embeddings, "
    "exact genre matching, and release year proximity."
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Recommendation Settings")
    
    strict_genre_toggle = st.checkbox(
        "Strict Genre Matching", 
        value=False,
        help="Only recommend movies that share at least one genre with the selected movie."
    )

    top_n_slider = st.slider("Number of Recommendations:", min_value=3, max_value=15, value=6)

    st.subheader("Weight Customization")
    plot_weight_val = st.slider("Plot Weight", 0.0, 1.0, DEFAULT_PLOT_WEIGHT, 0.05)
    genre_weight_val = st.slider("Genre Weight", 0.0, 1.0, DEFAULT_GENRE_WEIGHT, 0.05)
    year_weight_val = st.slider("Year Weight", 0.0, 1.0, DEFAULT_YEAR_WEIGHT, 0.05)

    # Normalize weights so they sum to 1.0
    weight_sum = plot_weight_val + genre_weight_val + year_weight_val
    if weight_sum > 0:
        norm_plot_w = plot_weight_val / weight_sum
        norm_genre_w = genre_weight_val / weight_sum
        norm_year_w = year_weight_val / weight_sum
    else:
        norm_plot_w, norm_genre_w, norm_year_w = 0.5, 0.35, 0.15

# Main input controls
movie_list = sorted(df["Movie"].dropna().unique())

col1, col2 = st.columns([3, 1])
with col1:
    selected_movie = st.selectbox("Pick a movie you liked:", movie_list)
with col2:
    st.write(" ")
    st.write(" ")
    get_rec_btn = st.button("✨ Get Recommendations", type="primary", use_container_width=True)

if get_rec_btn or selected_movie:
    # Get details of selected movie
    selected_info = df[df["Movie"].str.lower() == selected_movie.lower()].iloc[0]
    
    st.markdown("---")
    st.subheader("Selected Movie Details")
    s_col1, s_col2, s_col3 = st.columns([2, 2, 1])
    with s_col1:
        st.markdown(f"**Movie:** {selected_info['Movie']}")
        year_str = int(selected_info['Year']) if pd.notna(selected_info['Year']) else "N/A"
        st.markdown(f"**Year:** {year_str}")
    with s_col2:
        st.markdown(f"**Genre:** {selected_info['Genre']}")
        cert_str = selected_info['Certificate'] if selected_info['Certificate'] else "Unrated"
        st.markdown(f"**Certificate:** {cert_str}")
    with s_col3:
        rating_str = f"{selected_info['Rating']:.1f}" if pd.notna(selected_info['Rating']) else "N/A"
        st.metric("Rating", f"⭐ {rating_str}")

    st.caption(f"**Overview:** {selected_info['Overview']}")
    st.markdown("---")

    # Run recommendation
    results = recommend(
        selected_movie, 
        top_n=top_n_slider, 
        strict_genre=strict_genre_toggle,
        plot_w=norm_plot_w,
        genre_w=norm_genre_w,
        year_w=norm_year_w
    )

    if results is None or results.empty:
        st.warning("No suitable recommendations found. Try turning off 'Strict Genre Matching' or adjusting settings.")
    else:
        st.subheader(f"Recommended Movies similar to '{selected_movie}':")
        
        # Display cards in two columns
        card_cols = st.columns(2)
        for idx_pos, (_, row) in enumerate(results.iterrows()):
            col_target = card_cols[idx_pos % 2]
            with col_target:
                year_disp = int(row["Year"]) if pd.notna(row["Year"]) else "N/A"
                rating_disp = f"{row['Rating']:.1f}" if pd.notna(row["Rating"]) else "N/A"
                match_score = row["Match_Score"]
                
                with st.container(border=True):
                    head_col1, head_col2 = st.columns([3, 1])
                    with head_col1:
                        st.markdown(f"### {row['Movie']} ({year_disp})")
                    with head_col2:
                        st.markdown(f"**🎯 {match_score}% Match**")

                    st.markdown(f"**Genre:** `{row['Genre']}` | **Rating:** ⭐ `{rating_disp}`")
                    
                    if "No.of.Ratings" in row and pd.notna(row["No.of.Ratings"]):
                        st.caption(f"👥 Votes: {int(row['No.of.Ratings']):,}")
                        
                    st.write(row["Overview"])

st.caption("Built with Python, Scikit-learn, Pandas & Streamlit")

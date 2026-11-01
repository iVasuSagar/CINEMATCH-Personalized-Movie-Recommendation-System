import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_demo_data():
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return movies, ratings

def recommend_for_user(user_id, movies, ratings, n=5):
    """User-based collaborative filtering with a popularity fallback."""
    pivot = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    if user_id not in pivot.index:
        return pd.DataFrame(), "This user has no ratings in the selected data."

    # Similarity is computed from co-rated items; missing ratings are filled with 0
    # for this simple classroom demo.
    similarity = cosine_similarity(pivot.fillna(0))
    sim_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)
    user_sim = sim_df[user_id].drop(index=user_id, errors="ignore").sort_values(ascending=False)

    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    candidates = movies.loc[~movies["movieId"].isin(seen)].copy()

    scores = []
    for movie_id in candidates["movieId"]:
        relevant = ratings[(ratings["movieId"] == movie_id) & (ratings["userId"].isin(user_sim.index))]
        if relevant.empty:
            scores.append((movie_id, np.nan, 0))
            continue
        relevant = relevant.copy()
        relevant["similarity"] = relevant["userId"].map(user_sim).clip(lower=0)
        denom = relevant["similarity"].sum()
        score = (relevant["rating"] * relevant["similarity"]).sum() / denom if denom > 0 else np.nan
        scores.append((movie_id, score, len(relevant)))

    score_df = pd.DataFrame(scores, columns=["movieId", "predicted_rating", "neighbor_count"])
    candidates = candidates.merge(score_df, on="movieId", how="left")
    popularity = ratings.groupby("movieId").agg(average_rating=("rating", "mean"), rating_count=("rating", "count")).reset_index()
    candidates = candidates.merge(popularity, on="movieId", how="left")
    candidates["predicted_rating"] = candidates["predicted_rating"].fillna(candidates["average_rating"])
    candidates["rating_count"] = candidates["rating_count"].fillna(0).astype(int)
    candidates = candidates.sort_values(["predicted_rating", "rating_count"], ascending=False).head(n)
    return candidates, "Recommendations rank unseen movies using ratings from similar users; popular ratings are used when similarity data is insufficient."

st.title("🎬 CineMatch")
st.subheader("Personalized Movie Recommendation System")
st.caption("Academic demonstration • User-based collaborative filtering • Sample data is synthetic")

with st.sidebar:
    st.header("Demo controls")
    st.write("Choose a demo user and generate personalized recommendations.")
    movies, ratings = load_demo_data()
    user_ids = sorted(ratings["userId"].unique().tolist())
    selected_user = st.selectbox("Select user", user_ids, format_func=lambda x: f"Demo User {x}")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    st.divider()
    st.markdown("**Data source**")
    st.caption("Bundled synthetic ratings for a reliable offline demo.")
    uploaded = st.file_uploader("Optional: upload a MovieLens ratings.csv", type=["csv"])

if uploaded is not None:
    try:
        custom_ratings = pd.read_csv(uploaded)
        required = {"userId", "movieId", "rating"}
        if not required.issubset(custom_ratings.columns):
            st.error("CSV must contain columns: userId, movieId, rating.")
        else:
            custom_ratings = custom_ratings[["userId", "movieId", "rating"]].copy()
            custom_ratings["userId"] = pd.to_numeric(custom_ratings["userId"], errors="coerce")
            custom_ratings["movieId"] = pd.to_numeric(custom_ratings["movieId"], errors="coerce")
            custom_ratings["rating"] = pd.to_numeric(custom_ratings["rating"], errors="coerce")
            custom_ratings = custom_ratings.dropna()
            if len(custom_ratings):
                ratings = custom_ratings
                user_ids = sorted(ratings["userId"].unique().tolist())
                selected_user = st.sidebar.selectbox("Uploaded data user", user_ids, key="uploaded_user")
                st.info("Using uploaded ratings. Movie titles are matched against the bundled demo movie list; unknown movie IDs may not have titles.")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

# Basic validation and overview
ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies in catalog", f"{movies['movieId'].nunique():,}")
c2.metric("Ratings available", f"{len(ratings):,}")
c3.metric("Users", f"{ratings['userId'].nunique():,}")
c4.metric("Average rating", f"{ratings['rating'].mean():.2f}/5")

st.divider()
left, right = st.columns([1.2, 1])
with left:
    st.markdown("### Your rating history")
    history = ratings[ratings["userId"] == selected_user].merge(movies, on="movieId", how="left")
    if history.empty:
        st.warning("No ratings found for this user.")
    else:
        history = history.sort_values("rating", ascending=False)
        st.dataframe(history[["title", "genres", "rating"]].rename(columns={"title":"Movie", "genres":"Genres", "rating":"Your rating"}), use_container_width=True, hide_index=True)
with right:
    st.markdown("### How it works")
    st.markdown("""
    1. Represent each user by their movie ratings.
    2. Compare users with **cosine similarity**.
    3. Find movies rated by similar users that the selected user has not rated.
    4. Estimate a score and rank the unseen movies.
    """)
    st.info("This is a classroom-scale recommender, not a production streaming platform.")

st.divider()
st.markdown("### ✨ Recommended for you")
if st.button("Generate recommendations", type="primary", use_container_width=True):
    recs, explanation = recommend_for_user(selected_user, movies, ratings, top_n)
    if recs.empty:
        st.warning(explanation)
    else:
        st.success(explanation)
        cols = st.columns(min(5, len(recs)))
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"#### {row['title']}")
                st.caption(row.get("genres", "Genre unavailable"))
                score = row.get("predicted_rating", np.nan)
                if pd.notna(score):
                    st.metric("Estimated preference", f"{min(5, max(0, score)):.2f}/5")
                st.caption(f"Based on {int(row.get('rating_count', 0))} sample rating(s)")
        st.dataframe(recs[["title", "genres", "predicted_rating", "rating_count"]].rename(columns={
            "title":"Movie", "genres":"Genres", "predicted_rating":"Estimated rating", "rating_count":"Ratings in data"
        }), use_container_width=True, hide_index=True)
else:
    st.write("Click **Generate recommendations** to see the selected user's personalized results.")

with st.expander("Model notes and limitations"):
    st.markdown("""
    - **Method:** user-based collaborative filtering.
    - **Similarity:** cosine similarity on user rating vectors.
    - **Cold start:** a new user with no ratings cannot receive truly personalized results.
    - **Sparse data:** small sample datasets produce weak estimates; popularity is used as a fallback.
    - **Bias:** the recommendations reflect the sample ratings and can reproduce their biases.
    - **Data:** bundled data is synthetic and intended only for demonstration. It is not real user data.
    - **Production improvements:** use a larger MovieLens dataset, account for rating scale and missingness more carefully, evaluate with train/test splits, and add authentication/database storage if needed.
    """)
st.caption("Built for academic demonstration. No real user information is collected or stored.")

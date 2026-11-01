# CineMatch — Personalized Movie Recommendation System

## What this project does
CineMatch is an academic demonstration of a movie recommender. It uses **user-based collaborative filtering**: users with similar rating patterns help estimate which unseen movies another user may like.

**Important:** The bundled `data/ratings.csv` is synthetic demo data, not real MovieLens data. The interface supports uploading a MovieLens-style `ratings.csv` with `userId`, `movieId`, and `rating` columns. Movie titles are matched to the bundled catalog unless you extend `movies.csv`.

## Run it (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt or terminal in the extracted `movie_recommendation_project` folder.
4. Create a virtual environment (recommended):
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL printed in the terminal, usually `http://localhost:8501`.

If `py` does not work, try `python` instead.

## Demo flow
1. Choose a Demo User in the sidebar.
2. Point out the rating history and overview metrics.
3. Click **Generate recommendations**.
4. Explain that the app recommends movies the user has not rated, using the ratings of similar users.
5. Open **Model notes and limitations** and mention cold start, sparse data, and synthetic sample data honestly.

## Algorithm in plain English
1. Build a user–movie matrix: rows are users, columns are movies, cells contain ratings.
2. Compute cosine similarity between each pair of user rating vectors.
3. For each movie the target user has not rated, use ratings from similar users to calculate a similarity-weighted estimate.
4. Rank movies by estimated preference and show the top N.

For target user u and movie m, the simplified estimate is:

`predicted_score(u,m) = sum(similarity(u,v) * rating(v,m)) / sum(similarity(u,v))`

The implementation uses non-negative similarities for weighting. If there is not enough neighbor data, it falls back to the average rating for that movie.

## Files
- `app.py` — Streamlit interface and recommendation logic.
- `data/movies.csv` — demo movie catalog.
- `data/ratings.csv` — synthetic ratings.
- `requirements.txt` — Python dependencies.
- `docs/VIVA_NOTES.md` — study guide and viva questions.
- `docs/DEMO_SCRIPT.md` — short presentation script.
- `CineMatch_Presentation.pptx` — slides for the demonstration.

## Limitations and next steps
- Small synthetic data is only suitable for demonstrating the workflow.
- Filling missing ratings with zero for similarity is a simplified classroom approach.
- A new user with no ratings suffers from the cold-start problem.
- A larger real dataset, train/test evaluation (RMSE/MAE or ranking metrics), improved missing-data handling, user accounts, persistent storage, and a deployed server would improve the project.

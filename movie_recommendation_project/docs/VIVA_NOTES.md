# CineMatch — Viva Notes (Learn These First)

## 30-second project introduction
“My project is CineMatch, a Personalized Movie Recommendation System. It recommends movies to a user based on patterns in movie ratings from users with similar preferences. I used Python, Pandas, NumPy, scikit-learn, and Streamlit. The core method is user-based collaborative filtering, and cosine similarity measures how similar two users’ rating patterns are. The bundled dataset is synthetic demo data, so the project can run reliably offline.”

## Essential concepts
**Recommendation system:** Software that ranks items a person may prefer, based on behavior, attributes, or both.

**Collaborative filtering:** Uses interactions/preferences from many users. The idea is that people with similar past ratings may like some of the same unseen movies.

**User-based collaborative filtering:** Finds users whose rating patterns resemble the target user, then uses their ratings to estimate the target user’s preference.

**Item-based collaborative filtering:** Finds items that receive similar ratings from the same users, then recommends items similar to ones the user liked. This project implements user-based, not item-based, filtering.

**Cosine similarity:** Measures the angle between two vectors. It is 1 for same-direction vectors, 0 for orthogonal vectors, and can be -1 for opposite vectors in general. Here rating vectors are non-negative, so values are usually between 0 and 1.
Formula: cosine(A,B) = (A · B) / (||A|| × ||B||)

**User–item matrix:** Rows are users, columns are movies, values are ratings. Missing cells mean the user has not rated that movie; they are not actual zero-star ratings.

**Predicted rating:** A score estimated from ratings supplied by similar users. It is not a guarantee that the user will like the movie.

**Cold-start problem:** New users have no history, or new movies have no ratings, so collaborative filtering has little evidence to work with.

**Sparsity:** Most users rate only a small portion of all available movies, so the matrix contains many missing values.

**Popularity fallback:** If the system lacks enough similarity-based evidence, it uses the movie’s average observed rating. This is less personalized.

**Streamlit:** Python framework for quickly building interactive data apps.

**Pandas / NumPy:** Data manipulation and numerical computation libraries.

**scikit-learn:** Provides utilities such as `cosine_similarity`.

## How the algorithm works
1. Load the movies catalog and ratings table.
2. Pivot ratings into a user-by-movie matrix.
3. Fill missing cells with zero for this simplified demo’s similarity calculation.
4. Compute pairwise cosine similarity between user vectors.
5. Exclude movies the target user has already rated.
6. For each remaining movie, combine ratings from similar users using similarity as a weight.
7. Rank candidates by estimated score and display the top N.

Simplified weighted estimate:
predicted(u,m) = Σ(sim(u,v) × rating(v,m)) / Σ(sim(u,v))

Only non-negative similarities are used as weights in this implementation. When the denominator is zero or no neighbors rated the movie, the code falls back to the average movie rating.

## Likely viva questions and short answers
**1. Why did you choose this project?**
It solves a familiar personalization problem and demonstrates data processing, similarity measures, and an interactive application.

**2. What problem does it solve?**
It reduces the effort of finding relevant movies by ranking unseen movies based on a user’s preference patterns.

**3. Which algorithm did you use?**
User-based collaborative filtering with cosine similarity and a similarity-weighted rating estimate.

**4. Why cosine similarity?**
It is a common way to compare vectors and is straightforward to compute using scikit-learn.

**5. What is the difference between content-based and collaborative filtering?**
Content-based methods recommend items with attributes similar to items a user liked. Collaborative filtering uses patterns of user-item interactions. This project uses collaborative filtering.

**6. What is the dataset?**
The bundled dataset is a small synthetic dataset created for an offline demonstration. The app can accept a ratings CSV with `userId`, `movieId`, and `rating` columns. Do not claim the bundled data is real MovieLens data.

**7. Why are movie genres present?**
They make the catalog understandable to a viewer. The implemented recommendation score is based on rating behavior, not directly on genre similarity.

**8. What does cosine similarity of 1 mean?**
The vectors point in the same direction, indicating very similar rating patterns under this representation.

**9. What is the cold-start problem?**
The system cannot make strong personalized recommendations for a new user without ratings, or for a new movie without interactions.

**10. What are the limitations?**
Small synthetic data, sparse ratings, simplified missing-value handling, potential popularity bias, and no persistent user accounts.

**11. How would you evaluate the model?**
Split known ratings into training and test sets. Predict held-out ratings and calculate MAE or RMSE. For ranked recommendations, use Precision@K, Recall@K, or NDCG@K. Avoid evaluating on the same ratings used to fit the model.

**12. What is RMSE?**
Root Mean Squared Error; it measures the square root of the average squared difference between predicted and actual ratings. Lower is generally better.

**13. Why not use deep learning?**
For a small academic demo, collaborative filtering is easier to explain, faster to run, and does not require a large dataset or complex training pipeline.

**14. What would you improve next?**
Use a larger real MovieLens dataset, better missing-value handling, train/test evaluation, item-based filtering or matrix factorization, a database, user login, and deployment.

**15. Is it production-ready?**
No. It is a functional academic prototype intended to demonstrate the core idea. Production use needs larger validated data, stronger evaluation, privacy/security work, and reliability testing.

## Be honest in your presentation
- Say “prototype” or “academic demonstration,” not “Netflix-level system.”
- Clearly disclose that bundled ratings are synthetic.
- Do not claim a model accuracy score unless you have actually run a valid held-out evaluation.
- Explain that estimated ratings are predictions, not certainty.

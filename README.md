# CineMatch — NCF Movie Recommender

CineMatch is a Streamlit movie discovery app backed by a PyTorch Neural Collaborative Filtering model. It combines learned movie embeddings with a lightweight SQLite product layer so users can create accounts, rate films, receive recommendations, search the catalog, find similar movies, and maintain a watchlist.

## What is included

- Account creation and login with PBKDF2-hashed passwords
- New-user onboarding through movie ratings
- Personalized recommendations after five ratings
- Search by title and filter by genre
- Similar-movie recommendations from learned movie embeddings
- Persistent ratings and watchlists in `backend/recommender.db`
- A redesigned dark, cinematic Streamlit interface
- FastAPI endpoints that can also support a mobile or React client

The current feedback loop is designed for immediate demo feedback: ratings are saved to SQLite and recommendations are regenerated from the user's rated-movie embedding profile. The original NCF checkpoint is not modified on every click. For a production system, ratings can later be used for scheduled retraining or an online-learning service.

## Project structure

```text
backend/
  app.py                 FastAPI API, authentication, SQLite persistence
  model_handler.py       NCF inference, search, similarity, cold start
  models/final_model.pth trained checkpoint
frontend/
  streamlit_app.py       Streamlit-only user interface
data/
  movies_data.csv        MovieLens ratings joined with movie metadata
```

## Run locally

Use Python 3.9+ and install the two requirement files:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

Start the API from the repository root:

```bash
cd backend
python3 app.py
```

In another terminal, from the repository root:

```bash
streamlit run frontend/streamlit_app.py
```

The Streamlit app defaults to `http://localhost:8000`. To point it at a deployed API:

```bash
export CINEMATCH_API_URL=https://your-api.example.com
streamlit run frontend/streamlit_app.py
```

The SQLite database is created automatically on first API startup. It is intentionally local for this project; use a managed database and real session/token authentication before deploying for multiple users.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/signup` | Create an account |
| POST | `/auth/login` | Sign in |
| GET | `/movies/search?q=&genre=` | Search and filter movies |
| GET | `/movies/{movie_id}/similar` | Find similar movies |
| POST | `/ratings` | Save or update a rating |
| GET | `/accounts/{account_id}/ratings` | Read a user's ratings |
| POST | `/recommend` | Generate personalized recommendations |
| GET | `/accounts/{account_id}/watchlist` | Read a watchlist |
| POST | `/watchlist` | Add a movie to a watchlist |
| DELETE | `/watchlist/{account_id}/{movie_id}` | Remove a movie |

Interactive API documentation is available at `http://localhost:8000/docs` while the backend is running.

## Model notes

The checkpoint uses user and movie embeddings, bias terms, and an MLP prediction path. The app uses the movie embeddings for similarity and cold-start onboarding. Recommendations exclude movies already rated by the account.

## Dataset

The project uses the MovieLens latest-small dataset from GroupLens. `data/movies_data.csv` is the joined ratings/movie metadata file used by the API. The original source files are retained under `data/ml-latest-small/`.

## License

[MIT](https://choosealicense.com/licenses/mit/)

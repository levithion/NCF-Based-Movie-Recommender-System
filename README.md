# CineMatch — NCF Movie Recommender

CineMatch is a Streamlit movie discovery app backed by a PyTorch Neural Collaborative Filtering model. It combines learned movie embeddings with a lightweight SQLite product layer so users can create accounts, rate films, receive recommendations, search the catalog, find similar movies, and maintain a watchlist.

## What is included

- Account creation and login with PBKDF2-hashed passwords
- New-user onboarding through movie ratings
- Personalized recommendations after five ratings
- Search by title and filter by genre
- IMDb/OMDb search for new movies and poster artwork
- Similar-movie recommendations from learned movie embeddings
- Persistent ratings and watchlists in `backend/recommender.db`
- A redesigned dark, cinematic Streamlit interface
- FastAPI endpoints that can also support a mobile or React client

The feedback loop is designed for immediate demo feedback: ratings are saved to SQLite and recommendations are regenerated using the hybrid scorer. Known movies combine the NCF movie-bias prior, content similarity, and popularity. New catalog movies with no NCF embedding use content similarity and popularity until they collect ratings. The original NCF checkpoint is not modified on every click; ratings can later be used for scheduled retraining.

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

Render is pinned to Python 3.11 through `.python-version` because the PyTorch checkpoint should use a prebuilt compatible wheel rather than attempting a source build on the newest Python runtime.

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

For Streamlit Community Cloud, add this to the app's **Settings → Secrets**. `localhost` will not work for a hosted frontend:

```toml
CINEMATCH_API_URL = "https://your-public-backend.example.com"
```

The FastAPI backend must be deployed separately on a public HTTPS URL (for example, Render, Railway, or Fly.io). Use that URL as the secret value, without a trailing endpoint such as `/docs`.

### IMDb/OMDb integration

The app uses the OMDb API for IMDb-linked title search, IMDb IDs, movie details, and poster URLs. OMDb requires an API key; request one from the [official OMDb API page](https://www.omdbapi.com/apikey.aspx), then configure it only on the backend host:

```text
OMDB_API_KEY=your_omdb_key
```

On Render, add `OMDB_API_KEY` under the backend service's Environment settings and redeploy. The Streamlit frontend never receives this key. In Discover → IMDb search, users can search a title, view posters, and import a movie into the CineMatch catalog. Imported movies are stored in SQLite and become available to the hybrid recommender.

The SQLite database is created automatically on first API startup. For Render, attach a persistent disk and set `CINEMATCH_DB_PATH=/var/data/recommender.db`; otherwise accounts, ratings, and watchlists can disappear when the service is redeployed or restarted. For a larger deployment, use a managed database and real session/token authentication.

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

The checkpoint uses user and movie embeddings, bias terms, and an MLP prediction path. The hybrid layer builds TF-IDF content features from movie titles and genres. Add a new movie to `data/ml-latest-small/movies.csv` and redeploy the backend; it can then be searched and recommended without changing checkpoint dimensions. Recommendations exclude movies already rated by the account.

## Dataset

The project uses the MovieLens latest-small dataset from GroupLens. `data/movies_data.csv` is the joined ratings/movie metadata file used by the API. The original source files are retained under `data/ml-latest-small/`.

## License

[MIT](https://choosealicense.com/licenses/mit/)

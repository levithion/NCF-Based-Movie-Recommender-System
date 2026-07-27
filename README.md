# 🎬 CineMatch

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Model-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![IMDb](https://img.shields.io/badge/Metadata-IMDb-F5C518?logo=imdb&logoColor=black)](https://www.imdb.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)

## ✨ AI-powered movie discovery app

CineMatch is a personalized movie discovery application built for people who want better recommendations without endlessly scrolling. Users can create an account, rate movies, discover personalized suggestions, search IMDb-linked movie data, view posters, find similar films, and save movies to a watchlist.

The application combines a Streamlit interface with a FastAPI backend and a hybrid recommendation engine built on the existing PyTorch NCF model.

## 🍿 What users can do

### 🔐 Create an account

Users can sign up with a name, email, and password, then sign in to their personal CineMatch profile.

### ⭐ Rate movies

Users rate movies from 0.5 to 5 stars. These ratings form a personal taste profile and immediately influence future recommendations.

### 🎯 Get personalized recommendations

After rating a few movies, users receive a curated “Picked for you” list. Recommendations exclude movies they have already rated.

### 🔎 Search the CineMatch catalog

Users can search movies by title and filter results by genre. Catalog movie cards can show poster artwork, genres, ratings, and actions such as rating, finding similar movies, or adding to a watchlist.

### 🎞️ Search IMDb-linked movie data

The IMDb Search area uses OMDb to search a much larger external movie catalog. Users can search titles that are not currently in MovieLens, view posters and release years, and import a movie into CineMatch.

### 🧠 Discover similar movies

Users can request movies similar to a selected title. Similarity is calculated from movie representations and content features such as title and genre.

### 📌 Maintain a watchlist

Users can save movies for later and remove them when they have watched them.

### 👤 View a personal profile

The Profile page shows the number of rated films, average rating, and rating history.

## 🤖 How recommendations work

CineMatch uses a hybrid recommendation strategy:

```text
NCF collaborative signal
        +
Content similarity from title and genres
        +
Movie popularity
        =
Personalized recommendation score
```

- Existing movies use learned NCF signals, content similarity, and popularity.
- New movies without historical ratings use content similarity and popularity.
- New movies can therefore be searched, imported, and recommended before they have enough user ratings for model retraining.
- New user ratings are stored immediately and update the user's recommendation profile.

The NCF checkpoint is not retrained after every rating. Ratings can be collected for scheduled model retraining later.

## 🏗️ Application architecture

```text
Streamlit frontend
        ↓
FastAPI backend
        ↓
Hybrid recommendation engine
        ↓
PyTorch NCF model + TF-IDF content features
        ↓
SQLite account, rating, watchlist, and imported-movie storage
```

### 🧰 Technology stack

- Streamlit — user interface
- FastAPI — backend API
- PyTorch — Neural Collaborative Filtering model
- scikit-learn — TF-IDF and content similarity
- SQLite — accounts, ratings, watchlists, and imported catalog movies
- OMDb API — IMDb-linked search, movie details, and poster URLs
- MovieLens — training ratings and initial movie catalog

## 🗂️ Project structure

```text
backend/
  app.py                 FastAPI routes and persistence
  model_handler.py       NCF and hybrid recommendation logic
  models/final_model.pth trained model checkpoint
frontend/
  streamlit_app.py       Streamlit application
data/
  movies_data.csv        Joined MovieLens ratings and metadata
  ml-latest-small/       Original MovieLens files
```

## 🚀 Run locally

Python 3.11 is recommended because it has compatible prebuilt PyTorch wheels. The repository includes `.python-version` files for Render and local version managers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

Start the backend from the repository root:

```bash
cd backend
python3 app.py
```

Start Streamlit in a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

The local frontend uses `http://localhost:8000` by default. For a deployed backend:

```bash
export CINEMATCH_API_URL="https://your-backend.example.com"
streamlit run frontend/streamlit_app.py
```

For Streamlit Community Cloud, add this under **App settings → Secrets**:

```toml
CINEMATCH_API_URL = "https://your-backend.example.com"
```

## 🌐 IMDb/OMDb setup

The backend uses OMDb for IMDb-linked title search and poster images. Create an API key from the [official OMDb API page](https://www.omdbapi.com/apikey.aspx), then add it to the backend deployment only:

```text
OMDB_API_KEY=your_omdb_key
```

On Render, add `OMDB_API_KEY` under the web service's Environment settings and redeploy. Never commit the key to GitHub or add it to the Streamlit frontend.

## ☁️ Persistence and deployment

The backend creates `backend/recommender.db` automatically for local use. For Render, configure a persistent disk mounted at `/var/data` and add:

```text
CINEMATCH_DB_PATH=/var/data/recommender.db
```

Without persistent storage, accounts, ratings, watchlists, and imported movies may be lost when the service is redeployed or restarted. A managed PostgreSQL database is recommended for production.

## 🔌 API overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/signup` | Create an account |
| POST | `/auth/login` | Sign in |
| GET | `/movies/search?q=&genre=` | Search the CineMatch catalog |
| GET | `/external/movies/search?q=` | Search IMDb-linked OMDb data |
| GET | `/external/movies/{imdb_id}` | Get external movie details |
| POST | `/catalog/import` | Import an external movie |
| GET | `/movies/{movie_id}/similar` | Find similar movies |
| POST | `/ratings` | Save or update a rating |
| POST | `/recommend` | Generate recommendations |
| GET | `/accounts/{account_id}/watchlist` | Read a watchlist |
| POST | `/watchlist` | Add a movie |
| DELETE | `/watchlist/{account_id}/{movie_id}` | Remove a movie |

When running locally, interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

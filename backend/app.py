"""FastAPI service for the NCF movie recommender.

The SQLite store is intentionally local and lightweight for this demo. Replace it
with PostgreSQL or another managed database for a multi-instance deployment.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import secrets
import sqlite3
import os
from typing import List, Optional

import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .model_handler import MovieRecommenderModel
except ImportError:
    from model_handler import MovieRecommenderModel

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'movies_data.csv'
MODEL_PATH = ROOT / 'backend' / 'models' / 'final_model.pth'
CATALOG_PATH = ROOT / 'data' / 'ml-latest-small' / 'movies.csv'
OMDB_API_KEY = os.getenv('OMDB_API_KEY', '')
OMDB_URL = 'https://www.omdbapi.com/'
DB_PATH = Path(os.getenv('CINEMATCH_DB_PATH', str(ROOT / 'backend' / 'recommender.db')))
model_handler = None
movies_df = None


def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_ratings (
            account_id INTEGER NOT NULL, movie_id INTEGER NOT NULL,
            rating REAL NOT NULL CHECK(rating >= 0.5 AND rating <= 5),
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(account_id, movie_id), FOREIGN KEY(account_id) REFERENCES accounts(id)
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            account_id INTEGER NOT NULL, movie_id INTEGER NOT NULL,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(account_id, movie_id), FOREIGN KEY(account_id) REFERENCES accounts(id)
        );
        CREATE TABLE IF NOT EXISTS catalog_movies (
            movie_id INTEGER PRIMARY KEY, imdb_id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL, genres TEXT NOT NULL, year TEXT, poster TEXT
        );
        ''')


def hash_password(password: str, salt: Optional[str] = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 120_000).hex()
    return f'{salt}${digest}'


def verify_password(password, stored):
    salt, _ = stored.split('$', 1)
    return secrets.compare_digest(hash_password(password, salt), stored)


def movie(movie_id):
    rows = model_handler.catalog[model_handler.catalog.movieId == movie_id]
    if rows.empty:
        raise HTTPException(404, 'Movie not found')
    row = rows.iloc[0]
    poster = row['poster'] if 'poster' in row.index and pd.notna(row['poster']) else ''
    return {'movie_id': int(row.movieId), 'title': str(row.title), 'genres': str(row.genres), 'poster': str(poster)}


def omdb_request(params):
    if not OMDB_API_KEY:
        raise HTTPException(503, 'OMDB_API_KEY is not configured on the backend')
    try:
        response = requests.get(OMDB_URL, params={**params, 'apikey': OMDB_API_KEY}, timeout=15)
        response.raise_for_status()
        payload = response.json()
        if payload.get('Response') == 'False':
            raise HTTPException(404, payload.get('Error', 'Movie not found'))
        return payload
    except requests.RequestException as exc:
        raise HTTPException(502, f'IMDb/OMDb request failed: {exc}')


def account(account_id):
    with db() as conn:
        row = conn.execute('SELECT id, email, display_name FROM accounts WHERE id=?', (account_id,)).fetchone()
    if not row:
        raise HTTPException(404, 'Account not found')
    return dict(row)


class AuthRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)
    display_name: str = Field(min_length=1, max_length=60)


class LoginRequest(BaseModel):
    email: str
    password: str


class RatingRequest(BaseModel):
    account_id: int
    movie_id: int
    rating: float = Field(ge=0.5, le=5)


class WatchlistRequest(BaseModel):
    account_id: int
    movie_id: int


class CatalogImportRequest(BaseModel):
    imdb_id: str
    title: str
    year: Optional[str] = None
    genres: str = 'Drama'
    poster: str = ''


class RecommendationRequest(BaseModel):
    account_id: int
    top_k: int = Field(default=12, ge=1, le=50)


@asynccontextmanager
async def lifespan(app):
    global model_handler, movies_df
    init_db()
    model_handler = MovieRecommenderModel(str(MODEL_PATH), str(DATA_PATH), str(CATALOG_PATH))
    with db() as conn:
        imported = conn.execute('SELECT * FROM catalog_movies').fetchall()
    for item in imported:
        model_handler.add_catalog_movie(dict(item))
    movies_df = model_handler.catalog
    yield


app = FastAPI(title='CineMatch API', version='2.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True,
                   allow_methods=['*'], allow_headers=['*'])


@app.get('/')
async def health():
    return {'status': 'healthy', 'model_loaded': model_handler is not None}


@app.post('/auth/signup')
async def signup(request: AuthRequest):
    email = request.email.strip().lower()
    try:
        with db() as conn:
            cursor = conn.execute('INSERT INTO accounts(email,password_hash,display_name) VALUES (?,?,?)',
                                  (email, hash_password(request.password), request.display_name.strip()))
            # Commit before reading through the separate connection used by account().
            conn.commit()
            return {'account': account(cursor.lastrowid), 'message': 'Account created'}
    except sqlite3.IntegrityError:
        raise HTTPException(409, 'An account with this email already exists')


@app.post('/auth/login')
async def login(request: LoginRequest):
    with db() as conn:
        row = conn.execute('SELECT * FROM accounts WHERE email=?', (request.email.strip().lower(),)).fetchone()
    if not row or not verify_password(request.password, row['password_hash']):
        raise HTTPException(401, 'Invalid email or password')
    return {'account': {'id': row['id'], 'email': row['email'], 'display_name': row['display_name']}}


@app.get('/movies/search')
async def search_movies(q: str = '', genre: Optional[str] = None, limit: int = Query(40, ge=1, le=100)):
    return {'movies': model_handler.search_movies(q, genre, limit), 'genres': model_handler.genres()}


@app.get('/external/movies/search')
async def search_external_movies(q: str = Query(..., min_length=2), page: int = Query(1, ge=1, le=100)):
    payload = omdb_request({'s': q, 'type': 'movie', 'page': page})
    results = []
    for item in payload.get('Search', []):
        results.append({'imdb_id': item.get('imdbID'), 'title': item.get('Title'),
                        'year': item.get('Year'), 'poster': '' if item.get('Poster') == 'N/A' else item.get('Poster', '')})
    return {'total_results': len(results), 'movies': results}


@app.get('/external/movies/{imdb_id}')
async def external_movie_details(imdb_id: str):
    payload = omdb_request({'i': imdb_id, 'plot': 'full'})
    return {'imdb_id': payload.get('imdbID'), 'title': payload.get('Title'),
            'year': payload.get('Year'), 'genres': payload.get('Genre', 'Drama').replace(', ', '|'),
            'poster': '' if payload.get('Poster') == 'N/A' else payload.get('Poster', ''),
            'plot': payload.get('Plot', ''), 'imdb_rating': payload.get('imdbRating', 'N/A'),
            'runtime': payload.get('Runtime', 'N/A'), 'director': payload.get('Director', 'N/A')}


@app.post('/catalog/import')
async def import_catalog_movie(request: CatalogImportRequest):
    imdb_column = model_handler.catalog.get('imdbId', pd.Series('', index=model_handler.catalog.index)).astype(str)
    existing = model_handler.catalog[imdb_column == request.imdb_id]
    if not existing.empty:
        return movie(int(existing.iloc[0].movieId))
    new_id = int(model_handler.catalog.movieId.max()) + 1
    item = {'movie_id': new_id, 'imdb_id': request.imdb_id, 'title': request.title,
            'year': request.year, 'genres': request.genres, 'poster': request.poster}
    with db() as conn:
        conn.execute('INSERT INTO catalog_movies(movie_id,imdb_id,title,genres,year,poster) VALUES(?,?,?,?,?,?)',
                     (new_id, request.imdb_id, request.title, request.genres, request.year, request.poster))
    model_handler.add_catalog_movie(item)
    return movie(new_id)


@app.get('/movies/{movie_id}/similar')
async def similar_movies(movie_id: int, limit: int = Query(8, ge=1, le=20)):
    return {'movie': movie(movie_id), 'movies': model_handler.similar_movies(movie_id, limit)}


def ratings_for(account_id):
    with db() as conn:
        return [dict(r) for r in conn.execute('SELECT movie_id, rating FROM user_ratings WHERE account_id=?', (account_id,))]


@app.post('/recommend')
async def recommend(request: RecommendationRequest):
    account(request.account_id)
    ratings = ratings_for(request.account_id)
    if not ratings:
        return {'account_id': request.account_id, 'recommendations': [], 'message': 'Rate a few movies to get recommendations'}
    recommendations = model_handler.hybrid_recommendations(ratings, request.top_k)
    return {'account_id': request.account_id, 'recommendations': [
        {'movie_id': item['movieId'], 'title': item['title'], 'genres': item['genres'],
         'poster': item.get('poster', ''),
         'predicted_rating': round(item.get('predicted_rating', 0), 2)} for item in recommendations]}


@app.post('/ratings')
async def add_rating(request: RatingRequest):
    account(request.account_id); movie(request.movie_id)
    with db() as conn:
        conn.execute('INSERT INTO user_ratings(account_id,movie_id,rating) VALUES(?,?,?) '
                     'ON CONFLICT(account_id,movie_id) DO UPDATE SET rating=excluded.rating, updated_at=CURRENT_TIMESTAMP',
                     (request.account_id, request.movie_id, request.rating))
    return {'message': 'Rating saved', 'movie_id': request.movie_id, 'rating': request.rating}


@app.get('/accounts/{account_id}/ratings')
async def get_ratings(account_id: int):
    account(account_id)
    with db() as conn:
        rows = conn.execute('SELECT movie_id, rating FROM user_ratings WHERE account_id=? ORDER BY updated_at DESC', (account_id,)).fetchall()
    return {'ratings': [{**movie(r['movie_id']), 'rating': r['rating']} for r in rows]}


@app.get('/accounts/{account_id}/watchlist')
async def get_watchlist(account_id: int):
    account(account_id)
    with db() as conn:
        rows = conn.execute('SELECT movie_id FROM watchlist WHERE account_id=? ORDER BY added_at DESC', (account_id,)).fetchall()
    return {'movies': [movie(r['movie_id']) for r in rows]}


@app.post('/watchlist')
async def add_watchlist(request: WatchlistRequest):
    account(request.account_id); movie(request.movie_id)
    with db() as conn:
        conn.execute('INSERT OR IGNORE INTO watchlist(account_id,movie_id) VALUES(?,?)', (request.account_id, request.movie_id))
    return {'message': 'Added to watchlist'}


@app.delete('/watchlist/{account_id}/{movie_id}')
async def remove_watchlist(account_id: int, movie_id: int):
    account(account_id)
    with db() as conn:
        conn.execute('DELETE FROM watchlist WHERE account_id=? AND movie_id=?', (account_id, movie_id))
    return {'message': 'Removed from watchlist'}


if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)

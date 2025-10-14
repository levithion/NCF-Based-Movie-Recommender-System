from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import pandas as pd
import numpy as np
from model_handler import MovieRecommenderModel
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎬 Movie Recommender API",
    description="AI-powered movie recommendation system using Neural Collaborative Filtering",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model_handler = None
movies_df = None

# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    total_recommendations: int

class UserRating(BaseModel):
    user_id: int
    movie_id: int
    rating: float

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Initialize model and data when server starts"""
    global model_handler, movies_df
    
    try:
        logger.info("🚀 Starting Movie Recommender API...")
        
        # Load the trained model
        model_handler = MovieRecommenderModel(
            model_path="models/final_model.pth",
            data_path="../data/movies_data.csv"
        )
        
        # Load movies data
        movies_df = pd.read_csv("../data/movies_data.csv")
        
        logger.info("✅ Model and data loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {str(e)}")
        raise e

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Movie Recommender API is running! 🎬",
        model_loaded=model_handler is not None
    )

@app.get("/movies")
async def get_all_movies():
    """Get all available movies"""
    try:
        movies_list = movies_df[['movieId', 'title', 'genres']].to_dict('records')
        return {
            "total_movies": len(movies_list),
            "movies": movies_list[:100]  # Return first 100 for performance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching movies: {str(e)}")

@app.get("/users")
async def get_all_users():
    """Get all available users"""
    try:
        unique_users = sorted(movies_df['userId'].unique().tolist())
        return {
            "total_users": len(unique_users),
            "users": unique_users[:50]  # Return first 50 users
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    try:
        if model_handler is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Get recommendations from model
        recommendations = model_handler.get_recommendations(
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for user {request.user_id}"
            )
        
        # Format response
        movie_recommendations = [
            MovieRecommendation(
                movie_id=rec['movieId'],
                title=rec['title'],
                genres=rec['genres'],
                predicted_rating=round(rec['predicted_rating'], 2)
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=movie_recommendations,
            total_recommendations=len(movie_recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: int):
    """Get rating history for a specific user"""
    try:
        user_ratings = movies_df[movies_df['userId'] == user_id]
        
        if user_ratings.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Sort by rating (highest first)
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        
        history = user_ratings[['movieId', 'title', 'genres', 'rating']].to_dict('records')
        
        return {
            "user_id": user_id,
            "total_ratings": len(history),
            "average_rating": round(user_ratings['rating'].mean(), 2),
            "rating_history": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user history: {str(e)}")

@app.post("/rate")
async def add_rating(rating: UserRating):
    """Add a new rating (for demo purposes)"""
    try:
        # In a real application, you'd save this to a database
        # For now, we'll just return success
        return {
            "message": f"Rating added successfully! User {rating.user_id} rated movie {rating.movie_id} with {rating.rating} stars",
            "user_id": rating.user_id,
            "movie_id": rating.movie_id,
            "rating": rating.rating
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding rating: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

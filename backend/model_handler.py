import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ItemBasedNCF(nn.Module):
    """Neural Collaborative Filtering Model - EXACT match to your trained model"""
    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super(ItemBasedNCF, self).__init__()
        
        # Embedding layers - these learn representations
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Neural network layers - FIXED to match your trained model
        layers = []
        input_dim = embedding_dim * 2  # User + Movie embeddings
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)  # Updated dropout to match training
            ])
            input_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.neural_layers = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # Get bias terms
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        
        # Concatenate user and movie embeddings
        x = torch.cat([user_embed, movie_embed], dim=1)
        
        # Pass through neural network
        rating = self.neural_layers(x)
        
        # Add bias terms
        prediction = rating.squeeze() + user_bias + movie_bias + self.global_bias
        
        # Clamp predictions to valid rating range
        prediction = torch.clamp(prediction, 0.5, 5.0)
        
        return prediction

class MovieRecommenderModel:
    """Movie Recommender Model Handler for FastAPI Backend"""
    
    def __init__(self, model_path: str, data_path: str):
        """Initialize the movie recommender model"""
        self.device = self._get_device()
        self.df = pd.read_csv(data_path)
        
        logger.info(f"Loaded dataset with {len(self.df)} ratings")
        logger.info(f"Dataset columns: {self.df.columns.tolist()}")
        
        # Create mappings
        self.user_to_idx, self.movie_to_idx, self.idx_to_user, self.idx_to_movie = self._create_mappings()
        
        # Load model
        self.model = self._load_model(model_path)
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model supports {len(self.user_to_idx)} users and {len(self.movie_to_idx)} movies")
    
    def _get_device(self):
        """Get the best available device for inference"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon) device")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    def _create_mappings(self):
        """Create user and movie ID mappings for the model"""
        try:
            # Get unique users and movies, sorted for consistency
            unique_users = sorted(self.df['userId'].unique())
            unique_movies = sorted(self.df['movieId'].unique())
            
            # Create forward mappings (original ID -> model index)
            user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
            movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
            
            # Create reverse mappings (model index -> original ID)
            idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
            idx_to_movie = {idx: movie_id for movie_id, idx in movie_to_idx.items()}
            
            logger.info(f"Created mappings for {len(unique_users)} users and {len(unique_movies)} movies")
            
            return user_to_idx, movie_to_idx, idx_to_user, idx_to_movie
            
        except Exception as e:
            logger.error(f"Error creating mappings: {str(e)}")
            raise e
    
    def _load_model(self, model_path: str):
        """Load the trained PyTorch model"""
        try:
            num_users = len(self.user_to_idx)
            num_movies = len(self.movie_to_idx)
            
            logger.info(f"Creating model for {num_users} users and {num_movies} movies")
            
            # Create model instance with CORRECT architecture - FIXED!
            model = ItemBasedNCF(
                num_users=num_users, 
                num_movies=num_movies, 
                embedding_dim=64,  # Match your trained model
                hidden_dims=[128, 64, 32]  # FIXED: Added the missing 32 dimension
            )
            
            # Load trained weights
            logger.info(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            logger.info("Model loaded and moved to device successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Dict]:
        """Get movie recommendations for a user"""
        try:
            # Check if user exists in our mappings
            if user_id not in self.user_to_idx:
                logger.warning(f"User {user_id} not found in training data")
                return []
            
            user_idx = self.user_to_idx[user_id]
            
            # Get movies the user has already rated
            user_movies = set(self.df[self.df['userId'] == user_id]['movieId'].values)
            all_movies = set(self.movie_to_idx.keys())
            unrated_movies = all_movies - user_movies
            
            logger.info(f"User {user_id} has rated {len(user_movies)} movies, {len(unrated_movies)} unrated")
            
            if not unrated_movies:
                logger.warning(f"User {user_id} has rated all available movies")
                return []
            
            # Predict ratings for unrated movies
            predictions = []
            
            with torch.no_grad():
                for movie_id in unrated_movies:
                    movie_idx = self.movie_to_idx[movie_id]
                    
                    # Create tensors on the correct device
                    user_tensor = torch.LongTensor([user_idx]).to(self.device)
                    movie_tensor = torch.LongTensor([movie_idx]).to(self.device)
                    
                    # Get prediction
                    predicted_rating = self.model(user_tensor, movie_tensor).item()
                    predictions.append((movie_id, predicted_rating))
            
            # Sort by predicted rating (highest first) and get top-k
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = predictions[:top_k]
            
            # Format recommendations with movie details
            recommendations = []
            for movie_id, rating in top_recommendations:
                try:
                    # Get movie information from dataset
                    movie_info = self.df[self.df['movieId'] == movie_id].iloc[0]
                    
                    recommendations.append({
                        'movieId': int(movie_id),
                        'title': str(movie_info['title']),
                        'genres': str(movie_info['genres']),
                        'predicted_rating': float(rating)
                    })
                except Exception as e:
                    logger.warning(f"Could not get details for movie {movie_id}: {str(e)}")
                    continue
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return []
    
    def health_check(self) -> Dict:
        """Perform a health check on the model"""
        try:
            # Test prediction with a sample user and movie
            if len(self.user_to_idx) > 0 and len(self.movie_to_idx) > 0:
                sample_user_idx = 0
                sample_movie_idx = 0
                
                with torch.no_grad():
                    user_tensor = torch.LongTensor([sample_user_idx]).to(self.device)
                    movie_tensor = torch.LongTensor([sample_movie_idx]).to(self.device)
                    
                    prediction = self.model(user_tensor, movie_tensor).item()
                
                return {
                    'status': 'healthy',
                    'device': str(self.device),
                    'num_users': len(self.user_to_idx),
                    'num_movies': len(self.movie_to_idx),
                    'sample_prediction': prediction,
                    'model_parameters': sum(p.numel() for p in self.model.parameters())
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No users or movies found in dataset'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}'
            }

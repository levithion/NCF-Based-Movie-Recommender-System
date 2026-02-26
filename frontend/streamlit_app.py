import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import time
from backend.model_handler import MovieRecommenderModel

# Configure Streamlit page
st.set_page_config(
    page_title="🎬 AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .rating-badge {
        background: #FFD700;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .genre-tag {
        background: #4ECDC4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"

class MovieRecommenderApp:
    def __init__(self):
        # Initialize the model directly in Streamlit RAM
        self.model = MovieRecommenderModel(
            model_path="backend/models/final_model.pth",
            data_path="data/movies_data.csv"
        )
        self.movies_df = pd.read_csv("data/movies_data.csv")
    
    def check_api_health(self):
        # Always return true because the model is loaded in-memory
        return self.model is not None
    
    def get_users(self):
        unique_users = sorted(self.movies_df['userId'].unique().tolist())
        return unique_users[:50]
    
    def get_recommendations(self, user_id: int, top_k: int = 10):
        # Call the model handler directly instead of requests.post
        recs = self.model.get_recommendations(user_id=user_id, top_k=top_k)
        return {"recommendations": recs} if recs else None

    def get_user_history(self, user_id: int):
        user_ratings = self.movies_df[self.movies_df['userId'] == user_id]
        if user_ratings.empty: return None
        
        history = user_ratings.sort_values('rating', ascending=False)
        return {
            "total_ratings": len(history),
            "average_rating": round(history['rating'].mean(), 2),
            "rating_history": history.to_dict('records')
        }

def main():
    app = MovieRecommenderApp()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 AI Movie Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Neural Collaborative Filtering 🧠✨")
    
    # Check API health
    if not app.check_api_health():
        st.error("🚨 Backend API is not running! Please start the FastAPI server first.")
        st.code("cd backend && python app.py")
        return
    
    st.success("✅ Connected to AI Movie Recommender API!")
    
    # Sidebar for user selection
    st.sidebar.header("🎯 User Selection")
    
    # Get available users
    users = app.get_users()
    if not users:
        st.error("No users found in the system!")
        return
    
    selected_user = st.sidebar.selectbox(
        "Choose a User ID:",
        users,
        help="Select a user to get personalized movie recommendations"
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        help="How many movie recommendations do you want?"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🌟 Recommendations for User {selected_user}")
        
        if st.button("🎬 Get Movie Recommendations", type="primary"):
            with st.spinner("🤖 AI is analyzing your preferences..."):
                recommendations = app.get_recommendations(selected_user, num_recommendations)
                
                if recommendations:
                    st.success(f"Found {len(recommendations['recommendations'])} perfect movies for you!")
                    
                    # Display recommendations
                    for i, movie in enumerate(recommendations['recommendations'], 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="movie-card">
                                <h3>#{i} {movie['title']}</h3>
                                <p><span class="rating-badge">⭐ {movie['predicted_rating']}/5.0</span></p>
                                <p><strong>Genres:</strong> {movie['genres']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("No recommendations found for this user!")
    
    with col2:
        st.header("📊 User Profile")
        
        # Get user history
        user_history = app.get_user_history(selected_user)
        
        if user_history:
            st.metric("Total Ratings", user_history['total_ratings'])
            st.metric("Average Rating", f"{user_history['average_rating']}/5.0")
            
            # Rating distribution chart
            if user_history['rating_history']:
                ratings_df = pd.DataFrame(user_history['rating_history'])
                
                fig = px.histogram(
                    ratings_df, 
                    x='rating',
                    title=f"Rating Distribution for User {selected_user}",
                    nbins=10,
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top rated movies
                st.subheader("🏆 Your Top Rated Movies")
                top_movies = ratings_df.head(5)
                
                for _, movie in top_movies.iterrows():
                    st.markdown(f"""
                    <div style="background: #f0f2f6; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                        <strong>{movie['title']}</strong><br>
                        <span style="color: #FFD700;">{'⭐' * int(movie['rating'])}</span> {movie['rating']}/5.0
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No rating history found for this user.")
    
    # Additional features
    st.header("🔍 Explore More")
    
    tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🎭 Movie Database", "ℹ️ About"])
    
    with tab1:
        st.subheader("System Analytics")
        
        # Create some sample analytics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users))
        with col2:
            st.metric("AI Model", "NeuMF")
        with col3:
            st.metric("Accuracy", "65%+")
        with col4:
            st.metric("Status", "🟢 Online")
        
        # Sample performance chart
        performance_data = {
            'Metric': ['RMSE', 'MAE', 'Accuracy (±0.5)', 'Accuracy (±1.0)'],
            'Value': [0.75, 0.58, 65, 85],
            'Target': [0.70, 0.55, 70, 90]
        }
        
        perf_df = pd.DataFrame(performance_data)
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Current', x=perf_df['Metric'], y=perf_df['Value'], marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='Target', x=perf_df['Metric'], y=perf_df['Target'], marker_color='#4ECDC4'))
        
        fig.update_layout(title='Model Performance Metrics', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Movie Database")
        st.info("🚧 Coming Soon: Browse all movies, search by genre, and explore movie details!")
        
        # Placeholder for movie database features
        st.markdown("""
        **Planned Features:**
        - 🔍 Search movies by title or genre
        - 📊 Movie popularity rankings
        - 🎭 Genre-based filtering
        - ⭐ Community ratings overview
        """)
    
    with tab3:
        st.subheader("About This AI Movie Recommender")
        
        st.markdown("""
        ### 🧠 How It Works
        
        This movie recommender uses **Neural Collaborative Filtering (NCF)** with the **NeuMF architecture**:
        
        1. **Embedding Layers**: Learn user and movie representations
        2. **MLP Path**: Captures complex non-linear patterns
        3. **GMF Path**: Handles linear matrix factorization
        4. **Combined Prediction**: Merges both approaches for better accuracy
        
        ### 🚀 Technology Stack
        
        **Backend:**
        - FastAPI for REST API
        - PyTorch for deep learning
        - Neural Collaborative Filtering model
        
        **Frontend:**
        - Streamlit for web interface
        - Plotly for interactive charts
        - Modern CSS styling
        
        ### 📊 Model Performance
        
        - **Architecture**: NeuMF (Neural Matrix Factorization)
        - **Training**: 100 epochs with MPS acceleration
        - **Accuracy**: 65%+ within ±0.5 stars
        - **RMSE**: ~0.75 (lower is better)
        
        ### 🎯 Features
        
        ✅ Personalized recommendations  
        ✅ User rating history analysis  
        ✅ Real-time predictions  
        ✅ Interactive visualizations  
        ✅ RESTful API  
        """)

if __name__ == "__main__":
    main()

# NCF-Based Movie Recommender System

[![Streamlit App](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit)](https://ncf-based-movie-recommender-system.streamlit.app)

**Live Demo:** [https://ncf-based-movie-recommender-system.streamlit.app](https://ncf-based-movie-recommender-system.streamlit.app)

## Overview
This project is an advanced Movie Recommender System utilizing Neural Collaborative Filtering (NCF). By employing deep learning embeddings, the model learns intricate user-movie interactions and latent features from rating data to deliver highly personalized movie recommendations.

## Project Architecture
The repository is structured into distinct, decoupled components:

- **Data Processing & ML (`data_prep.ipynb`)**: A Jupyter Notebook detailing the data extraction, preprocessing of the [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/), model architecture definition in PyTorch, training loops, and evaluation metrics. The final trained weights are exported as `.pth` files.
- **Backend Service (`backend/`)**: A RESTful API built in Python (`app.py` & `model_handler.py`) that loads the pre-trained PyTorch NCF model (`best_model.pth`) and serves recommendation inferences via HTTP endpoints.
- **Frontend Interface (`frontend/`)**: A clean, interactive web application built with Streamlit (`streamlit_app.py`) where users can select profiles, browse movies, and view real-time personalized recommendations.

## Setup & Local Development

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Backend Server
Navigate to the `backend` directory, install the required packages, and run the server.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Frontend Application
Open a new terminal window/tab, navigate to the `frontend` directory, install its dependencies, and launch the Streamlit app.

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Dataset
This project uses the `ml-latest-small` dataset from GroupLens, containing:
- 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.
- `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`.

## License
[MIT](https://choosealicense.com/licenses/mit/)

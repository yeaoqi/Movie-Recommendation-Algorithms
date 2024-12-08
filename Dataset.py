import pandas as pd
import os
import requests
import zipfile
import urllib.request
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

# Define dataset
class MovielensDataset(Dataset):
    def __init__(self, ratings):
        """Initialize the dataset with user, movie, and rating data."""
        self.user_ids = torch.tensor(ratings["user_id"].values, dtype=torch.long)
        self.movie_ids = torch.tensor(ratings["movie_id"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings["rating"].values, dtype=torch.float32)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.ratings)

    def __getitem__(self, idx):
        """Return a single sample of user ID, movie ID, and rating."""
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


# Dataset for AutoRec model
class MovielensAutoRecDataset(Dataset):
    def __init__(self, ratings_matrix):
        """
        Initialize the dataset with user and movie ratings data.
        ratings_matrix: A full matrix of user-item ratings (num_users x num_movies)
        """
        self.ratings_matrix = torch.tensor(ratings_matrix, dtype=torch.float32)

    def __len__(self):
        """Return the total number of users (i.e., number of rows in ratings_matrix)."""
        return self.ratings_matrix.size(0)

    def __getitem__(self, idx):
        """
        Return a single sample of user ratings for a specific user.
        idx: user index (row in the ratings matrix).
        """
        user_ratings = self.ratings_matrix[idx]
        return user_ratings



def download_movielens_100k():
    # 定义下载地址和目标路径
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    data_dir = "./data/movielens/"
    zip_path = os.path.join(data_dir, "ml-100k.zip")

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 下载数据集
    if not os.path.exists(zip_path):
        print("Downloading Movielens dataset...")
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")

    # 解压数据集
    if not os.path.exists(os.path.join(data_dir, "ml-100k")):
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    # 加载数据
    ratings_path = os.path.join(data_dir, "ml-100k", "u.data")
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(ratings_path, sep="\t", names=columns)

    print("Sample of the dataset:")
    print(ratings.head())


# Load and preprocess data
def load_movielens_100k(data_dir):
    """Load and preprocess the MovieLens dataset."""
    ratings_path = os.path.join(data_dir, "u.data")
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(ratings_path, sep="\t", names=columns)

    # Map user and movie IDs to zero-based indices
    user_ids = ratings["user_id"].unique()
    movie_ids = ratings["movie_id"].unique()
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings["user_id"] = ratings["user_id"].map(user_to_index)
    ratings["movie_id"] = ratings["movie_id"].map(movie_to_index)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    return ratings, num_users, num_movies



def load_movielens_1m(data_dir="./data/movielens/ml-1m"):

    # Define dataset URL and local paths
    dataset_url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    extracted_path = os.path.join(data_dir, "ml-1m")

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset if it doesn't exist
    if not os.path.exists(extracted_path):
        print("Downloading MovieLens 1M dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    else:
        print("Dataset already exists. Skipping download and extraction.")

    # Load ratings data
    ratings_path = os.path.join(extracted_path, "ratings.dat")
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=columns,
        engine="python"
    )

    # Map user and movie IDs to zero-based indices
    user_ids = ratings["user_id"].unique()
    movie_ids = ratings["movie_id"].unique()
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings["user_id"] = ratings["user_id"].map(user_to_index)
    ratings["movie_id"] = ratings["movie_id"].map(movie_to_index)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    print(f"Loaded MovieLens 1M dataset: {num_users} users, {num_movies} movies.")
    return ratings, num_users, num_movies



def show_movielens1m_data():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_dir = "./data/movielens/ml-1m"
    extracted_path = os.path.join(data_dir, "ml-1m")
    ratings_path = os.path.join(extracted_path, "ratings.dat")
    movies_path = os.path.join(extracted_path, "movies.dat")
    users_path = os.path.join(extracted_path, "users.dat")

    # 读取文件时指定编码
    encoding = "latin1"  # 可根据实际检测结果调整
    movies = pd.read_csv(movies_path, sep='::', names=['MovieID', 'Title', 'Genres'], engine='python', encoding=encoding)
    ratings = pd.read_csv(ratings_path, sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding=encoding)
    users = pd.read_csv(users_path, sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding=encoding)

    # 数据展示和可视化
    print("Movies Data:")
    print(movies.head())
    print("\nRatings Data:")
    print(ratings.head())
    print("\nUsers Data:")
    print(users.head())
    print("Movies Info:")
    print(movies.info())
    print("\nRatings Info:")
    print(ratings.info())
    print("\nUsers Info:")
    print(users.info())
    print("\nRatings Statistics:")
    print(ratings.describe())

    # 评分分布
    sns.histplot(ratings['Rating'], bins=5, kde=False)
    plt.title("Ratings Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.show()
    # 年龄分布
    sns.countplot(x='Age', data=users)
    plt.title("User Age Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.show()

    # 电影流派统计
    genres = movies['Genres'].str.get_dummies('|').sum().sort_values(ascending=False)
    genres.plot(kind='bar')
    plt.title("Movie Genres Distribution")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.show()


show_movielens1m_data()
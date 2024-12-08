import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from Dataset import *


# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the matrix factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        interaction = (user_emb * movie_emb).sum(dim=1)
        return interaction + user_bias + movie_bias + self.global_bias

    def train_model(self, train_loader, num_epochs, lr, model_path, plot_path):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        loss_history = []

        # Load best model if available
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print("Loaded the best model from disk.")
        else:
            print("No existing model found. Training from scratch.")

        # Move model to the device (GPU/CPU)
        self.to(device)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for user_ids_batch, movie_ids_batch, ratings_batch in train_loader:
                # Move data to the device
                user_ids_batch = user_ids_batch.to(device)
                movie_ids_batch = movie_ids_batch.to(device)
                ratings_batch = ratings_batch.to(device)

                optimizer.zero_grad()
                predictions = self(user_ids_batch, movie_ids_batch)
                loss = criterion(predictions, ratings_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            loss_history.append(total_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

            # Save the best model
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch + 1} with loss {total_loss:.4f}")

        # Plot loss progression
        self.plot_loss(loss_history, plot_path)

    def plot_loss(self, loss_history, plot_path):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Progression")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def evaluate_model(self, test_loader):
        self.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_samples = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for user_ids_batch, movie_ids_batch, ratings_batch in test_loader:
                # Move data to the device
                user_ids_batch = user_ids_batch.to(device)
                movie_ids_batch = movie_ids_batch.to(device)
                ratings_batch = ratings_batch.to(device)

                predictions = self(user_ids_batch, movie_ids_batch)
                loss = criterion(predictions, ratings_batch)
                total_loss += loss.item()

                # RMSE and MAE calculations
                mse = torch.mean((predictions - ratings_batch) ** 2).item()
                mae = torch.mean(torch.abs(predictions - ratings_batch)).item()

                total_mse += mse
                total_mae += mae
                num_samples += 1

        avg_loss = total_loss / num_samples
        avg_mse = total_mse / num_samples
        avg_mae = total_mae / num_samples

        rmse = np.sqrt(avg_mse)

        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {avg_mae:.4f}")

        return avg_loss, rmse, avg_mae


# Main function
def main():
    data_dir = "./data/movielens/ml-1m/"
    ratings, num_users, num_movies = load_movielens_1m(data_dir)

    print(f"Number of users: {num_users}, Number of movies: {num_movies}")

    # Split the dataset into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    train_dataset = MovielensDataset(train_data)
    test_dataset = MovielensDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    embedding_dim = 5
    num_epochs = 100
    learning_rate = 0.0001
    model_path = "model/MF_ml1m_dim5.pth"
    plot_path = "pic/ml1m_dim5/MF_ml1m_dim5_2.png"

    # Initialize the model
    model = MatrixFactorization(num_users, num_movies, embedding_dim)

    # Train the model
    model.train_model(train_loader, num_epochs, learning_rate, model_path, plot_path)

    # Evaluate the model on the test set
    model.load_state_dict(torch.load(model_path))
    model.evaluate_model(test_loader)


if __name__ == "__main__":
    main()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from Dataset import *


# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# NeuMF Model definition
class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim, mlp_layers=[64, 32]):
        super(NeuMF, self).__init__()

        # GMF part (Generalized Matrix Factorization)
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding_gmf = nn.Embedding(num_movies, embedding_dim)

        # MLP part (Multilayer Perceptron)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding_mlp = nn.Embedding(num_movies, embedding_dim)

        # MLP layers
        self.mlp_layers = []
        input_size = embedding_dim * 2  # user + movie embeddings
        for layer_size in mlp_layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp_layers = nn.ModuleList(self.mlp_layers)

        # Final prediction layer
        self.final_layer = nn.Linear(input_size + embedding_dim, 1)  # Add GMF part

    def forward(self, user_ids, movie_ids):
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        movie_emb_gmf = self.movie_embedding_gmf(movie_ids)
        gmf_output = (user_emb_gmf * movie_emb_gmf)

        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        movie_emb_mlp = self.movie_embedding_mlp(movie_ids)
        mlp_input = torch.cat([user_emb_mlp, movie_emb_mlp], dim=1)
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Ensure that the dimensions of gmf_output and mlp_output are compatible for concatenation
        gmf_output = gmf_output.view(gmf_output.size(0), -1)  # Make sure gmf_output is a 2D tensor
        mlp_output = mlp_output.view(mlp_output.size(0), -1)  # Make sure mlp_output is a 2D tensor

        # Concatenate GMF and MLP outputs
        combined_output = torch.cat([gmf_output, mlp_output], dim=1)

        # Final prediction
        predictions = self.final_layer(combined_output)
        return predictions.squeeze()


# Training and evaluation loop
class NeuMFModel:
    def __init__(self, num_users, num_movies, embedding_dim, learning_rate, l2_lambda=0.001, mlp_layers=[64, 32], device=device):
        self.device = device
        self.model = NeuMF(num_users, num_movies, embedding_dim, mlp_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.l2_lambda = l2_lambda  # L2 regularization strength

    def train_model(self, train_loader, test_loader, num_epochs, model_path, plot_path):
        best_loss = float("inf")
        loss_history_train = []
        loss_history_test = []

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            total_loss_train = 0
            for user_ids_batch, movie_ids_batch, ratings_batch in train_loader:
                user_ids_batch = user_ids_batch.to(self.device)
                movie_ids_batch = movie_ids_batch.to(self.device)
                ratings_batch = ratings_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(user_ids_batch, movie_ids_batch)
                loss = self.criterion(predictions, ratings_batch)

                # L2 regularization: Add L2 penalty to the loss
                l2_reg = 0
                for param in self.model.parameters():
                    l2_reg += torch.norm(param) ** 2
                loss += self.l2_lambda * l2_reg  # Add L2 penalty

                loss.backward()
                self.optimizer.step()
                total_loss_train += loss.item()

            # Evaluate on the test set
            self.model.eval()
            total_loss_test = 0
            with torch.no_grad():
                for user_ids_batch, movie_ids_batch, ratings_batch in test_loader:
                    user_ids_batch = user_ids_batch.to(self.device)
                    movie_ids_batch = movie_ids_batch.to(self.device)
                    ratings_batch = ratings_batch.to(self.device)

                    predictions = self.model(user_ids_batch, movie_ids_batch)
                    loss = self.criterion(predictions, ratings_batch)
                    total_loss_test += loss.item()

            # Save training and testing loss
            loss_history_train.append(total_loss_train)
            loss_history_test.append(total_loss_test)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss_train:.4f}, Test Loss: {total_loss_test:.4f}")

            # Save the best model based on test_loss
            if total_loss_test < best_loss:
                best_loss = total_loss_test
                torch.save(self.model.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch + 1} with test loss {total_loss_test:.4f}")

        # Plot loss progression
        self.plot_loss(loss_history_train, loss_history_test, plot_path)

    def plot_loss(self, loss_history_train, loss_history_test, plot_path):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(loss_history_train) + 1), loss_history_train, marker='o', label="Train Loss")
        plt.plot(range(1, len(loss_history_test) + 1), loss_history_test, marker='x', label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Test Loss Progression")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def evaluate_model(self, test_loader):
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_mae = 0
        num_samples = 0

        with torch.no_grad():
            for user_ids_batch, movie_ids_batch, ratings_batch in test_loader:
                user_ids_batch = user_ids_batch.to(self.device)
                movie_ids_batch = movie_ids_batch.to(self.device)
                ratings_batch = ratings_batch.to(self.device)

                predictions = self.model(user_ids_batch, movie_ids_batch)
                loss = self.criterion(predictions, ratings_batch)
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


# Main function to train and evaluate the NeuMF model
def main():
    # Load the dataset
    data_dir = "./data/movielens/ml-1m"
    ratings, num_users, num_movies = load_movielens_1m(data_dir)

    print(f"Number of users: {num_users}, Number of movies: {num_movies}")

    # Split the dataset into training and testing sets (80% train, 20% test)
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    train_dataset = MovielensDataset(train_data)
    test_dataset = MovielensDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    embedding_dim = 5
    num_epochs = 50
    learning_rate = 0.001
    mlp_layers = [5]
    l2_lambda = 0.0001
    model_path = "model/NeuMF_dim5_1.pth"
    plot_path = "pic/neumf_dim5/neumf_dim5_1.png"

    # Initialize the NeuMF model
    model = NeuMFModel(num_users, num_movies, embedding_dim, learning_rate, l2_lambda, mlp_layers)

    # Train the model
    model.train_model(train_loader, test_loader, num_epochs, model_path, plot_path)

    # Evaluate the model on the test set
    model.model.load_state_dict(torch.load(model_path))
    model.evaluate_model(test_loader)

if __name__ == "__main__":
    main()



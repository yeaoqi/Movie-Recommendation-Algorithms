import torch
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim, nn
import torch.nn.functional as F
import network as nets
import matplotlib.pyplot as plt
import os


class Model:
    def __init__(self, hidden, learning_rate, batch_size):
        """
        Initialize the model with hyperparameters and the AutoEncoder network.

        Args:
            hidden (list): Layer sizes for the AutoEncoder.
            learning_rate (float): Learning rate for optimization.
            batch_size (int): Batch size for training.
        """
        self.batch_size = batch_size
        # Initialize the AutoEncoder network.
        self.net = nets.AutoEncoder(hidden)
        # Initialize the optimizer (Stochastic Gradient Descent with momentum and weight decay).
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
        # Input feature size (determined by the first layer in `hidden`).
        self.feature_size = hidden[0]

        # Initialize lists to track losses for plotting.
        self.train_losses = []
        self.test_rmse = []

        # Track the best model based on test RMSE.
        self.best_rmse = float('inf')  # Initialize with a very high value.
        self.best_model_state = None  # To store the best model state.

    def run(self, trainset, testlist, num_epoch, save_path="loss_plots"):
        """
        Run the training and testing process for a specified number of epochs.

        Args:
            trainset (Dataset): Training dataset object.
            testlist (list): List of test samples (user, item, rating).
            num_epoch (int): Number of epochs to train the model.
            save_path (str): Directory to save the loss plot and the best model.
        """
        for epoch in range(1, num_epoch + 1):
            # Create a DataLoader for batch processing of the training data.
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True, pin_memory=True)
            # Train the model for one epoch.
            train_loss = self.train(train_loader, epoch)
            # Test the model after training.
            rmse = self.test(trainset, testlist)

            # Record the losses for plotting.
            self.train_losses.append(train_loss)
            self.test_rmse.append(rmse)

            # Save the model if the current test RMSE is the best (lowest).
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_model_state = self.net.state_dict()  # Store the model state with the best RMSE.

        # Save the best model after all epochs.
        if self.best_model_state is not None:
            self.save_model(save_path)

        # Plot and save the losses.
        self.plot_losses(save_path)

    def train(self, train_loader, epoch):
        """
        Train the AutoEncoder model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for batch training.
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for this epoch.
        """
        self.net.train()  # Set the network to training mode.

        # Create placeholders for features and masks.
        features = Variable(torch.FloatTensor(self.batch_size, self.feature_size))
        masks = Variable(torch.FloatTensor(self.batch_size, self.feature_size))

        total_loss = 0
        count = 0

        # Iterate through the training data in batches.
        for bid, (feature, mask) in enumerate(train_loader):
            # Handle batch size mismatches (last batch).
            if mask.shape[0] == self.batch_size:
                features.data.copy_(feature)
                masks.data.copy_(mask)
            else:
                features = Variable(feature)
                masks = Variable(mask)

            # Zero the gradients from the previous iteration.
            self.opt.zero_grad()

            # Forward pass through the AutoEncoder.
            output = self.net(features)

            # Compute the masked mean squared error loss.
            loss = F.mse_loss(output * masks, features * masks)

            # Backward pass to compute gradients.
            loss.backward()

            # Update the weights using the optimizer.
            self.opt.step()

            # Accumulate loss for reporting.
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print("Epoch %d, train loss: %.4f" % (epoch, avg_loss))

        return avg_loss

    def test(self, trainset, testlist):
        """
        Test the AutoEncoder model on the test data.

        Args:
            trainset (Dataset): Training dataset object.
            testlist (list): List of test samples (user, item, rating).

        Returns:
            float: Root Mean Squared Error (RMSE) on the test data.
        """
        self.net.eval()  # Set the network to evaluation mode.

        # Get the full feature matrix, mask, and user-based flag from the training dataset.
        x_mat, mask, user_based = trainset.get_mat()
        features = Variable(x_mat)

        # Forward pass through the AutoEncoder to reconstruct the matrix.
        xc = self.net(features)

        # If not user-based, transpose the output matrix.
        if not user_based:
            xc = xc.t()

        # Convert the reconstructed matrix to a NumPy array for evaluation.
        xc = xc.cpu().data.numpy()

        # Compute Root Mean Squared Error (RMSE) on the test list.
        rmse = 0.0
        for (i, j, r) in testlist:
            rmse += (xc[i][j] - r) ** 2
        rmse = math.sqrt(rmse / len(testlist))

        print(" Test RMSE = %.4f" % rmse)

        return rmse

    def save_model(self, save_path):
        """
        Save the model with the lowest RMSE to a file.

        Args:
            save_path (str): Directory to save the model.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_path = os.path.join(save_path, 'best_model.pth')
        torch.save(self.best_model_state, model_path)
        print(f"Best model saved at {model_path}")

    def plot_losses(self, save_path):
        """
        Plot and save the training loss and test RMSE over epochs.

        Args:
            save_path (str): Directory to save the plot.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.test_rmse, label='Test RMSE', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / RMSE')
        plt.title('Training Loss and Test RMSE')
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file.
        plot_path = os.path.join(save_path, 'loss_plot.png')
        plt.savefig(plot_path)
        print(f"Loss plot saved at {plot_path}")
        plt.close()

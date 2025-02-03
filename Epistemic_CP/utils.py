# used torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor

# numpy and sklearn based packages
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt

from copy import deepcopy

# Interval score loss
def interval_score_loss(high_est, low_est, actual, alpha):
    return (
        high_est
        - low_est
        + 2 / alpha * (low_est - actual) * (actual < low_est)
        + 2 / alpha * (actual - high_est) * (actual > high_est)
    )

def average_interval_score_loss(high_est, low_est, actual, alpha):
    return np.mean(interval_score_loss(high_est, low_est, actual, alpha))

# general interval length
def compute_interval_length(upper_int, lower_int):
    return upper_int - lower_int

def average_coverage_clf(pred_sets, labels):
    empirical_coverage = pred_sets[np.arange(pred_sets.shape[0]), labels].mean()

    return empirical_coverage

# pearson correlation
def corr_coverage_widths(high_est, low_est, actual):
    coverage_indicator_vector = coverage_indicators(high_est, low_est, actual)
    widths_vector = compute_interval_length(high_est, low_est)
    return np.abs(np.corrcoef(coverage_indicator_vector, widths_vector)[0, 1])


# marginal coverage
def coverage_indicators(high_est, low_est, actual):
    return (high_est >= actual) & (low_est <= actual)

def average_coverage(high_est, low_est, actual):
    return np.mean(coverage_indicators(high_est, low_est, actual))


# base quantile regressors
class GradientBoostingQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Parameters:
        -----------
        I. alpha : float, default=0.1
            The significance level for quantile regression. The model will predict
            quantiles at alpha/II and I - alpha/II for I - alpha coverage.
        II. n_estimators : int, default=C
            The number of boosting stages.
        III. learning_rate : float, default=0.I
            Shrinks the contribution of each tree.
        IV. max_depth : int, default=III
            The maximum depth of each tree.

        Returns:
        --------
        self : object
            Returns self.
        """
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = {}

    def fit(self, X, y):
        """
        Fit two Gradient Boosting models for quantiles at alpha/2 and 1 - alpha/2.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            The input features.
        y : array-like, shape (n_samples,)
            The target values.
        """
        X = X.reshape(-1, 1)  # Ensure 2D input

        # Define quantile levels
        quantiles = [self.alpha / 2, 1 - self.alpha / 2]

        # Train separate models for each quantile
        for q in quantiles:
            model = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
            )
            model.fit(X, y)
            self.models[q] = model

        return self

    def predict(self, X):
        """
        Predict quantiles at alpha/2 and 1 - alpha/2.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 1)
            The input features.

        Returns:
        --------
        quantiles : array-like, shape (n_samples, 2)
            A 2D array where the first column contains the alpha/2 quantile predictions
            and the second column contains the 1 - alpha/2 quantile predictions.
        """
        X = X.reshape(-1, 1)  # Ensure 2D input

        q_low = self.models[self.alpha / 2].predict(X)
        q_high = self.models[1 - self.alpha / 2].predict(X)

        return np.column_stack((q_low, q_high))


# Neural network base model for regression
class Net_reg(nn.Module):
    """
    A neural network regression model with batch normalization, dropout, and early stopping.

    Attributes:
    -----------
    I. fc1 (nn.Linear): First fully connected layer.
    II. bn1 (nn.BatchNorm1d): Batch normalization for the first layer.
    III. dropout1 (nn.Dropout): Dropout for the first layer.
    IV. fc2 (nn.Linear): Second fully connected layer.
    V. bn2 (nn.BatchNorm1d): Batch normalization for the second layer.
    VI. dropout2 (nn.Dropout): Dropout for the second layer.
    VII. fc3 (nn.Linear): Third fully connected layer.
    VIII. bn3 (nn.BatchNorm1d): Batch normalization for the third layer.
    IX. dropout3 (nn.Dropout): Dropout for the third layer.
    X. fc4 (nn.Linear): Output layer.
    XI. random_numpy (int): Random state for reproducibility.
    XII. val_size (float): Validation set size.
    XIII. epochs (int): Number of epochs for training.
    XIV. batch_size (int): Batch size for training.
    XV. lr (float): Learning rate.
    XVI. patience (int): Patience for early stopping.
    XVII. x_scaler (StandardScaler): Scaler for input features.
    XVIII. y_scaler (MinMaxScaler): Scaler for output values.
    XIX. loss_history_validation (list): List to track validation loss history.
    XX. loss_history_train (list): List to track training loss history.
    XXI. epoch_list (list): List to track epochs.
    XXII. best_loss_history_val (list): List to track best validation loss history.

    Methods:
    --------
    I. __init__(self, input_dim, first_l=64, second_l=32, third_l=16, random_state=650, val_size=0.3, epochs=1000, batch_size=35, lr=0.01, patience=30):
        Initializes the neural network with the given parameters.

    II. init_weights(self):
        Initializes the weights of the network using Xavier normal initialization.

    III. forward(self, x):
        Defines the forward pass of the network.

    IV. predict(self, x):
        Predicts the output for the given input data.

    V. fit(self, x_train, y_train):
        Trains the neural network on the given training data.
    """
    def __init__(
        self,
        input_dim,
        first_l=64,
        second_l=32,
        third_l=16,
        random_state=650,
        val_size=0.3,
        epochs=1000,
        batch_size=35,
        lr=0.01,
        patience=30,
    ):
        super(Net_reg, self).__init__()
        self.fc1 = nn.Linear(input_dim, first_l)
        self.bn1 = nn.BatchNorm1d(first_l)  # Batch Norm
        self.dropout1 = nn.Dropout(0.2)  # Reduced dropout

        self.fc2 = nn.Linear(first_l, second_l)
        self.bn2 = nn.BatchNorm1d(second_l)  # Batch Norm
        self.dropout2 = nn.Dropout(0.1)  # Reduced dropout

        self.fc3 = nn.Linear(second_l, third_l)
        self.bn3 = nn.BatchNorm1d(third_l)  # Batch Norm
        self.dropout3 = nn.Dropout(0.05)  # Reduced dropout

        self.fc4 = nn.Linear(third_l, 1)
        self.random_numpy = random_state

        # validation size, epochs and batch_size
        self.val_size = val_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        # Initialize scalers
        self.x_scaler = StandardScaler()
        self.y_scaler = MinMaxScaler()  # Using MinMaxScaler for Y

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_normal_(m.weight, gain=gain)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

    def predict(self, x):
        # Normalize input features
        x = self.x_scaler.transform(x)
        x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y_pred = self.forward(x).numpy().reshape(-1)

        # Inverse transform the predicted values
        return self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    def fit(
        self,
        x_train,
        y_train,
    ):
        # Splitting into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.val_size,
            random_state=self.random_numpy,
        )

        # Normalize input features with StandardScaler
        self.x_scaler.fit(x_train)
        x_train = self.x_scaler.transform(x_train)
        x_val = self.x_scaler.transform(x_val)

        # Normalize output (y) with MinMaxScaler
        self.y_scaler.fit(y_train.reshape(-1, 1))
        y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        # Encoding data into torch tensor
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32), torch.tensor(
            x_val, dtype=torch.float32
        )
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32).view(
            -1, 1
        ), torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Create Tensor datasets
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)

        criterion = nn.SmoothL1Loss()  # Using Smooth L1 Loss
        optimizer = optim.Adamax(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.001,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # Early stopping initialization
        best_loss = np.inf
        counter = 0

        # Tracking loss and epochs
        self.loss_history_validation = []
        self.loss_history_train = []
        self.epoch_list = []
        self.best_loss_history_val = []

        # Create DataLoaders
        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=self.batch_size,
        )
        val_loader = DataLoader(val_data, shuffle=True, batch_size=self.batch_size)

        for epoch in range(self.epochs):  # Number of epochs
            with torch.set_grad_enabled(True):
                loss_vals_train = []
                for inputs, labels in train_loader:
                    inputs.requires_grad_(True)

                    optimizer.zero_grad()
                    outputs = self(inputs)

                    loss = criterion(outputs, labels)
                    loss_vals_train.append(loss.data.item())

                    loss.backward()
                    optimizer.step()

                avgloss_train = np.average(loss_vals_train)
                self.loss_history_train.append(avgloss_train)

            # Validation step
            loss_vals = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    val_outputs = self(inputs)
                    val_loss = criterion(val_outputs, labels)
                    loss_vals.append(val_loss.data.item())

            avgloss_valid = np.average(loss_vals)
            self.loss_history_validation.append(avgloss_valid)

            self.epoch_list.append(epoch)

            scheduler.step(avgloss_valid)

            # Early stopping
            if avgloss_valid < best_loss:
                best_loss = avgloss_valid
                self.best_loss_history_val.append(best_loss)
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Final loss: {avgloss_valid}")
        return self


# Quantile regression architecture made by Rosselini et.al. 2024
class QuantileRegressionNet(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=100, dropout=False, batch_norm=False
    ):
        super(QuantileRegressionNet, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        if dropout:
            self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc3(x)
        return x


class QuantileRegressionNN:
    def __init__(
        self,
        quantiles=[0.5],
        lr=1e-3,
        epochs=100,
        batch_size=32,
        dropout=0,
        normalize=True,
        weight_decay=0,
        hidden_size=100,
        batch_norm=True,
        gamma=0.999,
        step_size=10,
        random_state=None,
        epoch_model_tracking=False,
        verbose=False,
        use_gpu=True,
        undo_quantile_crossing=False,
        drop_last=False,
        running_batch_norm=False,
        train_first_batch_norm=False,
    ):
        self.quantiles = quantiles
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.normalize = normalize
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.random_state = random_state
        self.batch_norm = batch_norm
        self.gamma = gamma
        self.step_size = step_size
        self.epoch_model_tracking = epoch_model_tracking
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.undo_quantile_crossing = undo_quantile_crossing
        self.drop_last = drop_last
        self.running_batch_norm = running_batch_norm
        self.train_first_batch_norm = train_first_batch_norm
        if random_state:
            torch.manual_seed(random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X)
        y = np.array(y)
        if self.normalize:
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
            self.x_range = self.x_max - self.x_min
            self.x_range[self.x_range == 0] = 1
            self.y_min = y.min()
            self.y_max = y.max()
            X = (X - self.x_min) / self.x_range
            y = (y - self.y_min) / (self.y_max - self.y_min)

            if y_val is not None:
                y_val = (y_val - self.y_min) / (self.y_max - self.y_min)

        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.net = QuantileRegressionNet(
            input_size=X.shape[1],
            output_size=len(self.quantiles),
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            batch_norm=self.batch_norm,
        )

        self.net.to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last
        )

        y_pred_val_across_epochs = []
        self.saved_models = []
        for epoch in range(self.epochs):
            epoch_losses = []
            if self.running_batch_norm and not (self.train_first_batch_norm):
                for m in self.net.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
            elif self.running_batch_norm and epoch > 0:
                for m in self.net.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()

            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.net(X_batch)
                loss = 0.0
                for i, q in enumerate(self.quantiles):
                    error = y_batch - y_pred[:, i]
                    if q == "mean":
                        loss += torch.square(error).mean()
                    else:
                        loss += torch.max((q - 1) * error, q * error).mean()
                with torch.no_grad():
                    epoch_losses.append(loss.detach().cpu().numpy())
                loss.backward()
                self.optimizer.step()

            if X_val is not None and y_val is not None:
                preds = self.predict(X_val, use_seed=False, undo_normalization=False)
                loss_val = 0.0
                for i, q in enumerate(self.quantiles):
                    error = y_val - preds[i]
                    if q == "mean":
                        loss_val += np.square(error).mean()
                    else:
                        loss_val += np.maximum((q - 1) * error, q * error).mean()
                if self.verbose:
                    print(
                        f"Epoch: {epoch} \t Train Loss: {np.mean(epoch_losses)} Validation Loss: {loss_val}"
                    )
                y_pred_val_across_epochs.append(preds)

            self.scheduler.step()
            if self.epoch_model_tracking:
                self.saved_models.append(deepcopy(self.net.state_dict()))

        if y_pred_val_across_epochs != []:

            self.y_pred_val_across_epochs = np.stack(y_pred_val_across_epochs)

            # self.net.train()

    def predict(self, X, ensembling=None, use_seed=True, undo_normalization=True):
        if use_seed and self.random_state:
            torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        if self.x_min is not None and self.x_max is not None:
            X = (X - self.x_min) / self.x_range
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if ensembling and not (self.epoch_model_tracking):
            self.net.train()
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

            y_pred = list()

            with torch.no_grad():
                for t in range(ensembling):
                    X_out = self.net(X)
                    y_pred.append(X_out.cpu().squeeze())

            y_pred = torch.stack(y_pred)

        elif ensembling and self.epoch_model_tracking:
            y_pred = list()
            for state_dict in self.saved_models[-ensembling:]:
                self.net.load_state_dict(state_dict)
                self.net.eval()
                with torch.no_grad():
                    X_out = self.net(X)
                    y_pred.append(X_out.cpu().squeeze())

            y_pred = torch.stack(y_pred)

        else:
            self.net.eval()

            with torch.no_grad():
                y_pred = self.net(X)

        y_pred = y_pred.detach().cpu().numpy()
        if self.y_min is not None and self.y_max is not None and undo_normalization:
            y_pred = y_pred * (self.y_max - self.y_min) + self.y_min

        if self.undo_quantile_crossing and ensembling:
            y_pred[:, :, 0][y_pred[:, :, 0] > y_pred[:, :, -1]] = (
                0.5 * y_pred[:, :, -1][y_pred[:, :, 0] > y_pred[:, :, -1]]
                + 0.5 * y_pred[:, :, 0][y_pred[:, :, 0] > y_pred[:, :, -1]]
            )
            y_pred[:, :, -1][y_pred[:, :, 0] > y_pred[:, :, -1]] = (
                0.5 * y_pred[:, :, -1][y_pred[:, :, 0] > y_pred[:, :, -1]]
                + 0.5 * y_pred[:, :, 0][y_pred[:, :, 0] > y_pred[:, :, -1]]
            )
        elif self.undo_quantile_crossing:
            y_pred[:, 0][y_pred[:, 0] > y_pred[:, -1]] = (
                0.5 * y_pred[:, -1][y_pred[:, 0] > y_pred[:, -1]]
                + 0.5 * y_pred[:, 0][y_pred[:, 0] > y_pred[:, -1]]
            )
            y_pred[:, -1][y_pred[:, 0] > y_pred[:, -1]] = (
                0.5 * y_pred[:, -1][y_pred[:, 0] > y_pred[:, -1]]
                + 0.5 * y_pred[:, 0][y_pred[:, 0] > y_pred[:, -1]]
            )

        self.net.train()

        if ensembling:
            return np.moveaxis(y_pred, [0, 1, 2], [2, 1, 0])
        else:
            return np.moveaxis(y_pred, [0], [1]).T

    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        alpha = self.quantiles[0] + 1 - self.quantiles[-1]

        preds = self.predict(X)

        return np.mean(interval_score_loss(preds[-1], preds[0], y, alpha))

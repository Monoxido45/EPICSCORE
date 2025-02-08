######## Code for Predictive models

# used torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# variational gp package
import gpytorch

# BART package
import pymc as pm
import pymc_bart as pmb
from pymc_bart.split_rules import ContinuousSplitRule, OneHotSplitRule
import arviz as az

# numpy and sklearn based packages
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import scipy.stats as st
from scipy.stats import norm
from scipy.stats import gamma
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from tqdm import tqdm
import gpytorch


#### Mixture density network models
# General Base Mixture Density Network architecture
class MDN_base(nn.Module):
    def __init__(self, input_shape, num_components, hidden_layers, dropout_rate=0.4):
        """
        Flexible MDN architecture

        Input: (i) input_shape (int): Input dimension.
               (ii) num_components (int): Number of mixture components.
               (iii) hidden_layers (list): List containing the number of neurons per hidden layer.
               (iv) dropout_rate (float): Dropout rate applied to each layer. Default is 0.4.
        """
        super(MDN_base, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Creating hidden layers dinamically
        prev_units = input_shape
        for units in hidden_layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.batch_norms.append(nn.BatchNorm1d(units))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_units = units

        self.fc_out = nn.Linear(prev_units, num_components * 3)

    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = F.relu(layer(x))
            # x = torch.tanh(layer(x))
            x = bn(x)
            x = dropout(x)
        x = self.fc_out(x)
        return x


# For gaussian Mixture Density
def gaussian_pdf(y, mu, sigma):
    return torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


# For gamma Mixture Density
def gamma_pdf(y, mu, sigma):
    alpha = (mu**2) / (sigma**2)
    beta = mu / (sigma**2)

    return ((beta ** (alpha)) * y ** (alpha - 1) * torch.exp(-beta * y)) / torch.exp(
        torch.lgamma(alpha)
    )


# Mixture Density Network general model:
class MDN_model(BaseEstimator):
    """
    Mixture Density Network model.
    """

    def __init__(
        self,
        input_shape,
        num_components=5,
        hidden_layers=[64],
        dropout_rate=0.4,
        base_model_type=None,
        alpha=None,
        normalize_y=False,
        log_y=False,
        type="gaussian",
    ):
        """
        Input: (i) input_shape: Input dimension.
               (ii) num_components: Number of mixture components.
               (iii) hidden_layers: List containing the number of neurons in each hidden layer. The length of the list determines the amount of hidden layers in the model.
               (iv) dropout_rate: Dropout Rate for each hidden layer. Default is 0.4.
               (v) base_model_type: Type of base model to be fitted. Default is None.
        """
        self.input_shape = input_shape
        self.num_components = num_components
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        # defining base model according to parameters
        self.model = MDN_base(
            self.input_shape, self.num_components, self.hidden_layers, self.dropout_rate
        )
        self.base_model_type = base_model_type
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.log_y = log_y
        self.type = type

    ##### auxiliary functions
    # MDN loss
    @staticmethod
    def mdn_loss(pi, mu, sigma, y_true, type="gaussian"):
        y_true = y_true.view(-1, 1)
        if type == "gaussian":
            result = torch.sum(pi * gaussian_pdf(y_true, mu, sigma), dim=1)
        elif type == "gamma":
            result = torch.sum(pi * gamma_pdf(y_true, mu, sigma), dim=1)
        loss = -torch.log(result + 1e-8)  # numerical stability
        return torch.mean(loss)

    # Mixture coeficient obtention
    def get_mixture_coef(self, y_pred):
        pi = F.softmax(y_pred[:, : self.num_components], dim=1)
        # pi = pi / pi.sum(dim=1, keepdim=True)
        if self.type == "gaussian":
            mu = y_pred[:, self.num_components : 2 * self.num_components]
            sigma = F.softplus(y_pred[:, 2 * self.num_components :])

        elif self.type == "gamma":
            mu = F.softplus(y_pred[:, self.num_components : 2 * self.num_components])
            sigma = F.softplus(y_pred[:, 2 * self.num_components :])
        # sigma = torch.exp(y_pred[:, 2 * num_components:])
        return pi, mu, sigma

    def fit(
        self,
        X,
        y,
        proportion_train=0.7,
        epochs=500,
        lr=0.001,
        gamma=0.99,
        batch_size=32,
        step_size=5,
        weight_decay=0,
        verbose=0,
        patience=30,
        scale=False,
        random_seed_split=0,
        random_seed_fit=1250,
    ):
        """
        Fit MDN model.

        Input: (i) X (np.ndarray or torch.Tensor): Training input data.
               (ii) y (np.ndarray or torch.Tensor): Training target data.
               (iii) proportion_train (float): Proportion of data to be used for training (the rest for validation). Default is 0.7.
               (iv) epochs (int): Number of epochs for training. Default is 500.
               (v) lr (float): Learning rate for the optimizer. Default is 0.001.
               (vi) gamma (float): Gamma value for scheduler. Default is 0.99
               (vii) batch_size (int): Batch size. Default is 32.
               (viii) step_size (int): Step size for scheduler. Default is 5.
               (ix) weight_decay (float): Optimizer weight decay parameter. Default is 0.
               (x) verbose (int): Verbosity level (0, 1, or 2). If set to 0, does not print anything, if 1, prints the average loss of each epoch and if 2, prints the model learning curve at end of fitting.
               (xi) patience (int): Number of epochs with no improvement to trigger early stopping. Default is 30.
               (xii) scale (bool): Whether to scale or not the data. Default is False.
               (xiii) random_seed_split (int): Random seed fixed to perform data splitting. Default is 0.
               (xiv) random_seed_fit (int): Random seed fixed to model fitting. Default is 1250.

        Output: (i) fitted MDN_model object
        """
        self.optimizer = optim.Adamax(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # Splitting data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=1 - proportion_train, random_state=random_seed_split
        )

        # checking if scaling is needed
        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
            x_train = self.scaler.transform(x_train)
            x_val = self.scaler.transform(x_val)
            self.scale = True
        else:
            self.scale = False

        # checking if scaling the response is needed
        if self.normalize_y or self.log_y:
            if self.log_y:
                y_train, self.lmbda = st.boxcox(y_train)
                y_val = st.boxcox(y_val, lmbda=self.lmbda)
                if self.normalize_y:
                    self.y_scaler = StandardScaler()
                    self.y_scaler.fit(y_train.reshape(-1, 1))

                    y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
                    y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

            elif self.normalize_y:
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(y_train.reshape(-1, 1))
                y_train = self.y_scaler.transform(y_train.reshape(-1, 1)).flatten()
                y_val = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()

        # checking if is an instance of numpy
        if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
            x_train, x_val = (
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(x_val, dtype=torch.float32),
            )
            y_train, y_val = (
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
                torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            )

        # Training and validation
        train_dataset = TensorDataset(
            x_train.clone().detach().float(),
            (
                y_train.clone().detach().float()
                if isinstance(y_train, torch.Tensor)
                else torch.tensor(y_train, dtype=torch.float32)
            ),
        )
        val_dataset = TensorDataset(
            x_val.clone().detach().float(),
            (
                y_val.clone().detach().float()
                if isinstance(y_val, torch.Tensor)
                else torch.tensor(y_val, dtype=torch.float32)
            ),
        )

        # Setting batch size
        batch_size_train = int(proportion_train * batch_size)
        batch_size_val = int((1 - proportion_train) * batch_size)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

        losses_train = []
        losses_val = []

        # early stopping
        best_val_loss = float("inf")
        counter = 0

        torch.manual_seed(random_seed_fit)
        torch.cuda.manual_seed(random_seed_fit)
        # Training loop
        for epoch in tqdm(range(epochs), desc="Fitting MDN model"):
            self.model.train()
            train_loss_epoch = 0

            # Looping through batches
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output_train = self.model(x_batch)  # Network output
                pi_train, mu_train, sigma_train = self.get_mixture_coef(output_train)
                loss_train = self.mdn_loss(pi_train, mu_train, sigma_train, y_batch)
                loss_train.backward()
                self.optimizer.step()
                train_loss_epoch += loss_train.item()

            # Computing validation loss
            self.model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output_val = self.model(x_batch)  # Network output
                    pi_val, mu_val, sigma_val = self.get_mixture_coef(output_val)
                    loss_val = self.mdn_loss(
                        pi_val, mu_val, sigma_val, y_batch, type=self.type
                    )
                    val_loss_epoch += loss_val.item()

            # average loss by epoch
            train_loss_epoch /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            losses_train.append(train_loss_epoch)
            losses_val.append(val_loss_epoch)

            self.scheduler.step()

            if verbose == 1:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}"
                )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping in epoch {epoch} with best validation loss: {best_val_loss:.4f}"
                    )
                    break

        if verbose == 2:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            epochs_completed = len(losses_train)
            ax.set_xlim(0, epochs_completed)

            ax.plot(
                range(epochs_completed), losses_train, label="Train Loss", color="blue"
            )
            ax.plot(
                range(epochs_completed),
                losses_val,
                label="Validation Loss",
                color="green",
                linestyle="--",
            )
            plt.legend(loc="upper right")
            plt.show()

        return self

    def predict_mcdropout(self, x, num_samples=100, return_mean=False):
        """
        Make predictions with MC Dropout.

        Input:
            (i) x: Input data.
            (ii) num_samples: Number of Monte Carlo samples. Default is 100.
            (iii) return_mean: Whether to return the mean of the predictions. Default is False.

        Output:
            (i) Tuple containing the stacked tensors of the predictions (pi, mu, sigma) or their means if return_mean is True.
        """
        if isinstance(x, np.ndarray):
            if self.scale:
                x = self.scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32)

        # original_mode = self.model.training
        self.model.eval()
        self.model.train()

        pi_predictions = []
        mu_predictions = []
        sigma_predictions = []

        for _ in range(num_samples):
            # model.train()
            with torch.no_grad():
                pred = self.model(x)
                pred_pi, pred_mu, pred_sigma = self.get_mixture_coef(pred)
                pi_predictions.append(pred_pi)
                mu_predictions.append(pred_mu)
                sigma_predictions.append(pred_sigma)
            # self.model.train(original_mode)

        # stacking predictions
        pi_predictions = torch.stack(pi_predictions)
        mu_predictions = torch.stack(mu_predictions)
        sigma_predictions = torch.stack(sigma_predictions)

        if return_mean:
            pi_mean = torch.mean(pi_predictions, dim=0)
            mu_mean = torch.mean(mu_predictions, dim=0)
            sigma_mean = torch.mean(sigma_predictions, dim=0)

            return pi_mean, mu_mean, sigma_mean

        return pi_predictions, mu_predictions, sigma_predictions

    def predict(self, X_test):
        """
        Make predictions with MDN base model.

        Input: (i) X_test (np.ndarray or torch.Tensor): Input data.

        Output: (i) Numpy array containing the type of prediction selected for
            base model.
        """
        if self.scale:
            X_test = self.scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32).clone().detach().float()
        self.model.eval()
        with torch.no_grad():
            pred_test = self.model(X_test)
            pi, mu, sigma = self.get_mixture_coef(pred_test)
        if self.base_model_type == "regression":
            mean_test = self.mixture_mean(pi, mu).numpy()
            return mean_test
        elif self.base_model_type == "quantile":
            alphas = [self.alpha / 2, 1 - (self.alpha / 2)]
            quantiles_test = self.mixture_quantile(alphas, pi, mu, sigma)
            return quantiles_test

    def mixture_quantile(self, alphas, pi, mu, sigma, rng=0, N=1000):
        """
        Compute quantiles for each mixture component.

        Input:
            (i) alphas (list): List of quantiles (e.g., [0.1, 0.5, 0.9]).
            (ii) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (iii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iv) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (v) rng: Fixed random Seed or generator. Default is 0.
            (vi) N (int): Number of samples to generate per mixture.

        Output:
            (i) np.ndarray: Quantile matrix of shape (n_samples, len(alphas)).
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_sample, _ = pi.shape
        n_alphas = len(alphas)

        # N samples from each mixture
        samples = self.sample_from_mixture(pi, mu, sigma, N=N)

        # Quantile computation
        quantile_matrix = np.zeros((n_sample, n_alphas))
        for j, alpha in enumerate(alphas):
            quantile_matrix[:, j] = np.quantile(samples, alpha, axis=1)

        return quantile_matrix

    # Computing means using the mixture parameters
    @staticmethod
    def mixture_mean(pi, mu):
        """
        Compute mean of the Mixture density.

        Input:
            (i) pi (torch.Tensor): Mixture weights of shape (n_observations, n_components).
            (ii) mu (torch.Tensor): Means of the mixture components of shape (n_observations, n_components).

        Output:
            (i) torch.Tensor: Mixture means for each observation.
        """
        # Compute mean of mixture: weighted average of the components means
        mean = torch.sum(pi * mu, dim=1)
        return mean

    @staticmethod
    def mixture_cumulative(y, pi, mu, sigma, type="gaussian"):
        pi = np.array(pi)
        mu = np.array(mu)
        sigma = np.array(sigma)
        n_y = len(y)
        n = len(pi)
        cumulative_matrix = np.zeros((n, n_y))

        for i in range(n):
            if type == "gaussian":
                cumulative_matrix[i] = np.sum(pi[i] * norm.cdf(y, mu[i], sigma[i]))
            elif type == "gamma":
                alpha = (mu[i] ** 2) / (sigma[i] ** 2)
                beta = (mu[i]) / (sigma[i] ** 2)

                cumulative_matrix[i] = np.sum(
                    pi[i] * gamma.cdf(y, a=alpha, scale=1 / beta)
                )

        return cumulative_matrix

    def sample_from_mixture(self, pi, mu, sigma, rng=0, N=1):
        """
        Generates samples from the mixture network model for each observed sample x.

        Input:
            (i) pi (np.ndarray): Mixture weights of shape (n_samples, n_components).
            (ii) mu (np.ndarray): Means of the components of shape (n_samples, n_components).
            (iii) sigma (np.ndarray): Standard deviations of the components of shape (n_samples, n_components).
            (iv) rng: Fixed random Seed or generator. Default is 0.
            (v) N (int): Number of samples per mixture.

        Output:
            (i) np.ndarray: Generated samples, of shape (n_samples, N).
        """
        # fixing seed if a number is passed
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

        # Ensures that pi, mu, and sigma are numpy arrays
        pi = np.asarray(pi)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)

        n_samples, n_comp = pi.shape

        # Normalize the weights to ensure they sum to 1
        pi /= np.sum(pi, axis=1, keepdims=True)

        # Repeat the weights for all samples
        pi_cumsum = np.cumsum(pi, axis=1)  # Cumulative sum for sampling
        random_vals = rng.random((n_samples, N))  # Random values between 0 and 1

        # Determine the chosen components for each sample
        components = (random_vals[..., None] < pi_cumsum[:, None, :]).argmax(axis=2)

        # Select the means and standard deviations of the chosen components
        chosen_mu = np.take_along_axis(mu, components, axis=1)
        chosen_sigma = np.take_along_axis(sigma, components, axis=1)

        if self.type == "gaussian":
            # Generate normal samples
            samples = rng.normal(loc=chosen_mu, scale=chosen_sigma)
        elif self.type == "gamma":
            alpha = (mu**2) / (sigma**2)
            beta = mu / (sigma**2)
            samples = rng.gamma(shape=alpha, scale=1 / beta)

        return samples

    def mdn_generate(self, pi, mu, sigma, rng=0):
        """
        Generates samples from the predictive distribution of the MDN using the monte-carlo parameter samples.

        Input:
        (i) pi (torch.Tensor): Tensor of MC dropout mixture probabilities for each sample (n_mcdropout, n_samples, n_components).
        (ii) mu (torch.Tensor): Tensor of MC dropout mixture means for each sample (n_mcdropout, n_samples, n_components).
        (iii) sigma (torch.Tensor): Tensor of MC dropout mixture standard deviations for each sample (n_mcdropout, n_samples, n_components).
        (iv) rng: Fixed random Seed or generator. Default is 0.

        Output:
        (i) samples (np.ndarray): Generated samples from the mixture of normals (n_samples, n_mcdropout).
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(42)

        # Converting to numpy
        pi_np = pi.detach().numpy()
        mu_np = mu.detach().numpy()
        sigma_np = sigma.detach().numpy()

        n_mc, n_obs, n_comp = pi.shape
        sample = np.zeros((n_obs, n_mc))

        for i in range(n_obs):
            for j in range(n_mc):
                # normalizing pi
                pi_i = pi_np[j, i, :]
                pi_i = pi_i / pi_i.sum()

                # sampling based on component
                component = rng.choice(n_comp, size=1, p=pi_i)

                if self.type == "gaussian":
                    sample[i, j] = rng.normal(
                        mu_np[j, i, component[0]], sigma_np[j, i, component[0]]
                    )
                elif self.type == "gamma":
                    alpha = mu_np[j, i, component[0]] ** 2 / (
                        sigma_np[j, i, component[0]] ** 2
                    )
                    beta = mu_np[j, i, component[0]] / (
                        sigma_np[j, i, component[0]] ** 2
                    )

                    sample[i, j] = rng.gamma(shape=alpha, scale=1 / beta)

        return sample

    def mixture_cdf(self, sample, scores):
        """
        Computes the CDF of the mixture of normal distributions for each mixture and score
        using samples generated by Monte Carlo.

        Input:
        (i) sample (torch.Tensor or np.ndarray): Tensor of shape (n_samples, n_mcdropout),
        where n_samples is the number of samples, n_mcdropout is the number of mixtures generated by MC Dropout.
        (ii) scores (torch.Tensor or np.ndarray): Tensor of shape (n_samples,) with the scores for which to compute the CDF.

        Output:
        (i) cdf_values (torch.Tensor): Tensor of shape (n_samples,) containing the CDF values
        for each mixture and score.
        """

        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample)
        if isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        if self.log_y:
            scores = st.boxcox(scores, lmbda=self.lmbda)
            if self.normalize_y:
                scores = self.y_scaler.transform(scores.reshape(-1, 1)).flatten()
        elif self.normalize_y:
            scores = self.y_scaler.transform(scores.reshape(-1, 1)).flatten()

        n_samples, n_mixtures = sample.shape

        cdf_values = torch.zeros((n_samples,))

        for j in range(n_samples):
            score = scores[j]

            cdf_values[j] = torch.sum(sample[j, :] <= score).float() / n_mixtures

        return cdf_values

    def mixture_ppf(self, samples, probs):
        """
        Computes the percentiles (quantiles) for each mixture and given probability
        using samples generated by Monte Carlo, with linear interpolation.

        Input:
        (i) samples (torch.Tensor): Tensor of shape (n_samples, n_mcdropouts),
        where n_samples is the number of samples, n_mcdropouts is the number of MC samples.
        (ii) probs (torch.Tensor): Tensor of shape (len(probs),) with the probabilities for which to compute the quantile.

        Output:
        (i) quantiles (torch.Tensor): Tensor of shape (n_samples, len(probs)) containing the quantiles
        for each mixture and probability.
        """
        if isinstance(samples, np.ndarray):
            samples = torch.tensor(samples, dtype=torch.float32)
        if isinstance(probs, np.ndarray):
            probs = torch.tensor(probs, dtype=torch.float32)

        n_samples, _ = samples.shape
        n_probs = len(probs)

        quantiles = torch.zeros((n_samples, n_probs), dtype=torch.float32)

        for j in range(n_samples):
            for k, prob in enumerate(probs):

                quantiles[j, k] = np.quantile(samples[j, :], prob)

        return quantiles


#### GP models
# Basic GP model
class GP_model(BaseEstimator):
    """
    Gaussian Process model.

    Gaussian Process regressor using sklearn GP implementation.
    """

    def __init__(self, kernel=None, normalize_y=True, log_y=False):
        """
        Input:
        (i) kernel: Kernel specifying the covariance structure of the GP. Default is None.
        (ii) normalize_y: Whether to normalize the target variable. Default is True.
        (iii) log_y: Whether to apply log transformation to the target variable. Default is False.

        Output:
        (i) Initialized GP_model object.
        """
        self.kernel = kernel
        self.normalize_y = normalize_y
        self.log_y = log_y

    def fit(self, X, y, scale=False, random_state=0, optimizer="fmin_l_bfgs_b"):
        """
        Fit GP model.

        Input:
        (i) X (np.ndarray): Training input data.
        (ii) y (np.ndarray): Training target data.
        (iii) scale (bool): Whether to scale or not the data. Default is False.
        (iv) random_state (int): Random seed fixed to perform model fitting. Default is 0.
        (v) optimizer (str): Optimizer used for optimizing the kernel's parameter. Default is 'fmin_l_bfgs_b'.

        Output:
        (i) fitted GP_model object.
        """
        if scale:
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
            self.scale_x = True
        else:
            self.scale_x = False

        if self.log_y:
            y = np.log(y)

        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=self.normalize_y,
            random_state=random_state,
            optimizer=optimizer,
        )
        self.model.fit(X, y)

    def predict_cdf(self, X_test, y_test):
        """
        Predict conditional CDF of Y given X.

        Input:
            (i) X_test (np.ndarray): Testing feature matrix.
            (ii) y_test (np.ndarray): Testing labels.

        Output:
            (i) np.ndarray: Array with the conditional CDF values of Y|X.
        """
        if self.scale_x:
            X_test = self.scaler_X.transform(X_test)
        if self.log_y:
            y_test = np.log(y_test)

        pred_mean, pred_std = self.model.predict(X_test, return_std=True)

        s_prime = norm.cdf(y_test, loc=pred_mean, scale=pred_std)

        return s_prime

    def predict_params(self, X_test):
        """
        Predict predictive parameters given each X.

        Input:
            (i) X_test (np.ndarray): Testing feature matrix.

        Output:
            (i) Tuple (np.ndarray, np.ndarray): Tuple containing the predictive mean and standard deviation given X.
        """
        if self.scale_x:
            X_test = self.scaler_X.transform(X_test)

        if self.log_y:
            pred_mean, pred_std = self.model.predict(X_test, return_std=True)

        return pred_mean, pred_std


class GP_base(gpytorch.models.ApproximateGP):
    """
    A base class for Gaussian Process models using GPyTorch's ApproximateGP.
    This class implements a Gaussian Process model with a variational inference
    strategy using inducing points.
    """

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GP_base, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Variational GP model
class GPApprox_model(BaseEstimator):
    """
    Gaussian Process Approximate Model.
    """

    def __init__(
        self,
        num_inducing_points=100,
        kernel=None,
        lr_variational=0.01,
        lr_hyperparams=0.01,
        n_epochs=50,
        log_y=False,
    ):
        """
        Input:
        (i) num_inducing_points (int): Number of inducing points to approximate the GP.
        (ii) kernel: Kernel specifying the covariance structure. Default is RBF kernel.
        (iii) lr_variational (float): Learning rate for variational parameters optimizer. Default is 0.1.
        (iv) lr_hyperparams (float): Learning rate for hyperparameter optimizer. Default is 0.01.
        (v) n_epochs (int): Number of training epochs. Default is 50.
        (vi) log_y (bool): Whether to apply log transformation to the target variable. Default is False.

        Output:
        (i) Initialized GPApprox_model object.
        """
        self.num_inducing_points = num_inducing_points
        self.kernel = kernel or gpytorch.kernels.RBFKernel()
        self.lr_variational = lr_variational
        self.lr_hyperparams = lr_hyperparams
        self.n_epochs = n_epochs
        self.model = None
        self.likelihood = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.log_y = log_y

    def fit(
        self,
        X_train,
        y_train,
        batch_size=100,
        random_seed_split=45,
        random_seed_fit=0,
        proportion_train=0.7,
        verbose=2,
        patience=30,
    ):
        """
        Fit the approximate GP model.

        Input:
        (i) X_train: Training input data.
        (ii) y_train: Training target data.
        (iii) batch_size: Batch size for training. Default is 100.
        (iv) random_seed_split: Random seed for data splitting. Default is 45.
        (v) random_seed_fit: Random seed for model fitting. Default is 0.
        (vi) proportion_train: Proportion of data to be used for training. Default is 0.7.
        (vii) verbose: Verbosity level (0, 1, or 2). Default is 2.
        (viii) patience: Number of epochs with no improvement to trigger early stopping. Default is 30.

        Output:
        (i) Trained GPApprox_model object.
        """
        # Splitting data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=1 - proportion_train,
            random_state=random_seed_split,
        )

        # Standardize data
        X_train_standardized = torch.tensor(
            self.scaler_X.fit_transform(X_train), dtype=torch.float32
        )
        y_train_standardized = torch.tensor(
            self.scaler_y.fit_transform(y_train.reshape(-1, 1)), dtype=torch.float32
        ).view(-1)

        # standardize validation data
        X_val_standardized = torch.tensor(
            self.scaler_X.transform(X_val), dtype=torch.float32
        )
        y_val_standardized = torch.tensor(
            self.scaler_y.transform(y_val.reshape(-1, 1)), dtype=torch.float32
        ).view(-1)

        # Select inducing points
        inducing_points = X_train_standardized[
            torch.randperm(X_train_standardized.size(0))[: self.num_inducing_points]
        ]

        # Initialize model and likelihood
        self.model = GP_base(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Optimizers
        variational_ngd_optimizer = gpytorch.optim.NGD(
            self.model.variational_parameters(),
            num_data=y_train_standardized.size(0),
            lr=self.lr_variational,
        )

        hyperparameter_optimizer = torch.optim.Adam(
            [
                {"params": self.model.hyperparameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.lr_hyperparams,
        )

        # Setting batch size
        batch_size_train = int(proportion_train * batch_size)
        batch_size_val = int((1 - proportion_train) * batch_size)

        # DataLoader
        train_dataset = TensorDataset(
            X_train_standardized,
            y_train_standardized,
        )
        val_dataset = TensorDataset(
            X_val_standardized,
            y_val_standardized,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size_train, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True)

        # Train the model
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=y_train_standardized.size(0)
        )

        losses_train = []
        losses_val = []

        # early stopping
        best_val_loss = float("inf")
        counter = 0

        torch.manual_seed(random_seed_fit)
        torch.cuda.manual_seed(random_seed_fit)

        for epoch in tqdm(range(self.n_epochs), desc="Fitting GP model"):
            train_loss_epoch = 0
            self.model.train()
            self.likelihood.train()
            # Looping through batches

            for x_batch, y_batch in train_loader:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()
                output = self.model(x_batch)
                loss_train = -mll(output, y_batch)
                loss_train.backward()
                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()
                train_loss_epoch += loss_train.item()

            self.model.eval()
            self.likelihood.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    output_val = self.model(x_batch)
                    loss_val = -mll(output_val, y_batch)
                    val_loss_epoch += loss_val.item()

            # average loss by epoch
            train_loss_epoch /= len(train_loader)
            val_loss_epoch /= len(val_loader)
            losses_train.append(train_loss_epoch)
            losses_val.append(val_loss_epoch)

            if verbose == 1:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss_epoch:.4f}, Validation Loss: {val_loss_epoch:.4f}"
                )

            # Early stopping
            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(
                        f"Early stopping in epoch {epoch} with best validation loss:  {best_val_loss:.4f}"
                    )
                    break

        if verbose == 2:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

            epochs_completed = len(losses_train)
            ax.set_xlim(0, epochs_completed)

            ax.plot(
                range(epochs_completed), losses_train, label="Train Loss", color="blue"
            )
            ax.plot(
                range(epochs_completed),
                losses_val,
                label="Validation Loss",
                color="green",
                linestyle="--",
            )
            plt.legend(loc="upper right")
            plt.show()

        return self

    def predict_params(self, X_test):
        """
        Predict using the trained model.

        Input:
        (i) X_test: Testing input data.

        Output:
        (i) Tuple (mean predictions, standard deviation predictions).
        """
        self.model.eval()
        self.likelihood.eval()

        X_test_standardized = torch.tensor(
            self.scaler_X.transform(X_test), dtype=torch.float32
        )

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_distribution = self.likelihood(self.model(X_test_standardized))
            pred_mean = predictive_distribution.mean.numpy()
            pred_std = predictive_distribution.stddev.numpy()

        # Undo standardization on mean predictions
        pred_mean = self.scaler_y.inverse_transform(pred_mean.reshape(-1, 1)).flatten()
        pred_std = pred_std * self.scaler_y.scale_

        return pred_mean, pred_std

    def predict_cdf(self, X_test, y_test):
        """
        Predict the conditional CDF of Y|X.

        Input:
        (i) X_test (np.ndarray): Testing input data.
        (ii) y_test (np.ndarray): Testing target data.

        Output:
        (i) np.ndarray: Array with the conditional CDF values of Y|X.
        """
        pred_mean, pred_std = self.predict_params(X_test)
        cdf_values = norm.cdf(y_test, loc=pred_mean, scale=pred_std)

        return cdf_values


#### BART model
class BART_model(BaseEstimator):
    """
    Bayesian Additive Regression Trees model.
    """

    def __init__(
        self,
        m=100,
        type="normal",
        var="heteroscedastic",
        alpha=0.95,
        beta=2,
        response="constant",
        split_prior=None,
        separate_trees=False,
        n_cores=3,
        n_chains=4,
        normalize_y=False,
        progressbar=False,
    ):
        """
        Input:
        (i) m (int, optional): Number of regression trees. Default is 100.
        (ii) type (str, optional): Type of model. Default is "normal".
        (iii) var (str, optional): Type of variance. Default is "heteroscedastic".
        (iv) alpha (float, optional): Alpha parameter for the model. Default is 0.95.
        (v) beta (float, optional): Beta parameter for the model. Default is 2.
        (vi) response (str, optional): Type of response. Default is 'constant'.
        (vii) split_prior (optional): Prior for the split. Default is None.
        (viii) separate_trees (bool, optional): Whether to use separate trees. Default is False.
        (ix) n_cores (int, optional): Number of cores to use. Default is 3.
        (x) n_chains (int, optional): Number of chains for MCMC. Default is 4.
        (xi) normalize_y (bool, optional): Whether to normalize the target variable. Default is False.

        Output:
        (i) Initialized BART_model object.
        """
        self.m = m
        self.type = type
        self.var = var
        self.alpha = alpha
        self.beta = beta
        self.response = response
        self.split_prior = split_prior
        self.separate_trees = separate_trees
        self.n_cores = n_cores
        self.n_chains = n_chains
        self.normalize_y = normalize_y
        if self.normalize_y:
            self.scaler_y = StandardScaler()
        self.progressbar = progressbar

    def fit(self, X, y, n_sample=2000, random_seed=1250):
        """
        Fit the model to the provided data.

        Input:
        (i) X (numpy.ndarray): The input features for the model.
        (ii) y (numpy.ndarray): The target values for the model.
        (iii) n_sample (int, optional): The number of samples to draw in the MCMC process (default is 2000).
        (iv) random_seed (int, optional): The random seed for reproducibility (default is 1250).

        Output:
        (i) self (object): Returns the instance itself.

        Notes:
        This method fits a Bayesian Additive Regression Trees (BART) model to the data.
        The model can handle both homoscedastic and heteroscedastic variance structures.
        Depending on the `type` and `var` attributes of the instance, different models
        are constructed and fitted using PyMC3.

        Attributes:
        (i) model_bart (pm.Model): The fitted BART model.
        (ii) mc_sample (pm.backends.base.MultiTrace): The MCMC samples obtained from fitting the model.
        """

        n_obs = X.shape[0]

        # changing splitting styles according to variables being binary or not
        binary_columns = [i for i in range(X.shape[1]) if np.unique(X[:, i]).size == 2]

        if len(binary_columns) > 0:
            X = X.astype(float)
            self.type_X = True
        else:
            self.type_X = False

        # making splitting list
        split_types = np.repeat(ContinuousSplitRule, repeats=X.shape[1])
        split_types[binary_columns] = OneHotSplitRule
        split_types = split_types.tolist()

        if self.normalize_y:
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # fitting data to BART
        if self.type == "normal":
            if self.var == "heteroscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data("data_X", X)
                    w = pmb.BART(
                        "w",
                        self.X_data,
                        y,
                        m=self.m,
                        shape=(2, n_obs),
                        alpha=self.alpha,
                        beta=self.beta,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )
                    pm.Normal(
                        "y_pred",
                        w[0],
                        np.exp(w[1]),
                        observed=y,
                        shape=self.X_data.shape[0],
                    )

                    # running MCMC in the training sample
                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            elif self.var == "homoscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data("data_X", X)

                    # variance
                    sigma = pm.HalfNormal("sigma", 5)

                    # mu
                    mu = pmb.BART(
                        "mu,",
                        self.X_data,
                        y,
                        m=self.m,
                        alpha=self.alpha,
                        beta=self.beta,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )

                    pm.Normal(
                        "y_pred",
                        mu,
                        sigma,
                        observed=y,
                        shape=self.X_data.shape[0],
                    )

                    # running MCMC in the training sample
                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            self.model_bart = model_bart
        elif self.type == "gamma":
            if self.var == "heteroscedastic":
                with pm.Model() as model_bart:
                    self.X_data = pm.Data(
                        "data_X",
                        X,
                    )
                    Y = y

                    w = pmb.BART(
                        "w",
                        self.X_data,
                        np.log(Y),
                        m=100,
                        size=2,
                        alpha=self.alpha,
                        beta=self.beta,
                        response=self.response,
                        split_prior=self.split_prior,
                        split_rules=split_types,
                        separate_trees=self.separate_trees,
                    )

                    pm.Gamma(
                        "y_pred",
                        mu=pm.math.exp(w[0]),
                        sigma=pm.math.exp(w[1]),
                        shape=self.X_data.shape[0],
                        observed=Y,
                    )

                    self.mc_sample = pm.sample(
                        n_sample,
                        chains=self.n_chains,
                        random_seed=random_seed,
                        cores=self.n_cores,
                        progressbar=self.progressbar,
                    )
            self.model_bart = model_bart
        return self

    def predict_cdf(self, X_test, y_test, random_seed=0):
        """
        Predict the cumulative distribution function (CDF) for the given test data.

        Input:
        (i) X_test (array-like): The input features for the test data.
        (ii) y_test (array-like): The true target values for the test data.
        (iii) random_seed (int, optional): The random seed for reproducibility of the posterior predictive sampling. Default is 0.

        Output:
        (i) cdf_array (numpy.ndarray): An array containing the CDF values for the test data.
        """
        if self.type_X:
            X_test = X_test.astype(float)
        if self.normalize_y:
            y_test = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        with self.model_bart:
            self.X_data.set_value(X_test)
            posterior_predictive_test = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                random_seed=random_seed,
                var_names=["y_pred"],
                predictions=True,
                progressbar=self.progressbar,
            )

        pred_sample = az.extract(
            posterior_predictive_test,
            group="predictions",
            var_names=["y_pred"],
        ).T.to_numpy()

        # subtracting pred_sample from y_test
        cdf_array = np.mean((np.subtract(pred_sample, y_test) <= 0) + 0, axis=0)
        return cdf_array

    def predict_cutoff(self, X_test, t, random_seed=0):
        """
        Predict cutoff values for the given test data.

        Input:
        (i) X_test (array-like): Test data for which the cutoff values are to be predicted.
        (ii) t (float or array-like): Quantile(s) to compute, which should be between 0 and 1 inclusive.
        (iii) random_seed (int, optional): Seed for the random number generator to ensure reproducibility. Default is 0.

        Output:
        (i) cutoffs (array-like): Predicted cutoff values for the test data.
        """
        if self.type_X:
            X_test = X_test.astype(float)

        with self.model_bart:
            self.X_data.set_value(X_test)
            posterior_predictive_test = pm.sample_posterior_predictive(
                trace=self.mc_sample,
                random_seed=random_seed,
                var_names=["y_pred"],
                predictions=True,
                progressbar=self.progressbar,
            )

        pred_sample = az.extract(
            posterior_predictive_test,
            group="predictions",
            var_names=["y_pred"],
        ).T.to_numpy()

        cutoffs = np.quantile(pred_sample, q=t, axis=0)

        # inverse transforming
        if self.normalize_y:
            cutoffs = self.scaler_y.inverse_transform(cutoffs.reshape(-1, 1)).flatten()
        return cutoffs

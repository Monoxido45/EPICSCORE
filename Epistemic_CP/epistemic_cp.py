from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import scipy.stats as st
import torch

from Epistemic_CP.scores import (
    LocalRegressionScore,
    RegressionScore,
    APSScore,
)

from scipy.special import inv_boxcox
from Epistemic_CP.epistemic_models import (
    MDN_model,
    GP_model,
    GPApprox_model,
    BART_model,
)


class ECP_split(BaseEstimator):
    """
    Epistemic Conformal Prediction class.
    """

    def __init__(
        self,
        nc_score,
        base_model,
        alpha,
        is_fitted=False,
        base_model_type=None,
        **kwargs,
    ):
        """
        Input: (i) nc_score (Scores class): Conformity score of choosing. It
                can be specified by instantiating a conformal score class based on the Scores basic class.
               (ii) base_model (BaseEstimator class): Base model with fit and predict methods to be embedded in the conformity score class.
               (iii) alpha (float): Float between 0 and 1 specifying the miscoverage level of resulting prediction region.
               (iv) base_model_type (bool): Boolean indicating whether the base model ouputs quantiles or not. Default is False.
               (v) is_fitted (bool): Whether the base model is already fitted or not. Default is False.
               (vi) **kwargs: Additional keyword arguments passed to fit base_model.
        """
        self.base_model_type = base_model_type
        self.is_fitted = is_fitted
        if ("Quantile" in str(nc_score)) or (base_model_type == True):
            self.nc_score = nc_score(
                base_model, is_fitted=is_fitted, alpha=alpha, **kwargs
            )
        else:
            self.nc_score = nc_score(base_model, is_fitted=is_fitted, **kwargs)

        # checking if base model is fitted
        self.base_model = self.nc_score.base_model
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit base model embeded in the conformal score class to the training set.
        --------------------------------------------------------

        Input: (i)    X: Training numpy feature matrix
            (ii)   y: Training label array

        Output: LocartSplit object
        """
        self.nc_score.fit(X, y)
        return self

    def calib(
        self,
        X_calib,
        y_calib,
        random_seed=1250,
        epistemic_model="MC_dropout",
        scale=False,
        num_components=5,
        hidden_layers=[64, 64],
        dropout_rate=0.5,
        random_seed_split=0,
        random_seed_fit=45,
        split_calib=True,
        epistemic_test_thres=2000,
        N_samples_MC=500,
        kernel=None,
        normalize_y=False,
        log_y=False,
        type="gaussian",
        ensemble=False,
        n_cores=6,
        progress=False,
        **kwargs,
    ):
        """
        Calibrate conformity score using Predictive distribution.
        --------------------------------------------------------

        Input: (i) X_calib (np.ndarray): Calibration numpy feature matrix
               (ii) y_calib (np.ndarray): Calibration label array
               (iii) random_seed (int): Random seed used for data splitting for fit epistemic modeling of the conformal scores.
               (iv) epistemic_model (str): String indicating which predictive modeling approach to use. Options are: "BART", "GP_simple", "GP_variational", "MC_dropout". Default is "MC_dropout".
               (v) scale (bool): Whether to scale the X features or not. Default is False
               (vi) num_components (int): Number of components for the MDN model. Default is 5.
               (vii) hidden_layers (list): List with the amount of neural units per hidden layer for MDN model. Default is [64, 64].
               (viii) dropout_rate (float): Dropout Rate for MDN model. Default is 0.5.
               (ix) kernel (object): Kernel passed to GP simple model. Default is None (sklearn default).
               (x) normalize_y (bool): Whether to normalize the conformity score. Default is False.
               (xi) log_y (bool): Whether to use boxcox transformation on conformity score. Available only for non-negative conformal score. Default is False.
               (xii) type (str): Type of base distribution used in MDN or BART. Options are "normal", "gamma" and "gaussian". Default is "gaussian".
               (xiii) ensemble (bool): Whether the base model outputs three statistics (quantiles and median) or not. Default is False.
               (xiv) n_cores (int): Number of cores to use for parallel processing. Default is 6.
               (xv) progress (bool): Whether to print BART MCMC progress
               (xvi) **kwargs: Additional keyword arguments passed to epistemic model fitting step.
        Output: Vector of cutoffs.
        """
        # computing the scores
        scores = self.nc_score.compute(X_calib, y_calib, ensemble=ensemble)

        if X_calib.shape[0] >= epistemic_test_thres:
            epistemic_test_size = 1000 / X_calib.shape[0]
        else:
            epistemic_test_size = 0.3

        # splitting calibration into a training set and a cutoff set
        if split_calib:
            (
                X_calib_train,
                X_calib_test,
                scores_calib_train,
                scores_calib_test,
            ) = train_test_split(
                X_calib,
                scores,
                test_size=epistemic_test_size,
                random_state=random_seed,
            )
        else:
            (
                X_calib_train,
                X_calib_test,
                scores_calib_train,
                scores_calib_test,
            ) = (
                X_calib,
                X_calib,
                scores,
                scores,
            )

        # fitting epistemic model
        if epistemic_model == "MC_dropout":
            self.epistemic_obj = MDN_model(
                input_shape=X_calib.shape[1],
                num_components=num_components,
                dropout_rate=dropout_rate,
                hidden_layers=hidden_layers,
                normalize_y=normalize_y,
                log_y=log_y,
                type=type,
            )

            # fitting the epistemic model to data
            self.epistemic_obj.fit(
                X_calib_train,
                scores_calib_train,
                scale=scale,
                random_seed_fit=random_seed_fit,
                random_seed_split=random_seed_split,
                **kwargs,
            )

            # monte carlo dropout predictions for each parameter
            with torch.no_grad():
                pi_prime, mu_prime, sigma_prime = self.epistemic_obj.predict_mcdropout(
                    X_calib_test, num_samples=N_samples_MC
                )
            # Computing new cumulative score s' or s_prime
            sample_s = self.epistemic_obj.mdn_generate(
                pi_prime, mu_prime, sigma_prime, random_seed
            )
            s_prime_calibration = self.epistemic_obj.mixture_cdf(
                sample_s, scores_calib_test
            )

            # converting to numpy
            s_prime_calibration_np = s_prime_calibration.flatten()
            n = s_prime_calibration_np.shape[0]

            self.t_cutoff = np.quantile(
                s_prime_calibration_np, np.ceil((n + 1) * (1 - self.alpha)) / n
            )

        elif epistemic_model == "GP_simple":
            self.epistemic_obj = GP_model(
                kernel=kernel,
                normalize_y=normalize_y,
                log_y=log_y,
            )

            self.epistemic_obj.fit(
                X_calib_train,
                scores_calib_train,
                scale=scale,
                random_state=random_seed_fit,
                **kwargs,
            )

            # Computing new cumulative non-conf score s' or s_prime
            s_prime_calibration = self.epistemic_obj.predict_cdf(
                X_calib_test, y_test=scores_calib_test
            )
            n = s_prime_calibration.shape[0]
            self.t_cutoff = np.quantile(
                s_prime_calibration, np.ceil((n + 1) * (1 - self.alpha)) / n
            )

        elif epistemic_model == "GP_variational":
            # Inicializando o modelo GP variacional
            self.epistemic_obj = GPApprox_model(
                num_inducing_points=kwargs.get("num_inducing_points", 100),
                lr_variational=kwargs.get("lr_variational", 0.1),
                lr_hyperparams=kwargs.get("lr_hyperparams", 0.01),
                n_epochs=kwargs.get("n_epoch", 50),
                log_y=log_y,
            )

            # Treinando o modelo com os dados de calibração
            self.epistemic_obj.fit(
                X_calib_train,
                scores_calib_train,
                batch_size=kwargs.get("batch_size", 100),
                random_seed_fit=random_seed_fit,
                random_seed_split=random_seed_split,
                patience=kwargs.get("patience", 30),
                proportion_train=kwargs.get("proportion_train", 0.7),
                verbose=kwargs.get("verbose", 0),
            )

            # Calculando os novos scores cumulativos não conformes (s_prime)
            s_prime_calibration = self.epistemic_obj.predict_cdf(
                X_calib_test, y_test=scores_calib_test
            )

            # Determinando o ponto de corte t_cutoff com base em quantis
            n = s_prime_calibration.shape[0]
            self.t_cutoff = np.quantile(
                s_prime_calibration, np.ceil((n + 1) * (1 - self.alpha)) / n
            )
        elif epistemic_model == "BART":
            self.epistemic_obj = BART_model(
                m=kwargs.get("m", 100),
                type=kwargs.get("type", "normal"),
                var=kwargs.get("var", "heteroscedastic"),
                alpha=kwargs.get("alpha", 0.95),
                beta=kwargs.get("beta", 2),
                response=kwargs.get("response", "constant"),
                split_prior=kwargs.get("split_prior", None),
                separate_trees=kwargs.get("separate_trees", False),
                n_cores=n_cores,
                normalize_y=normalize_y,
            )

            # Obtaining MCMC samples for BART
            self.epistemic_obj.fit(
                X_calib_train,
                scores_calib_train,
                n_sample=N_samples_MC,
                random_seed=random_seed_fit,
            )

            # computing new cumulative scores
            s_prime_calibration = self.epistemic_obj.predict_cdf(
                X_calib_test, y_test=scores_calib_test, random_seed=random_seed_fit
            )

            # determining new cutoff point
            n = s_prime_calibration.shape[0]
            self.t_cutoff = np.quantile(
                s_prime_calibration, np.ceil((n + 1) * (1 - self.alpha)) / n
            )

        self.epistemic_model = epistemic_model
        return self.t_cutoff

    def predict(
        self,
        X_test,
        N_samples_MC=500,
        random_seed=45,
        ensemble=False,
    ):
        """
        Predict 1 - alpha prediction regions for each test sample using epistemic cutoffs.
        --------------------------------------------------------
        Input: (i) X_test (np.ndarray): Test numpy feature matrix
               (ii) N_samples_MC (int): Number of samples to simulate from MC dropout. Default is 500.
               (iii) random_seed (int): Random seed fixed to generate samples. Used in MC dropout and BART.
               (iv) ensemble (bool): Whether the base model outputs three statistics (quantiles and median) or not. Default is False.

        Output: Prediction regions for each test sample.
        """

        if self.epistemic_model == "MC_dropout":
            # predictions for mc_dropout
            with torch.no_grad():
                pi_test, mu_test, sigma_test = self.epistemic_obj.predict_mcdropout(
                    X_test, num_samples=N_samples_MC
                )
            # computing t_inverse for obtaining region in
            # the original non conf score
            sample_test = self.epistemic_obj.mdn_generate(
                pi_test, mu_test, sigma_test, random_seed
            )

            # boxcox transformation for reg split
            if self.epistemic_obj.log_y:
                t_inverse_test_og = (
                    self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                    .numpy()
                    .flatten()
                )
                # if also normalizing y
                if self.epistemic_obj.normalize_y:
                    # first inverting normalization
                    t_inverse_test = self.epistemic_obj.y_scaler.inverse_transform(
                        t_inverse_test_og.reshape(-1, 1)
                    ).flatten()

                    # then inverting box cox transformation
                    t_inverse_test = inv_boxcox(
                        t_inverse_test,
                        self.epistemic_obj.lmbda,
                    )
                else:
                    # inverting box cox transformation
                    t_inverse_test = inv_boxcox(
                        t_inverse_test_og,
                        self.epistemic_obj.lmbda,
                    )

            # standard normalization
            elif self.epistemic_obj.normalize_y:
                t_inverse_test = self.epistemic_obj.y_scaler.inverse_transform(
                    self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                    .numpy()
                    .flatten()
                    .reshape(-1, 1)
                ).flatten()

            else:
                t_inverse_test = (
                    self.epistemic_obj.mixture_ppf(sample_test, [self.t_cutoff])
                    .numpy()
                    .flatten()
                )

        elif (
            self.epistemic_model == "GP_simple"
            or self.epistemic_model == "GP_variational"
        ):
            # computing t_inverse for gaussian process
            pred_mean_test, pred_std_test = self.epistemic_obj.predict_params(X_test)

            t_inverse_test = st.norm.ppf(
                self.t_cutoff, loc=pred_mean_test, scale=pred_std_test
            )

            if self.epistemic_obj.log_y and self.epistemic_model == "GP_simple":
                t_inverse_test = np.exp(t_inverse_test)

        elif self.epistemic_model == "BART":
            t_inverse_test = self.epistemic_obj.predict_cutoff(
                X_test, t=self.t_cutoff, random_seed=random_seed
            )

        pred = self.nc_score.predict(
            X_test,
            t_inverse_test,
            ensemble=ensemble,
        )
        return pred


# Regression baselines
# Implementing classic Regression-split
class RegressionSplit(BaseEstimator):
    """
    Basic Regression Split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha=0.1, is_fitted=False, **kwargs) -> None:
        super().__init__()
        self.is_fitted = is_fitted
        self.nc_score = RegressionScore(base_model, is_fitted=is_fitted, **kwargs)
        self.alpha = alpha
        self.base_model = self.nc_score.base_model

    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(X_calib, y_calib)
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        return self.nc_score.predict(X_test, self.cutoff)


# Implementing APS
class APSSplit(BaseEstimator):
    """
    Basic APS split class
    ----------------------------------------------------------------
    """

    def __init__(
        self,
        base_model,
        alpha=0.1,
        is_fitted=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.is_fitted = is_fitted
        self.nc_score = APSScore(base_model, is_fitted=is_fitted, **kwargs)
        self.alpha = alpha
        self.base_model = self.nc_score.base_model

    def fit(self, X_train, y_train):
        self.nc_score.fit(X_train, y_train)

    def calibrate(self, X_calib, y_calib):
        res = self.nc_score.compute(
            X_calib,
            y_calib,
        )
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        return self.nc_score.predict(X_test, self.cutoff)


# Implementing weighted regression split
# Local regression split proposed by Lei et al
class LocalRegressionSplit(BaseEstimator):
    """
    Local Regression Split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha, is_fitted=False, **kwargs):
        """
        Input: (i)    base_model: Regression base model with fit and predict methods defined.
               (ii)   alpha: Miscalibration level.
               (iii)  is_fitted: Boolean informing whether the base model is already fitted.
               (iv)   **kwargs: Additional keyword arguments to be passed to the base model.
        """
        self.is_fitted = is_fitted
        self.nc_score = LocalRegressionScore(
            base_model, is_fitted=self.is_fitted, **kwargs
        )
        self.alpha = alpha
        self.base_model = self.nc_score.base_model

    def fit(self, X_train, y_train):
        """
        Fit base model embedded on local regression conformal score to the training set.
        ----------------------------------------------------------------

        Input: (i)    X_train: Training numpy feature matrix.
               (ii)   y_train: Training labels.

        Output: QuantileSplit object.
        """
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib):
        """
        Calibrate local regression score
        ----------------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix.
               (ii)   y_calib: Calibration labels.

        Output: None
        """
        res = self.nc_score.compute(X_calib, y_calib)
        n = X_calib.shape[0]
        self.cutoff = np.quantile(res, q=np.ceil((n + 1) * (1 - self.alpha)) / n)
        return None

    def predict(self, X_test):
        """
        Predict 1 - alpha prediction intervals for each test samples using local regression split cutoff
        ----------------------------------------------------------------

        Input: (i)    X_test: Test numpy feature matrix.

        Output: Prediction intervals for each test sample.
        """
        return self.nc_score.predict(X_test, self.cutoff)


# Mondrian split method proposed by Bostrom et al
class MondrianRegressionSplit(BaseEstimator):
    """
    Mondrian regression split class
    ----------------------------------------------------------------
    """

    def __init__(self, base_model, alpha, is_fitted=False, k=10, **kwargs):
        """
        Input: (i)    base_model: Regression base model with fit and predict methods defined.
               (ii)   alpha: Miscalibration level.
               (iii)  is_fitted: Boolean informing whether the base model is already fitted.
               (iv)   k: Number of bins for splitting the conditional variance (diffulty).
               (v)    **kwargs: Additional keyword arguments to be passed to the base model.
        """
        self.is_fitted = is_fitted
        self.base_model = base_model
        self.k = k
        self.nc_score = RegressionScore(
            self.base_model, is_fitted=self.is_fitted, **kwargs
        )
        self.alpha = alpha

    def fit(self, X_train, y_train, random_seed_tree=550, **kwargs):
        """
        Fit both base model embedded on the regression score to the training set and a Random Forest model to estimate variance in the calibration step.
        ----------------------------------------------------------------
        Input: (i)    X_train: Training numpy feature matrix.
               (ii)   y_train: Train numpy labels.
               (iii)  random_seed_tree: Random Forest random seed for variance estimation.
               (iv)   **kwargs: Additional keyword arguments to be passed to the Random Forest model.
        """
        # fitting the base model
        self.nc_score.fit(X_train, y_train)
        # training RandomForestRegressor for difficulty estimation if base model is not RandomForest
        if not isinstance(self.nc_score.base_model, RandomForestRegressor):
            self.dif_model = (
                RandomForestRegressor(random_state=random_seed_tree)
                .set_params(**kwargs)
                .fit(X_train, y_train)
            )
        else:
            self.dif_model = deepcopy(self.nc_score.base_model)

        return self

    def calibrate(
        self, X_calib, y_calib, split=False, binning_size=0.5, random_state=1250
    ):
        """
        Calibrate conformity scores using mondrian binning.
        ----------------------------------------------------------------

        Input: (i)    X_calib: Calibration numpy feature matrix.
               (ii)   y_calib: Calibration labels.
               (iii)  split: Whether to split the dataset into a binning set and a cutoff set in the binning step. Default is False.
               (iv)   binning_size: Proportion of the splitted dataset destined for binning if split is True. Default is 0.5.
               (v)    random_state: Random seed for splitting the dataset if split is True.

        Output: None
        """
        if split:
            # making the split
            X_score, X_final, y_score, y_final = train_test_split(
                X_calib, y_calib, test_size=1 - binning_size, random_state=random_state
            )
        else:
            X_score, X_final, y_score, y_final = X_calib, X_calib, y_calib, y_calib

        # computing the difficulty score for each X_score
        pred_dif = self.compute_difficulty(X_score)

        # computing vanilla score in held out data
        res = self.nc_score.compute(X_final, y_final)

        # now making local partitions based on variance percentile
        # binning into k percentiles
        alphas = np.arange(1, self.k) / self.k
        self.mondrian_quantiles = np.quantile(pred_dif, q=alphas, axis=0)

        # iterating percentiles to obtain local cutoffs
        # first obtaining interval index by apply function
        new_dif = self.compute_difficulty(X_final)
        int_idx = self.apply(new_dif)
        self.mondrian_cutoffs = np.zeros(self.k)

        # obtaing all cutoffs
        for i in range(0, self.k):
            current_res = res[np.where(int_idx == i)]
            n = current_res.shape[0]
            self.mondrian_cutoffs[i] = np.quantile(
                current_res, q=np.ceil((n + 1) * (1 - self.alpha)) / n
            )
        return None

    def compute_difficulty(self, X):
        """
        Auxiliary function to estimate difficulty (variance) for a given sample.
        ----------------------------------------------------------------

        Input: (i)    X: Calibration numpy feature matrix.

        Output: Difficulty estimates for each sample.
        """
        cart_pred = np.zeros((X.shape[0], len(self.dif_model.estimators_)))
        i = 0
        # computing the difficulty score for each X_score
        for cart in self.dif_model.estimators_:
            cart_pred[:, i] = cart.predict(X)
            i += 1
        # computing variance for each line
        return cart_pred.var(1)

    def apply(self, dif):
        """
        Auxiliary function to obtain bin index for each difficulty estimate.
        ----------------------------------------------------------------

        Input: (i)    dif: Difficulty estimate vector.

        Output: Vector of indices.
        """
        int_idx = np.zeros(dif.shape[0])
        for i in range(dif.shape[0]):
            index = np.where(dif[i] <= self.mondrian_quantiles)[0]
            # first testing if dif is in any interval before the last quantile
            if index.shape[0] >= 1:
                int_idx[i] = index[0]
            else:
                int_idx[i] = self.k - 1
        return int_idx

    def predict(self, X_test):
        """
        Predict 1 - alpha prediction intervals for each test samples using mondrian cutoffs.
        ----------------------------------------------------------------

        Input: (i)    X_test: Test numpy feature matrix.

        Output: Prediction intervals for each test sample.
        """
        # prediciting difficulty
        pred_dif = self.compute_difficulty(X_test)

        # assigning different cutoffs based on difficulty
        # first obtaining interval indexes
        int_idx = self.apply(pred_dif)
        cutoffs = self.mondrian_cutoffs[int_idx.astype(int)]

        return self.nc_score.predict(X_test, cutoffs)

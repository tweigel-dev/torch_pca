"""Incremental Principal Components Analysis for PyTorch."""

# Copyright (c) 2024 Valentin Goldité. All Rights Reserved.
# Inspired from scikit-learn IncrementalPCA (BSD-3-Clause License)

from typing import Optional

import torch
from sklearn.decomposition import IncrementalPCA
from torch import Tensor

from .pca_main import PCA


class IncrementalPCA(PCA):
    """Incremental Principal Component Analysis (IncrementalPCA).

    Memory-efficient PCA that processes data in batches.
    Useful for datasets too large to fit in memory.


    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If None, min(n_samples, n_features).
    whiten : bool, optional
        Whether to whiten the components. Default: False.
    batch_size : int, optional
        Number of samples per batch. If None, inferred as 5 * n_features.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        *,
        whiten: bool = False,
        batch_size: Optional[int] = None,
    ):
        super().__init__(n_components=n_components, whiten=whiten, svd_solver="full")
        self.batch_size = batch_size

        # Additional attributes for incremental fitting
        self.var_: Optional[Tensor] = None
        self.batch_size_: Optional[int] = None
        self.train()

    def fit(self, inputs: Tensor,*, determinist: bool = True) -> "IncrementalPCA":
        """Fit the model with X using batches.

        Parameters
        ----------
        inputs : Tensor
            Training data of shape (n_samples, n_features).


        Returns
        -------
        self : IncrementalPCA
        """
        # Reset state

        if not self.training:
            raise ValueError("Cannot fit model in eval mode. Call `train()` before fitting.")

        self.components_ = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        self.n_samples_ = 0
        self.mean_ = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)
        self.var_ = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)

        n_samples, n_features = inputs.shape
        self.n_features_in_ = n_features

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        # Process in batches
        for start_idx in range(0, n_samples, self.batch_size_):
            end_idx = min(start_idx + self.batch_size_, n_samples)
            X_batch = inputs[start_idx:end_idx]
            self.partial_fit(X_batch)

        return self

    def partial_fit(self, inputs: Tensor) -> "IncrementalPCA":
        """Incremental fit with X as a single batch.

        Parameters
        ----------
        inputs : Tensor
            Training data of shape (n_samples, n_features).


        Returns
        -------
        self : IncrementalPCA
        """
        n_samples, n_features = inputs.shape
        first_pass = not self.components_.any()

        # Determine n_components
        if self.n_components_ is None:
            if not self.components_.any():
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif self.n_components_ > n_features:
            raise ValueError(
                f"n_components={self.n_components_} invalid for "
                f"n_features={n_features}"
            )
        elif self.n_components_ > n_samples and first_pass:
            raise ValueError(
                f"n_components={self.n_components_} must be <= "
                f"batch size {n_samples} on first call"
            )

        # Initialize on first pass
        if first_pass:
            self.n_samples_ = 0
            self.mean_ = torch.zeros(n_features, dtype=inputs.dtype, device=inputs.device)
            self.var_ = torch.zeros(n_features, dtype=inputs.dtype, device=inputs.device)

        # Update mean and variance incrementally
        col_mean, col_var, n_total_samples = self._incremental_mean_var(
            inputs, self.mean_, self.var_, self.n_samples_
        )

        # Center the data
        if self.n_samples_ == 0:
            X_centered = inputs - col_mean
        else:
            col_batch_mean = inputs.mean(dim=0)
            X_centered = inputs - col_batch_mean

            # Combine with previous singular vectors
            mean_correction = torch.sqrt(
                torch.tensor(
                    (self.n_samples_ / n_total_samples) * n_samples,
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
            ) * (self.mean_ - col_batch_mean)

            X_centered = torch.vstack([
                self.singular_values_.reshape(-1, 1) * self.components_,
                X_centered,
                mean_correction.unsqueeze(0),
            ])

        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)

        # Compute explained variance
        explained_variance = S**2 / (n_total_samples - 1)
        total_var = (col_var * n_total_samples).sum()
        explained_variance_ratio = S**2 / total_var

        # Store results
        self.n_samples_ = n_total_samples
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]

        # Compute noise variance
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_:].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)

        return self

    def _incremental_mean_var(
        self, X: Tensor, last_mean: Tensor, last_var: Tensor, last_n: int
    ):
        """Compute incremental mean and variance."""
        n_samples = X.shape[0]
        n_total = last_n + n_samples

        if last_n == 0:
            return X.mean(dim=0), X.var(dim=0, unbiased=False), n_samples

        new_mean = X.mean(dim=0)
        new_var = X.var(dim=0, unbiased=False)

        # Update mean
        updated_mean = (last_n * last_mean + n_samples * new_mean) / n_total

        # Update variance (Welford's online algorithm)
        m_a = last_var * last_n
        m_b = new_var * n_samples
        M2 = m_a + m_b + (last_mean - new_mean) ** 2 * last_n * n_samples / n_total
        updated_var = M2 / n_total

        return updated_mean, updated_var, n_total

    def _svd_flip(self, U: Tensor, Vt: Tensor) -> tuple[Tensor, Tensor]:
        """Flip signs for deterministic output."""
        max_abs_cols = torch.argmax(torch.abs(Vt), dim=1)
        signs = torch.sign(Vt[torch.arange(Vt.shape[0]), max_abs_cols])
        U *= signs
        Vt *= signs.reshape(-1, 1)
        return U, Vt

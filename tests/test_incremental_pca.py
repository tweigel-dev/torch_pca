"""Tests for IncrementalPCA."""

from pathlib import Path
import torch
from torch_pca import PCA
from torch_pca.pca_incremental import IncrementalPCA
import pytest
from sklearn.decomposition import IncrementalPCA as SklearnIncrementalPCA, PCA as SklearnPCA


@pytest.fixture(scope="module")
def X() -> torch.Tensor:
    """Load test data."""
    return torch.load("tests/input_data.pt").to(torch.float32)


@pytest.fixture(scope="module")
def X_large() -> torch.Tensor:
    """Create larger test data for batch processing."""
    return torch.randn(200, 10, dtype=torch.float32)


def test_incremental_pca_basic_fit(X: torch.Tensor) -> None:
    """Test basic fit functionality of IncrementalPCA."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    assert ipca.components_ is not None, "Components should be fitted."
    assert ipca.components_.shape[0] == 3, "Should have 3 components."
    assert ipca.components_.shape[1] == X.shape[1], "Components should match feature dimension."
    assert ipca.n_samples_ == X.shape[0], "Should track number of samples."
    assert ipca.mean_ is not None, "Mean should be computed."
    assert ipca.var_ is not None, "Variance should be computed."


def test_incremental_pca_partial_fit(X: torch.Tensor) -> None:
    """Test partial_fit functionality."""
    ipca = IncrementalPCA(n_components=3)
    
    # Split data into batches
    batch_size = 20
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i + batch_size]
        ipca.partial_fit(batch)
    
    assert ipca.components_ is not None, "Components should be fitted after partial fits."
    assert ipca.components_.shape[0] == 3, "Should have 3 components."
    assert ipca.n_samples_ == X.shape[0], "Should track total number of samples."


def test_incremental_vs_standard_pca(X: torch.Tensor) -> None:
    """Compare IncrementalPCA results with standard PCA."""
    n_components = 3
    
    # Standard PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    X_ipca = ipca.fit_transform(X)
    
    # Results should be similar (not exactly equal due to numerical differences)
    assert torch.allclose(torch.abs(X_pca), torch.abs(X_ipca), rtol=1e-3, atol=1e-3), \
        "IncrementalPCA should produce similar results to standard PCA."
    
    # Explained variance should be similar
    assert torch.allclose(
        ipca.explained_variance_,
        pca.explained_variance_,
        rtol=1e-2,
        atol=1e-3
    ), "Explained variance should be similar."


def test_incremental_pca_batch_size(X_large: torch.Tensor) -> None:
    """Test IncrementalPCA with custom batch size."""
    batch_size = 30
    ipca = IncrementalPCA(n_components=5, batch_size=batch_size)
    ipca.fit(X_large)
    
    assert ipca.batch_size_ == batch_size, "Batch size should be set correctly."
    assert ipca.components_ is not None, "Components should be fitted."
    assert ipca.n_samples_ == X_large.shape[0], "Should process all samples."


def test_incremental_pca_auto_batch_size(X: torch.Tensor) -> None:
    """Test IncrementalPCA with automatic batch size."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    expected_batch_size = 5 * X.shape[1]
    assert ipca.batch_size_ == expected_batch_size, \
        f"Auto batch size should be 5 * n_features = {expected_batch_size}."


def test_incremental_pca_transform(X: torch.Tensor) -> None:
    """Test transform method."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    X_transformed = ipca.transform(X)
    assert X_transformed.shape == (X.shape[0], 3), \
        "Transformed shape should be (n_samples, n_components)."


def test_incremental_pca_fit_transform(X: torch.Tensor) -> None:
    """Test fit_transform method."""
    ipca = IncrementalPCA(n_components=3)
    X_transformed = ipca.fit_transform(X)
    
    assert X_transformed.shape == (X.shape[0], 3), \
        "Transformed shape should be (n_samples, n_components)."
    assert ipca.components_ is not None, "Model should be fitted."


def test_incremental_pca_inverse_transform(X: torch.Tensor) -> None:
    """Test inverse_transform method."""
    ipca = IncrementalPCA(n_components=3)
    X_transformed = ipca.fit_transform(X)
    X_reconstructed = ipca.inverse_transform(X_transformed)
    
    assert X_reconstructed.shape == X.shape, \
        "Reconstructed data should have original shape."
    
    # Reconstruction should be approximate (not exact due to dimensionality reduction)
    reconstruction_error = torch.norm(X - X_reconstructed) / torch.norm(X)
    assert reconstruction_error < 0.5, \
        "Reconstruction error should be reasonable for 3 components."



def test_incremental_pca_n_components_none(X: torch.Tensor) -> None:
    """Test IncrementalPCA with n_components=None."""
    ipca = IncrementalPCA()
    ipca.fit(X)
    
    expected_components = min(X.shape)
    assert ipca.n_components_ == expected_components, \
        f"Should use min(n_samples, n_features) = {expected_components} components."


def test_incremental_pca_invalid_n_components(X: torch.Tensor) -> None:
    """Test IncrementalPCA with invalid n_components."""
    ipca = IncrementalPCA(n_components=1000)
    
    with pytest.raises(ValueError, match="invalid for"):
        ipca.fit(X)


def test_incremental_pca_explained_variance_ratio(X: torch.Tensor) -> None:
    """Test explained variance ratio sums to <= 1."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    ratio_sum = ipca.explained_variance_ratio_.sum()
    assert 0 < ratio_sum <= 1.0, \
        "Explained variance ratio should sum to a value between 0 and 1."


def test_incremental_pca_singular_values(X: torch.Tensor) -> None:
    """Test that singular values are computed."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    assert ipca.singular_values_ is not None, "Singular values should be computed."
    assert ipca.singular_values_.shape[0] == 3, "Should have 3 singular values."
    # Singular values should be in descending order
    assert torch.all(ipca.singular_values_[:-1] >= ipca.singular_values_[1:]), \
        "Singular values should be in descending order."


def test_incremental_pca_noise_variance(X: torch.Tensor) -> None:
    """Test noise variance computation."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    assert ipca.noise_variance_ is not None, "Noise variance should be computed."
    assert ipca.noise_variance_ >= 0, "Noise variance should be non-negative."


def test_incremental_pca_forward_training_mode(X: torch.Tensor) -> None:
    """Test forward method in training mode."""
    ipca = IncrementalPCA(n_components=3)
    ipca.train()
    
    X_transformed = ipca.forward(X)
    
    assert ipca.components_ is not None, "Model should be fitted in training mode."
    assert X_transformed.shape == (X.shape[0], 3), \
        "Transformed shape should be correct."


def test_incremental_pca_forward_eval_mode(X: torch.Tensor) -> None:
    """Test forward method in eval mode."""
    ipca = IncrementalPCA(n_components=3)

    ipca.fit(X)
    ipca.eval()
    
    X_transformed = ipca.forward(X)
    assert X_transformed.shape == (X.shape[0], 3), \
        "Transformed shape should be correct in eval mode."

def test_incremental_pca_fit_eval_mode(X: torch.Tensor) -> None:
    """Test forward method in eval mode."""
    ipca = IncrementalPCA(n_components=3)
    with pytest.raises(ValueError, match="Cannot fit model in eval mode."):
        ipca.eval()
        ipca.fit(X)

    

def test_incremental_pca_state_dict(tmp_path: Path, X: torch.Tensor) -> None:
    """Test saving and loading state dict."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    components_before = ipca.components_.clone()
    
    # Save state
    target_file = tmp_path / "ipca.pt"
    torch.save(ipca.state_dict(), target_file)
    
    # Load state into new model
    ipca_loaded = IncrementalPCA(n_components=3)
    ipca_loaded.load_state_dict(torch.load(target_file), strict=False)
    
    assert torch.equal(ipca_loaded.components_, components_before), \
        "Loaded components should match saved components."


def test_incremental_pca_multiple_partial_fits_consistency(X_large: torch.Tensor) -> None:
    """Test that multiple partial fits maintain consistency."""
    ipca = IncrementalPCA(n_components=3)
    
    # First partial fit
    batch1 = X_large[:30]
    ipca.partial_fit(batch1)
    n_samples_after_batch1 = ipca.n_samples_
    
    # Second partial fit
    batch2 = X_large[30:60]
    ipca.partial_fit(batch2)
    n_samples_after_batch2 = ipca.n_samples_
    ipca2 = IncrementalPCA(n_components=3)
    ipca2.partial_fit(batch2)
    assert n_samples_after_batch1 == 30, "Should track 30 samples after first batch."
    assert n_samples_after_batch2 == 60, "Should track 60 samples after second batch."
    assert not torch.allclose(ipca.components_, ipca2.components_), \
        "Components should differ between models with different training data."


def test_incremental_pca_dtype_preservation(X: torch.Tensor) -> None:
    """Test that dtype is preserved through operations."""
    X_float64 = X.to(torch.float64)
    
    ipca = IncrementalPCA(n_components=3)
    X_transformed = ipca.fit_transform(X_float64)
    
    assert X_transformed.dtype == torch.float64, "Output dtype should match input dtype."
    assert ipca.components_.dtype == torch.float64, "Components dtype should match input."


def test_incremental_pca_device_consistency(X: torch.Tensor) -> None:
    """Test that operations maintain device consistency."""
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X)
    
    assert ipca.components_.device == X.device, "Components should be on same device as input."
    assert ipca.mean_.device == X.device, "Mean should be on same device as input."


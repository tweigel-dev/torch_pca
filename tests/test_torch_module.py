from pathlib import Path
import torch
from torch_pca import PCAModule

import pytest

@pytest.fixture(scope="module")
def X() -> torch.Tensor:
    return torch.load("tests/input_data.pt").to(torch.float32)


def test_pca_as_torch_module(tmp_path: Path, X: torch.Tensor) -> None:
    # Create a random dataset
    # Initialize PCA with 2 components
    pca = PCAModule(n_components=5)
    # Fit and transform the data
    com_before = pca.components_
    X_reduced = pca.forward(X)
    comps_after = pca.components_.clone()
    assert not com_before == comps_after, "The components should be updated after fitting the PCA model."
    target_file = tmp_path/"pca.pt"
    torch.save(pca.state_dict(), target_file)
    loaded_pca_state = torch.load(target_file)
    pca.load_state_dict(loaded_pca_state, strict=False)
    assert torch.equal(pca.components_, comps_after), "The loaded components should match the saved components."
    forward_after_loading = pca.forward(X)  # Test that the model can still be used after loading state dict
    assert torch.equal(forward_after_loading, X_reduced), "The output after loading state dict should match the output before loading."

def test_pca_eval_failure_on_unfitted_model(X):
    # Create a random dataset
    # Initialize PCA with 2 components
    pca = PCAModule(n_components=2)
    pca.eval()  # Set the model to evaluation mode
    with pytest.raises(ValueError) as exc_info:
        pca.forward(X)
        assert False, "Expected an error when calling forward on an unfitted PCA model."
   
def test_pca_module_inverse_no_reduction(X):


    # Initialize PCA with 2 components
    pca = PCAModule()
    # Fit and transform the data
    X_reduced = pca.forward(X)
    # Inverse transform the reduced data
    X_approx = pca.pca.inverse_transform(X_reduced)
    # Check that the shape of the approximated data matches the original data
    assert X_approx.shape == X.shape, "The shape of the approximated data should match the original data."
    assert torch.allclose(X_approx, X), "The approximated data should match the original data when n_components=0 (no dimensionality reduction)."

def test_pca_module_inverse(X):

    # Initialize PCA with 2 components
    pca = PCAModule(n_components=1)
    # Fit and transform the data
    X_reduced = pca.forward(X)
    # Inverse transform the reduced data
    X_approx = pca.inverse(X_reduced)
    # Check that the shape of the approximated data matches the original data
    assert X_approx.shape == X.shape, "The shape of the approximated data should match the original data."
    assert not torch.allclose(X_approx, X), "The approximated data should not match the original data when n_components=2 (dimensionality reduction)."
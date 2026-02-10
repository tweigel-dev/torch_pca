from pathlib import Path
import torch
from torch_pca import PCA

import pytest

@pytest.fixture(scope="module")
def X() -> torch.Tensor:
    return torch.load("tests/input_data.pt").to(torch.float32)


def test_pca_as_torch_module(tmp_path: Path, X: torch.Tensor) -> None:
    pca = PCA(n_components=5)
    com_before = pca.components_
    assert not com_before.any(), "Components should be None before fitting."
    assert com_before.shape == torch.Size([0]), "Components shape should be empty before fitting."
    X_reduced = pca.forward(X)
    comps_after = pca.components_.clone()
    assert  comps_after.any(), "The components should be updated after fitting the PCA model."
    target_file = tmp_path/"pca.pt"
    torch.save(pca.state_dict(), target_file)
    loaded_pca_state = torch.load(target_file)
    pca = PCA(n_components=5)
    pca.load_state_dict(loaded_pca_state,strict=False)
    assert torch.equal(pca.components_, comps_after), "The loaded components should match the saved components."
    forward_after_loading = pca.forward(X)
    assert torch.equal(forward_after_loading, X_reduced), "The output after loading state dict should match the output before loading."

def test_pca_eval_failure_on_unfitted_model(X):
    pca = PCA(n_components=2)
    pca.eval()
    with pytest.raises(ValueError):
        pca.forward(X)
   
def test_pca_module_inverse_no_reduction(X):
    pca = PCA()
    X_reduced = pca.forward(X)
    X_approx = pca.inverse_transform(X_reduced)
    assert X_approx.shape == X.shape, "The shape of the approximated data should match the original data."
    assert torch.allclose(X_approx, X), "The approximated data should match the original data when all components are retained."

def test_pca_module_inverse(X):
    pca = PCA(n_components=1)
    X_reduced = pca.forward(X)
    X_approx = pca.inverse_transform(X_reduced)
    assert X_approx.shape == X.shape, "The shape of the approximated data should match the original data."
    assert not torch.allclose(X_approx, X), "The approximated data should not match the original data when n_components=1 (dimensionality reduction)."


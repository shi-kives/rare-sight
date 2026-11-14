import numpy as np
import pytest
import torch
import torch.nn as nn
from src.uncertainty.mc_dropout import mc_predict, calibrate_mc_threshold, apply_uncertainty_threshold
from src.uncertainty.confidence import compute_margin, distance_flag, compute_retrieval_agreement, overall_uncertainty

class DummyEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(12, 32)
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(32, 8)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.linear2(x)


@pytest.fixture
def model():
    torch.manual_seed(42)
    return DummyEmbeddingModel()


@pytest.fixture
def image():
    torch.manual_seed(42)
    return torch.randn(3, 2, 2)


# MC DROPOUT TESTS

def test_mc_predict_output_shapes(model, image):
    result = mc_predict(model, image, n_runs=20)

    assert result["mean_embedding"].shape == (8,)
    assert result["std_per_dim"].shape == (8,)


def test_mc_predict_outputs_are_finite(model, image):
    result = mc_predict(model, image, n_runs=20)

    assert np.isfinite(result["mean_embedding"]).all()
    assert np.isfinite(result["std_per_dim"]).all()
    assert np.isfinite(result["variance"])


def test_mc_dropout_produces_nonzero_variance(model, image):
    result = mc_predict(model, image, n_runs=30)

    assert result["variance"] > 0


def test_mc_predict_variance_is_nonnegative(model, image):
    result = mc_predict(model, image, n_runs=20)

    assert result["variance"] >= 0
    assert np.all(result["std_per_dim"] >= 0)


def test_mc_predict_returns_no_decision_before_calibration(model, image):
    result = mc_predict(model, image, n_runs=20)

    assert result["epistemic_uncertain"] is None


def test_mc_predict_rejects_invalid_n_runs(model, image):
    with pytest.raises(ValueError):
        mc_predict(model, image, n_runs=1)


def test_mc_predict_rejects_invalid_input_dimensions(model):
    with pytest.raises(ValueError):
        mc_predict(model, torch.randn(3, 2), n_runs=10)


def test_mc_predict_rejects_batch_size_greater_than_one(model):
    with pytest.raises(ValueError):
        mc_predict(model, torch.randn(2, 3, 2, 2), n_runs=10)


def test_mc_predict_rejects_integer_input(model):
    with pytest.raises(TypeError):
        mc_predict(model, torch.randint(0, 255, (3, 2, 2)), n_runs=10)


# MODEL STATE AND DROPOUT TESTS

def test_batchnorm_remains_in_eval_mode(model, image):
    model.train()

    original_mean = model.bn.running_mean.clone()
    original_var = model.bn.running_var.clone()

    mc_predict(model, image, n_runs=20)

    assert torch.equal(model.bn.running_mean, original_mean)
    assert torch.equal(model.bn.running_var, original_var)


def test_model_training_state_is_restored(model, image):
    model.train()
    original_modes = {module: module.training for module in model.modules()}

    mc_predict(model, image, n_runs=10)

    for module, training_mode in original_modes.items():
        assert module.training == training_mode


def test_model_eval_state_is_restored(model, image):
    model.eval()

    mc_predict(model, image, n_runs=10)

    assert model.training is False
    assert model.bn.training is False


# THRESHOLD CALIBRATION TESTS

def test_calibration_returns_finite_threshold(model):
    torch.manual_seed(42)
    val_images = [torch.randn(3, 2, 2) for _ in range(20)]

    threshold = calibrate_mc_threshold(model, val_images, n_runs=10, percentile=95)

    assert isinstance(threshold, float)
    assert np.isfinite(threshold)
    assert threshold >= 0


def test_calibration_accepts_tuple_dataset(model):
    torch.manual_seed(42)
    val_images = [(torch.randn(3, 2, 2), i % 3) for i in range(10)]

    threshold = calibrate_mc_threshold(model, val_images, n_runs=10)

    assert isinstance(threshold, float)


def test_calibration_rejects_empty_validation_data(model):
    with pytest.raises(ValueError):
        calibrate_mc_threshold(model, [], n_runs=10)


def test_calibration_rejects_invalid_n_runs(model):
    with pytest.raises(ValueError):
        calibrate_mc_threshold(model, [torch.randn(3, 2, 2)], n_runs=1)


def test_calibration_rejects_invalid_percentile(model):
    with pytest.raises(ValueError):
        calibrate_mc_threshold(model, [torch.randn(3, 2, 2)], percentile=100)


def test_calibration_percentile_ordering(model):
    torch.manual_seed(42)
    val_images = [torch.randn(3, 2, 2) for _ in range(30)]

    threshold_90 = calibrate_mc_threshold(model, val_images, n_runs=10, percentile=90)
    threshold_95 = calibrate_mc_threshold(model, val_images, n_runs=10, percentile=95)

    assert threshold_95 >= threshold_90


# UNCERTAINTY DECISION TESTS

def test_prediction_is_accepted_below_threshold():
    assert apply_uncertainty_threshold(0.02, 0.05) is False


def test_prediction_is_rejected_above_threshold():
    assert apply_uncertainty_threshold(0.08, 0.05) is True


def test_prediction_at_threshold_is_not_rejected():
    assert apply_uncertainty_threshold(0.05, 0.05) is False


def test_uncertainty_decision_rejects_negative_variance():
    with pytest.raises(ValueError):
        apply_uncertainty_threshold(-0.01, 0.05)


def test_uncertainty_decision_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        apply_uncertainty_threshold(float("nan"), 0.05)


# END-TO-END UNCERTAINTY UNIT TEST

def test_complete_uncertainty_pipeline(model, image):
    val_images = [torch.randn(3, 2, 2) for _ in range(20)]

    threshold = calibrate_mc_threshold(model, val_images, n_runs=20, percentile=95)

    result = mc_predict(model, image, n_runs=20)

    uncertain = apply_uncertainty_threshold(result["variance"], threshold)

    assert isinstance(uncertain, bool)
    assert result["variance"] >= 0
    assert threshold >= 0
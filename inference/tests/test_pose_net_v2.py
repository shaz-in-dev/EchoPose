"""tests/test_pose_net_v2.py — PoseNetV2 architecture coverage."""

import pytest
import torch
import numpy as np
from pipeline.pose_net_v2 import PoseNetV2, FEATURE_SHAPE, MAX_PEOPLE, NUM_KEYPOINTS


@pytest.fixture
def model():
    m = PoseNetV2()
    m.eval()
    return m


def _random_input(batch=1):
    return torch.randn(batch, *FEATURE_SHAPE)


def test_forward_output_shape(model):
    x = _random_input()
    out = model(x)
    assert out.shape == (1, MAX_PEOPLE, NUM_KEYPOINTS, 4)


def test_batch_forward(model):
    x = _random_input(batch=4)
    out = model(x)
    assert out.shape == (4, MAX_PEOPLE, NUM_KEYPOINTS, 4)


def test_output_range_sigmoid(model):
    x = _random_input()
    out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_encoder_output_shape(model):
    x = _random_input()
    emb = model.encoder(x)
    assert emb.shape == (1, 256)


def test_encoder_batch(model):
    x = _random_input(batch=3)
    emb = model.encoder(x)
    assert emb.shape == (3, 256)


def test_deterministic_eval(model):
    x = _random_input()
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_multi_scale_layers_exist(model):
    assert hasattr(model, "ext_3")
    assert hasattr(model, "ext_5")
    assert hasattr(model, "ext_7")


def test_temporal_lstm_exists(model):
    assert hasattr(model, "temporal")
    assert isinstance(model.temporal, torch.nn.LSTM)


def test_spatial_attention_exists(model):
    assert hasattr(model, "spatial_attention")
    assert isinstance(model.spatial_attention, torch.nn.MultiheadAttention)


def test_gradients_flow():
    model = PoseNetV2()
    model.train()
    x = _random_input()
    out = model(x)
    loss = out.sum()
    loss.backward()
    # Check at least one parameter has gradients
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0

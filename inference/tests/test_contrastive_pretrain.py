import torch

from research.contrastive_pretrain import (
    ContrastiveConfig,
    CSIContrastiveModel,
    csi_augment,
    nt_xent_loss,
    contrastive_step,
)


def test_csi_augment_preserves_shape():
    x = torch.randn(8, 3, 64, 16)
    y = csi_augment(x)
    assert y.shape == x.shape


def test_model_output_is_normalized():
    model = CSIContrastiveModel(ContrastiveConfig())
    z = model(torch.randn(4, 3, 64, 16))
    norms = torch.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_nt_xent_loss_is_finite():
    z1 = torch.randn(6, 64)
    z2 = torch.randn(6, 64)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    loss = nt_xent_loss(z1, z2)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_contrastive_step_runs():
    model = CSIContrastiveModel(ContrastiveConfig())
    batch = torch.randn(5, 3, 64, 16)
    loss, z1, z2 = contrastive_step(model, batch)
    assert loss.item() > 0
    assert z1.shape == z2.shape

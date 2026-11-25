"""Quick model-construction test (uses random weights, no download): pytest -q"""
import numpy as np

from src.model import IMG_SIZE, build_model, unfreeze_top


def test_build_and_unfreeze():
    model, base = build_model(num_classes=5, weights=None)
    out = model.predict(np.random.rand(2, *IMG_SIZE, 3).astype("float32") * 255, verbose=0)
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)
    assert not base.trainable
    unfreeze_top(base, 30)
    trainable = [l for l in base.layers if l.trainable and l.weights]
    assert 0 < len(trainable) <= 30

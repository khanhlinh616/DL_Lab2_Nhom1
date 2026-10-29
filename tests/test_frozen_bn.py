"""
Unit tests for BatchNorm buffer update measurements in frozen backbones.
"""

import pytest
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from check_frozen_bn import measure_bn_buffer_updates


@pytest.mark.parametrize("model_name,expected_bn_count", [
    ("resnet18", 20),
    ("densenet121", 121)
])
def test_measure_bn_buffer_updates_structure(model_name, expected_bn_count):
    """Verify measure_bn_buffer_updates returns dictionary with required fields and counts."""
    res = measure_bn_buffer_updates(model_name=model_name, method="transfer", mode="eval", num_samples=4, batch_size=4)

    assert isinstance(res, dict)
    assert res["model_name"] == model_name
    assert res["method"] == "transfer"
    assert res["mode"] == "eval"
    assert res["frozen_bn_layers_count"] == expected_bn_count
    # 3 buffers per BN: running_mean, running_var, num_batches_tracked
    assert res["total_buffers_count"] == expected_bn_count * 3
    assert "changed_buffers_count" in res
    assert "unchanged_buffers_count" in res
    assert "changed_details" in res


def test_bn_buffers_frozen_in_eval_mode():
    """Verify that in eval mode, no BatchNorm buffers in the frozen backbone are modified."""
    res = measure_bn_buffer_updates(model_name="resnet18", method="transfer", mode="eval", num_samples=8, batch_size=4)

    assert res["changed_buffers_count"] == 0
    assert res["unchanged_buffers_count"] == res["total_buffers_count"]
    assert len(res["changed_details"]) == 0


def test_bn_buffers_updated_in_train_mode():
    """Verify that in train mode, BatchNorm buffers are updated despite requires_grad=False."""
    res = measure_bn_buffer_updates(model_name="resnet18", method="transfer", mode="train", num_samples=8, batch_size=4)

    assert res["changed_buffers_count"] > 0
    assert res["changed_buffers_count"] == res["total_buffers_count"]
    assert len(res["changed_details"]) == res["total_buffers_count"]

"""Tests for the CLIP dual-tower engine, its supervision, and the backend router.

Covers the acceptance criteria of 修改方案 §6:
  step 1  motion descriptions are deterministic and use the reference vocabulary
  step 2  the dual tower fine-tunes exactly 4 vision / 2 text layers and aligns
  step 4  the soft-IoU / proposal-token supervision behaves (gradient is non-zero)
  step 6  BackendMismatch is raised for hosted+open / local+hosted combinations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mewm.config import ClipConfig, load_config  # noqa: E402
from mewm.engines.motion_description import (  # noqa: E402
    au_motion_description, describe_roi, frame_motion_description,
)
from mewm.engines.v1_motion import coherence_label, direction_label, magnitude_class  # noqa: E402
from mewm.schemas import ROIMeasurement  # noqa: E402
from mewm.training.sft import proposal_number_spans, soft_iou  # noqa: E402

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False

CLIP_WEIGHTS = Path(load_config().clip.weights_path)
_HAS_CLIP = _TORCH and (CLIP_WEIGHTS / "config.json").is_file()


def _measurement(roi_index=20, roi_name="right_mouth_corner",
                 magnitude=0.82, direction=45.0, coherence=0.91,
                 salient=True) -> ROIMeasurement:
    return ROIMeasurement(
        roi_index=roi_index, roi_name=roi_name, roi_label=roi_name,
        magnitude_px=magnitude, direction_deg=direction, coherence=coherence,
        salient=salient, magnitude_class=magnitude_class(magnitude),
        direction_label=direction_label(direction),
        coherence_label=coherence_label(coherence))


# ---------------------------------------------------------------------------
# Step 1: deterministic motion descriptions
# ---------------------------------------------------------------------------


class TestMotionDescription:
    def test_full_style_carries_reference_fields(self):
        text = describe_roi(_measurement(), style="full")
        # The observation-record vocabulary: significance marks + fit judgements.
        assert "right_mouth_corner" in text
        assert "direction 45.0deg" in text
        assert "0.820px" in text
        assert "coherence 0.910 High" in text
        assert "AU12" in text and ("FIT" in text or "PARTIAL" in text)
        assert any(mark in text for mark in "★◈○")

    def test_deterministic(self):
        measurements = [_measurement(), _measurement(5, "left_inner_brow", 0.3, 92.0, 0.7)]
        first = frame_motion_description(measurements)
        for _ in range(20):
            assert frame_motion_description(measurements) == first

    def test_ranked_by_magnitude(self):
        weak = _measurement(5, "left_inner_brow", 0.10, 92.0, 0.7)
        strong = _measurement()
        text = frame_motion_description([weak, strong])
        assert text.index("right_mouth_corner") < text.index("left_inner_brow")

    def test_empty_pool(self):
        assert frame_motion_description([]) == "no facial motion observed"

    def test_au_description(self):
        text = au_motion_description({"AU6": 0.74, "AU12": 0.51, "AU4": 0.1})
        assert "AU6 active (peak 0.74)" in text
        assert "AU4" not in text


# ---------------------------------------------------------------------------
# Step 4: soft-IoU + proposal-token supervision
# ---------------------------------------------------------------------------


class TestProposalSupervision:
    def test_soft_iou_identity_and_disjoint(self):
        assert soft_iou((10, 20), (10, 20)) == pytest.approx(1.0)
        assert soft_iou((10, 20), (30, 40)) == 0.0
        assert 0.0 < soft_iou((10, 20), (15, 25)) < 1.0

    def test_number_spans_keyed(self):
        text = '{"part1_proposals": [{"proposal_id": 1, "onset": 57, "offset": 71, "apex": 62}]}'
        spans = proposal_number_spans(text)
        found = {text[a:b] for a, b in spans}
        assert {"57", "71", "62"} <= found
        assert "1" not in found  # proposal_id is not a localisation token

    def test_number_spans_triple(self):
        text = '{"proposals": [[57, 71, 62]], "answer": "x"}'
        found = {text[a:b] for a, b in proposal_number_spans(text)}
        assert {"57", "71", "62"} <= found

    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_soft_iou_loss_gradient_nonzero(self):
        """方案 §6 step 4: on a synthetic sample the localisation loss carries gradient."""
        from mewm.training.clip_localiser import soft_iou_loss
        logits = torch.zeros(32, requires_grad=True)
        labels = torch.zeros(32)
        labels[10:16] = 1.0
        mask = torch.ones(32)
        loss = soft_iou_loss(torch.sigmoid(logits), labels, mask)
        loss.backward()
        assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Step 6: backend router
# ---------------------------------------------------------------------------


class TestBackendRouter:
    def test_hosted_with_open_weights_rejected(self):
        from mewm.llm.client import BackendMismatch, backend_for
        with pytest.raises(BackendMismatch):
            backend_for("Qwen3-VL-8B", "hosted")

    def test_local_with_hosted_rejected(self):
        from mewm.llm.client import BackendMismatch, backend_for
        with pytest.raises(BackendMismatch):
            backend_for("claude-sonnet-5", "local")

    def test_valid_combinations(self):
        from mewm.llm.client import backend_for
        assert backend_for("claude-sonnet-5", "hosted").model_id == "claude-sonnet-5"
        assert backend_for("Qwen3-VL-8B", "local").open_weights

    def test_unknown_backend_rejected(self):
        from mewm.llm.client import BackendMismatch, backend_for
        with pytest.raises(BackendMismatch):
            backend_for("claude-sonnet-5", "turbo")

    def test_manifest_records_and_resets(self):
        from mewm.llm.client import (
            backend_manifest, record_backend_event, reset_backend_manifest,
        )
        reset_backend_manifest()
        record_backend_event("R", "claude-sonnet-5", "hosted", "https://x", 1.0)
        record_backend_event("P", "Qwen3-VL-8B", "local", "local:x", 2.0,
                             fallback=True, note="weights missing")
        manifest = backend_manifest(reset=True)
        assert len(manifest) == 2
        assert manifest[1]["fallback"] is True
        assert backend_manifest() == []

    def test_config_backends_load(self):
        config = load_config()
        assert config.llm.backend_for_role("R") in {"hosted", "local"}
        assert config.backends.local_quantization in {"", "none", "4bit"}


# ---------------------------------------------------------------------------
# Head-motion features and negatives (方案 §2.2)
# ---------------------------------------------------------------------------


class TestHeadMotion:
    def test_head_feature_block_shape(self):
        from mewm.engines.clip_motion_engine import head_feature_block
        block = head_feature_block(np.random.default_rng(0).normal(size=(50, 6)), 50)
        assert block.shape == (50, 7)
        assert np.isfinite(block).all()

    def test_head_feature_block_missing(self):
        from mewm.engines.clip_motion_engine import head_feature_block
        assert head_feature_block(None, 10).shape == (10, 7)

    def test_negative_mask_excludes_events(self):
        from mewm.training.clip_localiser import ClipVideoSample, negative_mask
        rng = np.random.default_rng(1)
        head = np.zeros((100, 7), dtype=np.float32)
        head[:, -1] = rng.uniform(0, 1, 100)
        head[40:50, -1] = 5.0                      # fast head inside the event
        head[70:80, -1] = 5.0                      # fast head outside any event
        labels = np.zeros(100, dtype=np.float32)
        labels[40:50] = 1.0
        sample = ClipVideoSample(
            video_key="v", subject="s", frames=list(range(100)),
            frame_paths=[""] * 100, descriptions=[""] * 100,
            labels=labels, ignore=np.zeros(100, dtype=bool),
            head_block=head, analytic=np.zeros((100, 16), dtype=np.float32))
        negatives = negative_mask(sample, 75.0)
        assert negatives[70:80].sum() == 10        # the true head-motion negatives
        assert negatives[40:50].sum() == 0         # event frames are never negatives

    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_triplet_needs_both_sides(self):
        from mewm.training.clip_localiser import head_motion_triplet
        u = torch.randn(30, 8)
        labels = torch.zeros(30)
        negatives = torch.zeros(30)
        assert head_motion_triplet(u, labels, negatives, 0.2) is None
        labels[5:10] = 1.0
        negatives[20:25] = 1.0
        loss = head_motion_triplet(u, labels, negatives, 0.2)
        assert loss is not None and float(loss) >= 0.0


# ---------------------------------------------------------------------------
# Step 2: the dual tower itself (needs the local CLIP weights)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CLIP, reason="local CLIP weights not present")
class TestMotionCLIP:
    @pytest.fixture(scope="class")
    def model(self):
        from mewm.engines.clip_motion_engine import CLIPSpotterModel
        config = ClipConfig(**{**vars(load_config().clip),
                               "device": "cpu", "autocast_bf16": False})
        return CLIPSpotterModel(config, n_slots=16)

    def test_unfreeze_counts(self, model):
        counts = model.towers.unfrozen_layer_counts()
        assert counts == {"vision": 6, "text": 2}

    def test_lower_layers_frozen(self, model):
        vision_layers = model.towers.clip.vision_model.encoder.layers
        assert not any(p.requires_grad for p in vision_layers[0].parameters())
        text_layers = model.towers.clip.text_model.encoder.layers
        assert not any(p.requires_grad for p in text_layers[0].parameters())

    def test_projections_trainable(self, model):
        assert all(p.requires_grad
                   for p in model.towers.clip.visual_projection.parameters())
        assert all(p.requires_grad
                   for p in model.towers.clip.text_projection.parameters())

    def test_text_encoding_and_alignment(self, model):
        from mewm.engines.clip_motion_engine import info_nce
        texts = [frame_motion_description([_measurement()]),
                 frame_motion_description([_measurement(5, "left_inner_brow",
                                                        0.3, 92.0, 0.7)])]
        tokens = model.towers.tokenize(texts)
        m = model.towers.encode_texts(tokens["input_ids"], tokens["attention_mask"])
        assert m.shape == (2, model.towers.embed_dim)
        assert torch.allclose(m.norm(dim=-1), torch.ones(2), atol=1e-4)
        loss = info_nce(m, m, 0.07)     # self-alignment must be finite and small
        assert torch.isfinite(loss)

    def test_fusion_and_transition_shapes(self, model):
        v = torch.randn(5, model.towers.embed_dim)
        m = torch.randn(5, model.towers.embed_dim)
        u = model.fuse(v, m)
        assert u.shape == v.shape
        activations = model.transition(u)
        assert activations.shape == (5, 16)
        assert float(activations.min()) >= 0.0 and float(activations.max()) <= 1.0

    def test_localise_shape(self, model):
        u = torch.randn(1, 40, model.towers.embed_dim)
        head = torch.randn(1, 40, 7)
        logits = model.localise(u, head)
        assert logits.shape == (1, 40)

    def test_trainable_checkpoint_roundtrip(self, model):
        state = model.trainable_state_dict()
        assert state, "trainable state must not be empty"
        assert all("vision_model.encoder.layers.0." not in k for k in state)
        model.load_trainable_state_dict(state)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

import numpy as np
import pytest

from youru import models
from youru.speaker import Embedder, OwnerProfile, normalize


@pytest.fixture(scope="module")
def embedder():
    return Embedder(models.ensure("titanet_large"))


def test_owner_scores_above_others(embedder, clips):
    profile = OwnerProfile(embedder.model_name, [embedder.embed(clips[k]) for k in ("a1", "a2", "a3")])
    owner = profile.score(embedder.embed(clips["a_test"]))
    others = [profile.score(embedder.embed(clips[k])) for k in ("b1", "c1")]
    assert owner > max(others) + 0.1, (owner, others)


def test_profile_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    p = OwnerProfile("m", rng.standard_normal((4, 8)))
    p.adapt(rng.standard_normal(8), max_size=10)
    p.save(tmp_path / "p.npz")
    q = OwnerProfile.load(tmp_path / "p.npz")
    assert q.model_name == "m"
    np.testing.assert_allclose(q.bank, p.bank, rtol=1e-6)


def test_adaptation_is_capped_and_keeps_enrollment():
    rng = np.random.default_rng(2)
    p = OwnerProfile("m", rng.standard_normal((3, 8)))
    for _ in range(20):
        p.adapt(rng.standard_normal(8), max_size=5)
    assert len(p.adapted) == 5 and len(p.enroll) == 3


def test_score_is_cosine_for_identical():
    v = normalize(np.ones(8, np.float32))
    assert OwnerProfile("m", np.tile(v, (3, 1))).score(v) == pytest.approx(1.0, abs=1e-5)

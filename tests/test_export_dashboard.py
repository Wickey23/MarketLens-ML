import numpy as np

from src.export_dashboard import block_bootstrap_lift_ci


def test_block_bootstrap_lift_ci_detects_clear_positive_lift():
    rng=np.random.default_rng(7)
    n=500
    similar=np.zeros(n,dtype=bool)
    similar[::3]=True
    # Deterministic synthetic sequence with materially higher hit rate in the
    # marked similar-signal subset.
    y=rng.binomial(1,0.45,size=n).astype(float)
    y[similar]=rng.binomial(1,0.72,size=int(similar.sum()))
    lo,hi=block_bootstrap_lift_ci(y,similar,block_size=5,samples=600,seed=42)
    assert lo is not None and hi is not None
    assert lo>0
    assert hi>lo


def test_block_bootstrap_lift_ci_requires_enough_similar_observations():
    y=np.array([0,1]*20,dtype=float)
    similar=np.zeros(len(y),dtype=bool)
    similar[:5]=True
    assert block_bootstrap_lift_ci(y,similar)==[None,None]

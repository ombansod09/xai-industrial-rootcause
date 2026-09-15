import numpy as np
from xai_industrial.xai.agreement import jaccard
from xai_industrial.xai.stability import cosine_similarity

def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1/3

def test_cosine():
    assert np.isclose(cosine_similarity(np.array([1.,0.]), np.array([1.,0.])), 1.0)

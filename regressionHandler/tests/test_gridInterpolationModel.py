import numpy as np
import pytest

from pythonLibs.regressionHandler import GridInterpolationModel, loadModel


def _grid(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def test_reproducesMultilinearFunctionsExactly():
    axes = [np.array([0.0, 1.0, 3.0]), np.array([-1.0, 2.0]), np.array([0.0, 0.5, 1.0, 4.0])]
    pts = _grid(axes)
    f = lambda p: 1.0 + 2.0 * p[:, 0] - p[:, 1] + 0.5 * p[:, 0] * p[:, 1] * p[:, 2]
    perm = np.random.default_rng(0).permutation(len(pts))
    m = GridInterpolationModel().fit(pts[perm], f(pts)[perm])
    q = np.random.default_rng(1).uniform([-1.0, -2.0, -1.0], [4.0, 3.0, 5.0], (200, 3))
    assert m(q) == pytest.approx(f(q), abs=1e-12)


def test_extrapolationModes():
    axes = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]
    pts = _grid(axes)
    z = pts[:, 0] + 2.0 * pts[:, 1]
    q = np.array([[2.0, 0.0], [0.5, 0.5]])
    assert GridInterpolationModel().fit(pts, z)(q) == pytest.approx([2.0, 1.5])
    assert GridInterpolationModel(extrapolation="clamp").fit(pts, z)(q) == pytest.approx([1.0, 1.5])
    filled = GridInterpolationModel(extrapolation="fill").fit(pts, z)(q)
    assert np.isnan(filled[0]) and filled[1] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        GridInterpolationModel(extrapolation="error").fit(pts, z)(q)


def test_rejectsIncompleteGrid():
    pts = _grid([np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])])
    with pytest.raises(ValueError):
        GridInterpolationModel().fit(pts[:-1], pts[:-1, 0])


def test_matchesScipyRegularGridInterpolator():
    interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(2)
    axes = [np.sort(rng.uniform(-5.0, 5.0, 6)), np.sort(rng.uniform(0.0, 3.0, 4))]
    vals = rng.normal(size=(6, 4))
    m = GridInterpolationModel(extrapolation="error").fit(_grid(axes), vals.ravel())
    q = np.column_stack([rng.uniform(axes[0][0], axes[0][-1], 300), rng.uniform(axes[1][0], axes[1][-1], 300)])
    assert m(q) == pytest.approx(interpolate.RegularGridInterpolator(tuple(axes), vals)(q), abs=1e-12)


def test_saveLoadRoundTrip(tmp_path):
    pts = _grid([np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])])
    m = GridInterpolationModel().fit(pts, np.sin(pts[:, 0]) + pts[:, 1])
    m.save(str(tmp_path / "grid.json"))
    q = np.array([[0.3, 0.2], [1.7, 0.9]])
    assert loadModel(str(tmp_path / "grid.json"))(q) == pytest.approx(m(q), abs=0.0)

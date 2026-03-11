"""
Unit tests for calibration metrics and differentiable losses.
"""

import numpy as np
import pytest

from ignacio.calibration import (
    _rasterize_perimeter,
    _sample_pseudo_landmarks,
    _kabsch_rotation_2d,
    jaccard_index,
    sorensen_coefficient,
    centroid_size_ratio,
    procrustes_shape_rmse,
    orientation_error,
)


# =============================================================================
# Helpers
# =============================================================================


def _circle(cx, cy, r, n=100):
    """Generate a circle polygon as (N, 2) array."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])
    return xy


def _ellipse(cx, cy, a, b, angle=0.0, n=100):
    """Generate a rotated ellipse polygon."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    c, s = np.cos(angle), np.sin(angle)
    xr = c * x - s * y + cx
    yr = s * x + c * y + cy
    return np.column_stack([xr, yr])


# =============================================================================
# Test rasterization helper
# =============================================================================


class TestRasterize:
    def test_circle_rasterized(self):
        xy = _circle(50.0, 50.0, 20.0, n=200)
        mask = _rasterize_perimeter(xy, (0, 0, 100, 100), resolution=1.0)
        area = mask.sum()
        expected = np.pi * 20**2
        assert abs(area - expected) / expected < 0.05

    def test_small_square(self):
        xy = np.array([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]], dtype=float)
        mask = _rasterize_perimeter(xy, (0, 0, 30, 30), resolution=1.0)
        area = mask.sum()
        assert abs(area - 100) < 15  # roughly 10x10


# =============================================================================
# Test pseudo-landmark sampling
# =============================================================================


class TestLandmarks:
    def test_even_spacing(self):
        xy = _circle(0, 0, 10.0, n=500)
        lm = _sample_pseudo_landmarks(xy, 50)
        assert lm.shape == (50, 2)
        # All should be roughly distance 10 from center
        dists = np.sqrt(np.sum(lm**2, axis=1))
        np.testing.assert_allclose(dists, 10.0, atol=0.5)

    def test_correct_count(self):
        xy = _circle(0, 0, 5.0, n=100)
        for n in [10, 50, 200]:
            lm = _sample_pseudo_landmarks(xy, n)
            assert lm.shape == (n, 2)


# =============================================================================
# Test Kabsch rotation
# =============================================================================


class TestKabsch:
    def test_identity_rotation(self):
        pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        rotated = _kabsch_rotation_2d(pts, pts)
        np.testing.assert_allclose(rotated, pts, atol=1e-10)

    def test_90_degree_rotation(self):
        P = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        # Rotate P by 90 degrees
        R90 = np.array([[0, -1], [1, 0]], dtype=float)
        Q = P @ R90.T
        # Kabsch should recover Q from P
        P_aligned = _kabsch_rotation_2d(P, Q)
        np.testing.assert_allclose(P_aligned, Q, atol=1e-10)


# =============================================================================
# Test Jaccard and Sorensen
# =============================================================================


class TestOverlapMetrics:
    def test_identical_circles(self):
        c = _circle(50, 50, 20, n=200)
        bounds = (0, 0, 100, 100)
        assert jaccard_index(c, c, bounds, resolution=1.0) > 0.95
        assert sorensen_coefficient(c, c, bounds, resolution=1.0) > 0.97

    def test_non_overlapping(self):
        c1 = _circle(10, 10, 5, n=100)
        c2 = _circle(90, 90, 5, n=100)
        bounds = (0, 0, 100, 100)
        assert jaccard_index(c1, c2, bounds, resolution=1.0) < 0.01
        assert sorensen_coefficient(c1, c2, bounds, resolution=1.0) < 0.01

    def test_partial_overlap(self):
        c1 = _circle(50, 50, 20, n=200)
        c2 = _circle(60, 50, 20, n=200)
        bounds = (0, 0, 120, 100)
        j = jaccard_index(c1, c2, bounds, resolution=1.0)
        assert 0.3 < j < 0.9  # partial overlap
        s = sorensen_coefficient(c1, c2, bounds, resolution=1.0)
        assert s > j  # Sorensen ≥ Jaccard always

    def test_sorensen_geq_jaccard(self):
        c1 = _circle(50, 50, 15, n=200)
        c2 = _circle(55, 50, 20, n=200)
        bounds = (0, 0, 100, 100)
        j = jaccard_index(c1, c2, bounds, resolution=1.0)
        s = sorensen_coefficient(c1, c2, bounds, resolution=1.0)
        assert s >= j - 0.01  # numerical tolerance


# =============================================================================
# Test centroid size ratio
# =============================================================================


class TestCentroidSize:
    def test_same_size(self):
        c = _circle(0, 0, 10, n=200)
        ratio = centroid_size_ratio(c, c, origin=np.array([0.0, 0.0]))
        assert abs(ratio) < 0.01

    def test_larger_sim(self):
        sim = _circle(0, 0, 20, n=200)
        obs = _circle(0, 0, 10, n=200)
        ratio = centroid_size_ratio(sim, obs, origin=np.array([0.0, 0.0]))
        assert abs(ratio - np.log(2.0)) < 0.05

    def test_smaller_sim(self):
        sim = _circle(0, 0, 5, n=200)
        obs = _circle(0, 0, 10, n=200)
        ratio = centroid_size_ratio(sim, obs, origin=np.array([0.0, 0.0]))
        assert ratio < 0  # sim smaller → negative log ratio


# =============================================================================
# Test Procrustes RMSE
# =============================================================================


class TestProcrustes:
    def test_identical_shapes(self):
        c = _circle(0, 0, 10, n=200)
        rmse = procrustes_shape_rmse(c, c, origin=np.array([0.0, 0.0]))
        assert rmse < 0.05

    def test_different_shapes(self):
        sim = _ellipse(0, 0, 20, 10, n=200)
        obs = _circle(0, 0, 15, n=200)
        rmse = procrustes_shape_rmse(sim, obs, origin=np.array([0.0, 0.0]))
        assert rmse > 0.1  # clearly different shapes


# =============================================================================
# Test orientation error
# =============================================================================


class TestOrientation:
    def test_same_orientation(self):
        e = _ellipse(0, 0, 20, 5, angle=0.0, n=200)
        err = orientation_error(e, e, origin=np.array([0.0, 0.0]))
        assert err < 1.0  # degrees

    def test_90_degree_difference(self):
        e1 = _ellipse(0, 0, 20, 5, angle=0.0, n=200)
        e2 = _ellipse(0, 0, 20, 5, angle=np.pi / 2, n=200)
        err = orientation_error(e1, e2, origin=np.array([0.0, 0.0]))
        assert abs(err - 90.0) < 5.0

    def test_180_wraps_to_zero(self):
        # 180 degree difference should be close to 180 (or 0 for symmetric shapes)
        e1 = _ellipse(0, 0, 20, 5, angle=0.0, n=200)
        e2 = _ellipse(0, 0, 20, 5, angle=np.pi, n=200)
        err = orientation_error(e1, e2, origin=np.array([0.0, 0.0]))
        assert err < 5.0 or abs(err - 180.0) < 5.0


# =============================================================================
# Test differentiable losses (JAX)
# =============================================================================


jax_installed = pytest.importorskip("jax", reason="JAX not installed")


class TestDifferentiableLosses:
    def test_phi_perimeter_loss_zero_at_boundary(self):
        """If obs points lie on φ=0, loss should be ~0."""
        import jax.numpy as jnp
        from ignacio.calibration import phi_perimeter_loss

        x = jnp.linspace(-5, 5, 50)
        y = jnp.linspace(-5, 5, 50)
        X, Y = jnp.meshgrid(x, y)
        phi = jnp.sqrt(X**2 + Y**2) - 3.0  # circle r=3

        # Points on the circle
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        obs = jnp.array(np.column_stack([3.0 * np.cos(theta), 3.0 * np.sin(theta)]))

        loss = phi_perimeter_loss(phi, obs, x, y)
        assert float(loss) < 0.1

    def test_phi_perimeter_loss_nonzero_off_boundary(self):
        """Points far from φ=0 should give high loss."""
        import jax.numpy as jnp
        from ignacio.calibration import phi_perimeter_loss

        x = jnp.linspace(-10, 10, 50)
        y = jnp.linspace(-10, 10, 50)
        X, Y = jnp.meshgrid(x, y)
        phi = jnp.sqrt(X**2 + Y**2) - 3.0

        # Points far from circle
        obs = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # inside, φ < 0
        loss = phi_perimeter_loss(phi, obs, x, y)
        assert float(loss) > 1.0

    def test_phi_area_loss_correct_area(self):
        """When simulated area matches observed, loss should be ~0."""
        import jax.numpy as jnp
        from ignacio.calibration import phi_area_loss

        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        phi = jnp.sqrt(X**2 + Y**2) - 3.0

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        # True area ≈ π * 3² ≈ 28.27
        obs_area = np.pi * 9.0

        loss = phi_area_loss(phi, obs_area, dx, dy, epsilon=0.5)
        # Should be close to 0
        assert float(loss) < 5.0  # some smoothing error is ok

    def test_phi_area_loss_wrong_area(self):
        """Mismatched area should give high loss."""
        import jax.numpy as jnp
        from ignacio.calibration import phi_area_loss

        x = jnp.linspace(-5, 5, 100)
        y = jnp.linspace(-5, 5, 100)
        X, Y = jnp.meshgrid(x, y)
        phi = jnp.sqrt(X**2 + Y**2) - 3.0

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])

        loss = phi_area_loss(phi, 100.0, dx, dy, epsilon=0.5)
        assert float(loss) > 100.0

    def test_composite_loss_differentiable(self):
        """composite_loss should be differentiable w.r.t. phi."""
        import jax
        import jax.numpy as jnp
        from ignacio.calibration import composite_loss

        x = jnp.linspace(-5, 5, 30)
        y = jnp.linspace(-5, 5, 30)
        X, Y = jnp.meshgrid(x, y)
        phi = jnp.sqrt(X**2 + Y**2) - 3.0

        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        obs = jnp.array(np.column_stack([3.0 * np.cos(theta), 3.0 * np.sin(theta)]))
        obs_area = np.pi * 9.0

        grad_fn = jax.grad(composite_loss)
        grad_phi = grad_fn(phi, obs, obs_area, x, y)
        assert grad_phi.shape == phi.shape
        assert jnp.any(grad_phi != 0)

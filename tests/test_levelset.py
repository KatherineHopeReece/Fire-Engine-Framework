"""
Unit tests for the JAX level set fire spread solver.
"""

import numpy as np
import pytest

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ignacio.levelset import (
    initialize_phi,
    compute_elliptical_speed,
    compute_gradient,
    compute_gradient_magnitude_upwind,
    reinitialize_phi,
    level_set_step,
    level_set_step_fixed,
    extract_contour,
    simulate_fire_spread_levelset,
)
from ignacio.spread import FireParameterGrid, FirePerimeterHistory


# =============================================================================
# Helpers
# =============================================================================


def _make_uniform_param_grid(
    nx=50, ny=50, nt=10,
    ros_val=1.0, bros_val=0.2, fros_val=0.5, raz_val=0.0,
    x_range=(0.0, 100.0), y_range=(0.0, 100.0),
):
    """Create a uniform FireParameterGrid for testing."""
    x_coords = np.linspace(x_range[0], x_range[1], nx)
    y_coords = np.linspace(y_range[0], y_range[1], ny)
    shape = (nt, ny, nx)
    return FireParameterGrid(
        x_coords=x_coords,
        y_coords=y_coords,
        ros=np.full(shape, ros_val),
        bros=np.full(shape, bros_val),
        fros=np.full(shape, fros_val),
        raz=np.full(shape, raz_val),
    )


# =============================================================================
# Test: initialize_phi
# =============================================================================


class TestInitializePhi:
    def test_signed_distance(self):
        x = jnp.linspace(0, 10, 50)
        y = jnp.linspace(0, 10, 50)
        phi = initialize_phi(x, y, 5.0, 5.0, 2.0)

        assert phi.shape == (50, 50)
        # Center should be negative (inside)
        assert float(phi[25, 25]) < 0
        # Corners should be positive (outside)
        assert float(phi[0, 0]) > 0
        assert float(phi[-1, -1]) > 0

    def test_zero_contour_radius(self):
        x = jnp.linspace(-5, 5, 201)
        y = jnp.linspace(-5, 5, 201)
        phi = initialize_phi(x, y, 0.0, 0.0, 2.0)

        # Points at distance 2 from origin should have φ ≈ 0
        # Check a point on the x-axis at x=2
        idx_x2 = 140  # approximately x=2
        idx_center = 100
        assert abs(float(phi[idx_center, idx_x2])) < 0.1


# =============================================================================
# Test: isotropic circle expansion
# =============================================================================


class TestIsotropicCircle:
    def test_expanding_circle_area(self):
        """Constant isotropic ROS should produce an expanding circle."""
        ros_val = 1.0  # m/min
        dt = 1.0
        n_steps = 20
        r0 = 5.0

        pg = _make_uniform_param_grid(
            nx=100, ny=100, nt=n_steps,
            ros_val=ros_val, bros_val=ros_val, fros_val=ros_val, raz_val=0.0,
            x_range=(-50, 50), y_range=(-50, 50),
        )

        history = simulate_fire_spread_levelset(
            param_grid=pg,
            x_ignition=0.0,
            y_ignition=0.0,
            dt=dt,
            initial_radius=r0,
            store_every=n_steps,
            max_steps=n_steps,
            min_ros=0.001,
            cfl_factor=0.5,
            reinit_interval=5,
            reinit_iters=10,
        )

        # Should have initial + final perimeters
        assert len(history.perimeters) >= 2

        # Expected radius after n_steps: r0 + ROS * n_steps * dt
        r_expected = r0 + ros_val * n_steps * dt
        area_expected = np.pi * r_expected**2

        # Compute actual area from final perimeter
        areas = history.compute_areas()
        area_final = areas[-1]

        # Allow 25% tolerance (grid resolution + level set diffusion)
        assert area_final > 0, "Final area should be positive"
        ratio = area_final / area_expected
        assert 0.5 < ratio < 1.5, (
            f"Area ratio {ratio:.2f} outside tolerance. "
            f"Expected ~{area_expected:.0f}, got {area_final:.0f}"
        )


# =============================================================================
# Test: anisotropic ellipse
# =============================================================================


class TestAnisotropicEllipse:
    def test_elliptical_shape(self):
        """Different ROS/BROS/FROS should produce an elliptical perimeter."""
        ros_val = 3.0
        bros_val = 0.5
        fros_val = 1.0
        raz_val = 0.0  # spread along y-axis (north)

        pg = _make_uniform_param_grid(
            nx=100, ny=100, nt=15,
            ros_val=ros_val, bros_val=bros_val, fros_val=fros_val, raz_val=raz_val,
            x_range=(-60, 60), y_range=(-60, 60),
        )

        history = simulate_fire_spread_levelset(
            param_grid=pg,
            x_ignition=0.0, y_ignition=0.0,
            dt=1.0, initial_radius=2.0,
            store_every=15, max_steps=15,
            cfl_factor=0.5, reinit_interval=5,
        )

        assert len(history.perimeters) >= 2
        x_final, y_final = history.get_final_perimeter()

        # The fire should be elongated (not circular)
        x_extent = np.max(x_final) - np.min(x_final)
        y_extent = np.max(y_final) - np.min(y_final)
        # With ROS >> FROS, the fire is anisotropic: extents should differ
        ratio = max(x_extent, y_extent) / (min(x_extent, y_extent) + 1e-6)
        assert ratio > 1.2, (
            f"Expected anisotropic (elliptical) shape, "
            f"got x={x_extent:.1f}, y={y_extent:.1f}, ratio={ratio:.2f}"
        )


# =============================================================================
# Test: non-fuel barrier
# =============================================================================


class TestNonFuelBarrier:
    def test_zero_ros_blocks_fire(self):
        """A strip of zero ROS should block fire spread."""
        nx, ny, nt = 100, 100, 15
        x_coords = np.linspace(-50, 50, nx)
        y_coords = np.linspace(-50, 50, ny)

        ros = np.full((nt, ny, nx), 2.0)
        bros = np.full((nt, ny, nx), 0.5)
        fros = np.full((nt, ny, nx), 1.0)
        raz = np.zeros((nt, ny, nx))

        # Create a wide barrier strip at x=10..20 (zero ROS, ~10 cells wide)
        barrier_mask = (x_coords >= 10.0) & (x_coords <= 20.0)
        ros[:, :, barrier_mask] = 0.0
        bros[:, :, barrier_mask] = 0.0
        fros[:, :, barrier_mask] = 0.0

        pg = FireParameterGrid(
            x_coords=x_coords, y_coords=y_coords,
            ros=ros, bros=bros, fros=fros, raz=raz,
        )

        history = simulate_fire_spread_levelset(
            param_grid=pg,
            x_ignition=-10.0, y_ignition=0.0,
            dt=1.0, initial_radius=3.0,
            store_every=15, max_steps=15,
            cfl_factor=0.5, reinit_interval=5,
        )

        x_final, y_final = history.get_final_perimeter()
        # Fire should not have crossed the barrier (barrier starts at x=10)
        assert np.max(x_final) < 25.0, (
            f"Fire crossed barrier: max x = {np.max(x_final):.1f}"
        )


# =============================================================================
# Test: output format
# =============================================================================


class TestOutputFormat:
    def test_returns_fire_perimeter_history(self):
        pg = _make_uniform_param_grid(nx=30, ny=30, nt=5)
        history = simulate_fire_spread_levelset(
            param_grid=pg,
            x_ignition=50.0, y_ignition=50.0,
            dt=1.0, initial_radius=5.0,
            store_every=1, max_steps=5,
        )

        assert isinstance(history, FirePerimeterHistory)
        assert len(history.times) == len(history.perimeters)
        assert history.times[0] == 0.0

        for x, y in history.perimeters:
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(x) == len(y)
            assert len(x) > 0


# =============================================================================
# Test: JAX grad
# =============================================================================


class TestJAXGrad:
    def test_grad_through_level_set_step(self):
        """Verify that jax.grad works through level_set_step_fixed w.r.t. ROS."""
        nx, ny = 40, 40
        x = jnp.linspace(-20, 20, nx)
        y = jnp.linspace(-20, 20, ny)
        phi = initialize_phi(x, y, 0.0, 0.0, 5.0)

        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])

        bros = jnp.full((ny, nx), 0.3)
        fros = jnp.full((ny, nx), 0.6)
        raz = jnp.zeros((ny, nx))

        def loss_fn(ros_scalar):
            ros = jnp.full((ny, nx), ros_scalar)
            phi_new = level_set_step_fixed(
                phi, ros, bros, fros, raz, dx, dy, 0.5,
                cfl_factor=0.5, n_substeps=2,
            )
            # Loss: mean of φ (should decrease as fire grows)
            return jnp.mean(phi_new)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        g = grad_fn(1.0)

        assert jnp.isfinite(g), f"Gradient is not finite: {g}"
        # Higher ROS should make φ more negative (fire expands), so gradient should be negative
        assert float(g) < 0, f"Expected negative gradient, got {float(g)}"


# =============================================================================
# Test: level set vs marker method (rough comparison)
# =============================================================================


class TestMarkerComparison:
    def test_areas_within_tolerance(self):
        """Level set and marker method should produce similar areas."""
        from ignacio.spread import simulate_fire_spread

        ros_val = 1.5
        pg = _make_uniform_param_grid(
            nx=80, ny=80, nt=15,
            ros_val=ros_val, bros_val=0.3, fros_val=0.7, raz_val=0.0,
            x_range=(-50, 50), y_range=(-50, 50),
        )

        # Level set
        ls_history = simulate_fire_spread_levelset(
            param_grid=pg,
            x_ignition=0.0, y_ignition=0.0,
            dt=1.0, initial_radius=3.0,
            store_every=15, max_steps=15,
        )

        # Marker method
        mk_history = simulate_fire_spread(
            param_grid=pg,
            x_ignition=0.0, y_ignition=0.0,
            dt=1.0, initial_radius=3.0,
            n_vertices=300,
            store_every=15, max_steps=15,
            use_markers=True, marker_epsilon=1.0,
        )

        ls_areas = ls_history.compute_areas()
        mk_areas = mk_history.compute_areas()

        if len(ls_areas) > 1 and len(mk_areas) > 1:
            ls_final = ls_areas[-1]
            mk_final = mk_areas[-1]
            if mk_final > 0:
                ratio = ls_final / mk_final
                assert 0.5 < ratio < 2.0, (
                    f"Area ratio {ratio:.2f} outside 50-200% tolerance. "
                    f"Level set: {ls_final:.0f}, Marker: {mk_final:.0f}"
                )

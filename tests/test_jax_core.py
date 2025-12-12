"""
Tests for Ignacio JAX Core Module.

Tests cover:
- FBP calculations (ISI, ROS by fuel type)
- Richards' velocity equations
- Perimeter evolution
- Calibration workflow
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

from ignacio.jax_core import (
    # FBP
    FBPCalibrationParams,
    calculate_isi_jax,
    compute_ros_components_jax,
    default_params,
    ros_c2_jax,
    ros_o1_jax,
    # Spread
    compute_fire_area_jax,
    compute_spatial_derivatives_jax,
    create_initial_perimeter_jax,
    evolve_perimeter_step_jax,
    richards_velocity_jax,
    ros_to_ellipse_params_jax,
    # Calibration
    CalibrationObservation,
    create_synthetic_observation,
    run_fire_forward_model,
)


# =============================================================================
# FBP Tests
# =============================================================================


class TestISICalculation:
    """Test ISI calculation."""
    
    def test_isi_scalar(self):
        """Test ISI with scalar inputs."""
        params = default_params()
        ffmc = jnp.array(90.0)
        wind_speed = jnp.array(20.0)
        
        isi = calculate_isi_jax(ffmc, wind_speed, params)
        
        assert isi > 0, "ISI should be positive"
        assert jnp.isfinite(isi), "ISI should be finite"
    
    def test_isi_increases_with_wind(self):
        """ISI should increase with wind speed."""
        params = default_params()
        ffmc = jnp.array(90.0)
        
        isi_low = calculate_isi_jax(ffmc, jnp.array(10.0), params)
        isi_high = calculate_isi_jax(ffmc, jnp.array(30.0), params)
        
        assert isi_high > isi_low, "ISI should increase with wind speed"
    
    def test_isi_increases_with_ffmc(self):
        """ISI should increase with FFMC (drier = more spread)."""
        params = default_params()
        wind_speed = jnp.array(20.0)
        
        isi_wet = calculate_isi_jax(jnp.array(70.0), wind_speed, params)
        isi_dry = calculate_isi_jax(jnp.array(95.0), wind_speed, params)
        
        assert isi_dry > isi_wet, "ISI should increase with FFMC"
    
    def test_isi_wind_adjustment(self):
        """Wind adjustment should scale effective wind speed."""
        ffmc = jnp.array(90.0)
        wind_speed = jnp.array(20.0)
        
        params_normal = FBPCalibrationParams(wind_adj=1.0)
        params_high = FBPCalibrationParams(wind_adj=1.5)
        
        isi_normal = calculate_isi_jax(ffmc, wind_speed, params_normal)
        isi_high = calculate_isi_jax(ffmc, wind_speed, params_high)
        
        assert isi_high > isi_normal, "Higher wind_adj should increase ISI"
    
    def test_isi_ffmc_adjustment(self):
        """FFMC adjustment should shift effective moisture."""
        wind_speed = jnp.array(20.0)
        ffmc = jnp.array(88.0)
        
        params_normal = FBPCalibrationParams(ffmc_adj=0.0)
        params_drier = FBPCalibrationParams(ffmc_adj=5.0)  # Drier conditions
        
        isi_normal = calculate_isi_jax(ffmc, wind_speed, params_normal)
        isi_drier = calculate_isi_jax(ffmc, wind_speed, params_drier)
        
        assert isi_drier > isi_normal, "Positive ffmc_adj should increase ISI"


class TestROSCalculation:
    """Test ROS by fuel type."""
    
    def test_ros_c2_positive(self):
        """C-2 ROS should be positive for normal conditions."""
        isi = jnp.array(10.0)
        bui = jnp.array(80.0)
        
        ros = ros_c2_jax(isi, bui)
        
        assert ros > 0, "ROS should be positive"
        assert ros < 200, "ROS should be reasonable (< 200 m/min)"
    
    def test_ros_o1_curing_effect(self):
        """Grass ROS should increase with curing."""
        isi = jnp.array(15.0)
        
        ros_green = ros_o1_jax(isi, jnp.array(50.0))
        ros_cured = ros_o1_jax(isi, jnp.array(95.0))
        
        assert ros_cured > ros_green, "Cured grass should spread faster"
    
    def test_ros_components(self):
        """Test ROS component calculation."""
        params = default_params()
        ros_head = jnp.array(10.0)
        wind_speed = jnp.array(25.0)
        wind_dir = jnp.array(270.0)
        
        ros, bros, fros, raz = compute_ros_components_jax(
            ros_head, wind_speed, wind_dir, params
        )
        
        assert ros == ros_head, "Head ROS should match input"
        assert bros < ros, "Back fire should be slower than head"
        assert fros < ros, "Flank fire should be slower than head"
        assert 0 <= raz <= 2 * jnp.pi, "RAZ should be in radians"


# =============================================================================
# Spread Tests
# =============================================================================


class TestSpatialDerivatives:
    """Test spatial derivative computation."""
    
    def test_circle_tangents(self):
        """Tangents on a circle should be perpendicular to radii."""
        n = 100
        theta = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        x = jnp.cos(theta)
        y = jnp.sin(theta)
        
        x_s, y_s = compute_spatial_derivatives_jax(x, y, normalize=True)
        
        # Check normalization
        mags = jnp.sqrt(x_s**2 + y_s**2)
        assert jnp.allclose(mags, 1.0, atol=1e-6), "Tangents should be normalized"
        
        # Check perpendicularity to radii (dot product should be ~0)
        dots = x * x_s + y * y_s
        assert jnp.allclose(dots, 0.0, atol=1e-2), "Tangents should be perpendicular to radii"


class TestRichardsVelocity:
    """Test Richards' velocity equations."""
    
    def test_circular_spread(self):
        """With a=b (circle), velocity should be radial outward."""
        n = 50
        x, y = create_initial_perimeter_jax(0.0, 0.0, 1.0, n)
        
        # Circular ellipse parameters (a=b, c=0)
        a = jnp.full(n, 1.0)
        b = jnp.full(n, 1.0)
        c = jnp.full(n, 0.0)
        theta = jnp.zeros(n)
        
        x_t, y_t = richards_velocity_jax(x, y, a, b, c, theta)
        
        # Velocity should be roughly radial (parallel to position vector)
        # Normalize both
        r_mag = jnp.sqrt(x**2 + y**2)
        v_mag = jnp.sqrt(x_t**2 + y_t**2)
        
        # Direction cosine (should be close to 1 for radial)
        cos_angle = (x * x_t + y * y_t) / (r_mag * v_mag + 1e-10)
        
        assert jnp.mean(cos_angle) > 0.9, "Velocity should be roughly outward"
    
    def test_velocity_magnitude(self):
        """Velocity magnitude should scale with ellipse size."""
        n = 50
        x, y = create_initial_perimeter_jax(0.0, 0.0, 1.0, n)
        
        a1, b1, c1 = jnp.full(n, 1.0), jnp.full(n, 1.0), jnp.zeros(n)
        a2, b2, c2 = jnp.full(n, 2.0), jnp.full(n, 2.0), jnp.zeros(n)
        theta = jnp.zeros(n)
        
        x_t1, y_t1 = richards_velocity_jax(x, y, a1, b1, c1, theta)
        x_t2, y_t2 = richards_velocity_jax(x, y, a2, b2, c2, theta)
        
        v1 = jnp.sqrt(x_t1**2 + y_t1**2)
        v2 = jnp.sqrt(x_t2**2 + y_t2**2)
        
        assert jnp.mean(v2) > jnp.mean(v1), "Larger ellipse should give faster spread"


class TestPerimeterEvolution:
    """Test perimeter evolution."""
    
    def test_area_increases(self):
        """Fire area should increase over time (for spreading fire)."""
        n = 100
        x, y = create_initial_perimeter_jax(0.0, 0.0, 10.0, n)
        
        # Uniform ROS conditions
        ros = jnp.full(n, 5.0)
        bros = jnp.full(n, 1.0)
        fros = jnp.full(n, 2.0)
        raz = jnp.full(n, 0.0)
        dt = 1.0
        
        area_before = compute_fire_area_jax(x, y)
        
        x_new, y_new = evolve_perimeter_step_jax(x, y, ros, bros, fros, raz, dt)
        
        area_after = compute_fire_area_jax(x_new, y_new)
        
        assert area_after > area_before, "Fire area should increase"
    
    def test_multiple_steps(self):
        """Area should continue to increase over multiple steps."""
        n = 100
        x, y = create_initial_perimeter_jax(0.0, 0.0, 10.0, n)
        
        ros = jnp.full(n, 3.0)
        bros = jnp.full(n, 0.6)
        fros = jnp.full(n, 1.2)
        raz = jnp.full(n, 0.0)
        dt = 1.0
        
        areas = [compute_fire_area_jax(x, y)]
        
        for _ in range(5):
            x, y = evolve_perimeter_step_jax(x, y, ros, bros, fros, raz, dt)
            areas.append(compute_fire_area_jax(x, y))
        
        # Check monotonic increase
        for i in range(1, len(areas)):
            assert areas[i] > areas[i-1], f"Area should increase at step {i}"


class TestFireArea:
    """Test fire area calculation."""
    
    def test_circle_area(self):
        """Area of circular perimeter should match πr²."""
        radius = 100.0
        n = 500  # Many vertices for accuracy
        x, y = create_initial_perimeter_jax(0.0, 0.0, radius, n)
        
        computed_area = compute_fire_area_jax(x, y)
        expected_area = jnp.pi * radius**2
        
        # Should be within 1% for many vertices
        relative_error = abs(computed_area - expected_area) / expected_area
        assert relative_error < 0.01, f"Circle area error: {relative_error:.2%}"


# =============================================================================
# Calibration Tests
# =============================================================================


class TestForwardModel:
    """Test forward model execution."""
    
    def test_forward_model_runs(self):
        """Forward model should run without errors."""
        params = default_params()
        obs = create_synthetic_observation()
        
        x, y, area = run_fire_forward_model(params, obs)
        
        assert len(x) > 0, "Should return perimeter vertices"
        assert area > 0, "Area should be positive"
    
    def test_forward_model_is_differentiable(self):
        """Should be able to compute gradients through forward model."""
        obs = create_synthetic_observation(observed_area=20000.0)
        
        def loss_fn(params_array):
            params = FBPCalibrationParams(
                wind_adj=params_array[0],
                ffmc_adj=params_array[1],
            )
            _, _, area = run_fire_forward_model(params, obs)
            return (area - obs.observed_area) ** 2
        
        grad_fn = jax.grad(loss_fn)
        params_array = jnp.array([1.0, 0.0])
        
        grads = grad_fn(params_array)
        
        assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"


class TestSyntheticObservation:
    """Test synthetic observation creation."""
    
    def test_creates_valid_observation(self):
        """Should create valid observation with all required fields."""
        obs = create_synthetic_observation(
            fire_id="test",
            observed_area=10000.0,
            duration_min=60.0,
        )
        
        assert obs.fire_id == "test"
        assert obs.observed_area == 10000.0
        assert obs.duration_min == 60.0
        assert obs.fuel_grid.shape[0] > 0
        assert len(obs.x_coords) > 0


# =============================================================================
# Gradient Tests
# =============================================================================


class TestGradients:
    """Test gradient computation."""
    
    def test_isi_gradients(self):
        """ISI should have well-defined gradients."""
        def isi_loss(ffmc, wind_speed):
            params = default_params()
            isi = calculate_isi_jax(ffmc, wind_speed, params)
            return isi
        
        grad_fn = jax.grad(isi_loss, argnums=(0, 1))
        
        ffmc = jnp.array(90.0)
        wind_speed = jnp.array(20.0)
        
        grad_ffmc, grad_wind = grad_fn(ffmc, wind_speed)
        
        assert jnp.isfinite(grad_ffmc), "Gradient w.r.t. FFMC should be finite"
        assert jnp.isfinite(grad_wind), "Gradient w.r.t. wind should be finite"
        assert grad_ffmc > 0, "dISI/dFFMC should be positive"
        assert grad_wind > 0, "dISI/dWind should be positive"
    
    def test_area_gradients(self):
        """Area computation should have well-defined gradients."""
        def area_fn(radius):
            x, y = create_initial_perimeter_jax(0.0, 0.0, radius, 100)
            return compute_fire_area_jax(x, y)
        
        grad_fn = jax.grad(area_fn)
        radius = jnp.array(50.0)
        
        grad_radius = grad_fn(radius)
        
        # dA/dr = 2πr for circle
        expected_grad = 2 * jnp.pi * radius
        
        assert jnp.isfinite(grad_radius), "Gradient should be finite"
        # Allow 10% error due to discretization
        assert abs(grad_radius - expected_grad) / expected_grad < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

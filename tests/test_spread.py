"""
Tests for the spread module.
"""

import numpy as np
import pytest

from ignacio.spread import (
    FireParameterGrid,
    FirePerimeterHistory,
    compute_active_vertices,
    compute_spatial_derivatives,
    compute_turning_number,
    create_initial_perimeter,
    evolve_perimeter_step,
    richards_velocity,
    ros_to_ellipse_params,
)


class TestInitialPerimeter:
    """Tests for initial perimeter creation."""
    
    def test_create_circular_perimeter(self):
        """Test circular perimeter creation."""
        x, y = create_initial_perimeter(
            x_center=100.0,
            y_center=200.0,
            radius=10.0,
            n_vertices=36,
        )
        
        assert len(x) == 36
        assert len(y) == 36
        
        # Check center
        assert np.mean(x) == pytest.approx(100.0, abs=0.01)
        assert np.mean(y) == pytest.approx(200.0, abs=0.01)
        
        # Check radius
        distances = np.sqrt((x - 100.0)**2 + (y - 200.0)**2)
        assert np.allclose(distances, 10.0, atol=0.01)
    
    def test_perimeter_vertex_count(self):
        """Test that vertex count is respected."""
        for n in [10, 50, 100, 300]:
            x, y = create_initial_perimeter(0, 0, 1.0, n)
            assert len(x) == n
            assert len(y) == n


class TestSpatialDerivatives:
    """Tests for spatial derivative computation."""
    
    def test_circle_derivatives(self):
        """Test derivatives on a circle."""
        n = 100
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        
        x_s, y_s = compute_spatial_derivatives(x, y)
        
        # Derivatives should have same magnitude around circle
        magnitudes = np.hypot(x_s, y_s)
        assert np.allclose(magnitudes, magnitudes[0], rtol=0.1)
    
    def test_periodic_boundary(self):
        """Test that derivatives handle periodic boundary."""
        # Create a closed loop (pentagon-ish shape)
        theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        
        x_s, y_s = compute_spatial_derivatives(x, y)
        
        # All derivatives should be defined (no NaN)
        assert np.all(np.isfinite(x_s))
        assert np.all(np.isfinite(y_s))


class TestROSToEllipse:
    """Tests for ROS to ellipse parameter conversion."""
    
    def test_basic_conversion(self):
        """Test basic ellipse parameter conversion."""
        ros = np.array([10.0])
        bros = np.array([2.0])
        fros = np.array([4.0])
        raz = np.array([np.pi / 2])
        
        a, b, c, theta = ros_to_ellipse_params(ros, bros, fros, raz)
        
        assert a[0] == pytest.approx(6.0)  # (10 + 2) / 2
        assert b[0] == pytest.approx(4.0)  # FROS
        assert c[0] == pytest.approx(4.0)  # (10 - 2) / 2
        assert theta[0] == pytest.approx(np.pi / 2)
    
    def test_symmetric_fire(self):
        """Test symmetric fire (ROS = BROS)."""
        ros = np.array([10.0])
        bros = np.array([10.0])
        fros = np.array([5.0])
        raz = np.array([0.0])
        
        a, b, c, theta = ros_to_ellipse_params(ros, bros, fros, raz)
        
        assert c[0] == 0.0  # No asymmetry


class TestRichardsVelocity:
    """Tests for Richards' velocity computation."""
    
    def test_velocity_output_shape(self):
        """Test that velocity output has correct shape."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        a = np.full(n, 5.0)
        b = np.full(n, 2.0)
        c = np.full(n, 3.0)
        theta = np.full(n, 0.0)
        
        x_t, y_t = richards_velocity(x, y, a, b, c, theta)
        
        assert len(x_t) == n
        assert len(y_t) == n
    
    def test_velocity_nonzero(self):
        """Test that velocities are non-zero for non-zero ROS."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        a = np.full(n, 5.0)
        b = np.full(n, 2.0)
        c = np.full(n, 3.0)
        theta = np.full(n, 0.0)
        
        x_t, y_t = richards_velocity(x, y, a, b, c, theta)
        
        # At least some velocities should be non-zero
        assert np.any(x_t != 0) or np.any(y_t != 0)
    
    def test_zero_ros_zero_velocity(self):
        """Test that zero ROS gives zero velocity."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        # All zeros
        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        theta = np.zeros(n)
        
        x_t, y_t = richards_velocity(x, y, a, b, c, theta)
        
        # Velocities should be very small (numerical precision)
        assert np.allclose(x_t, 0.0, atol=1e-6)
        assert np.allclose(y_t, 0.0, atol=1e-6)


class TestTurningNumber:
    """Tests for turning number computation."""
    
    def test_interior_point(self):
        """Test that interior points have non-zero turning number."""
        # Square
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        
        # Interior point
        tn = compute_turning_number(0.5, 0.5, x, y)
        assert tn != 0
    
    def test_exterior_point(self):
        """Test that exterior points have zero turning number."""
        # Square
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        
        # Exterior point
        tn = compute_turning_number(10.0, 10.0, x, y)
        assert tn == 0
    
    def test_circle_interior(self):
        """Test turning number for circular perimeter."""
        n = 100
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Center
        tn = compute_turning_number(0.0, 0.0, x, y)
        assert abs(tn) == 1


class TestActiveVertices:
    """Tests for active vertex detection."""
    
    def test_all_active_on_circle(self):
        """Test that all vertices are active on simple circle."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        active = compute_active_vertices(x, y, epsilon=0.1)
        
        # All should be active on simple convex shape
        assert np.all(active)
    
    def test_active_detection_shape(self):
        """Test that active array has correct shape."""
        n = 100
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        active = compute_active_vertices(x, y)
        
        assert len(active) == n
        assert active.dtype == bool


class TestPerimeterEvolution:
    """Tests for perimeter evolution."""
    
    def test_evolve_step_shape(self):
        """Test that evolved perimeter has same vertex count."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        ros = np.full(n, 5.0)
        bros = np.full(n, 1.0)
        fros = np.full(n, 2.0)
        raz = np.zeros(n)
        
        x_new, y_new = evolve_perimeter_step(
            x, y, ros, bros, fros, raz, dt=1.0,
        )
        
        assert len(x_new) == n
        assert len(y_new) == n
    
    def test_evolve_increases_area(self):
        """Test that fire perimeter grows with positive ROS."""
        n = 50
        x, y = create_initial_perimeter(0, 0, 10.0, n)
        
        # Compute initial area (shoelace)
        initial_area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        
        ros = np.full(n, 5.0)
        bros = np.full(n, 1.0)
        fros = np.full(n, 2.0)
        raz = np.zeros(n)
        
        x_new, y_new = evolve_perimeter_step(
            x, y, ros, bros, fros, raz, dt=1.0,
            use_markers=False,  # Disable markers for simple test
        )
        
        final_area = 0.5 * np.abs(np.sum(x_new * np.roll(y_new, -1) - np.roll(x_new, -1) * y_new))
        
        assert final_area > initial_area


class TestFireParameterGrid:
    """Tests for FireParameterGrid class."""
    
    def test_grid_initialization(self):
        """Test grid initialization."""
        nx, ny, nt = 10, 10, 5
        x_coords = np.arange(nx) * 10.0
        y_coords = np.arange(ny) * 10.0
        
        grid = FireParameterGrid(
            x_coords=x_coords,
            y_coords=y_coords,
            ros=np.ones((nt, ny, nx)),
            bros=np.ones((nt, ny, nx)) * 0.2,
            fros=np.ones((nt, ny, nx)) * 0.5,
            raz=np.zeros((nt, ny, nx)),
        )
        
        assert grid.nx == nx
        assert grid.ny == ny
        assert grid.nt == nt
    
    def test_sample_at(self):
        """Test sampling at arbitrary positions."""
        nx, ny, nt = 10, 10, 5
        x_coords = np.arange(nx) * 10.0
        y_coords = np.arange(ny) * 10.0
        
        ros_val = 5.0
        grid = FireParameterGrid(
            x_coords=x_coords,
            y_coords=y_coords,
            ros=np.full((nt, ny, nx), ros_val),
            bros=np.full((nt, ny, nx), 1.0),
            fros=np.full((nt, ny, nx), 2.0),
            raz=np.zeros((nt, ny, nx)),
        )
        
        # Sample at center
        x = np.array([45.0])
        y = np.array([45.0])
        
        ros, bros, fros, raz = grid.sample_at(0, x, y)
        
        assert ros[0] == pytest.approx(ros_val, rel=0.01)


class TestFirePerimeterHistory:
    """Tests for FirePerimeterHistory class."""
    
    def test_history_storage(self):
        """Test history storage."""
        perimeters = [
            (np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])),
            (np.array([0, 2, 2, 0]), np.array([0, 0, 2, 2])),
        ]
        times = [0.0, 1.0]
        
        history = FirePerimeterHistory(perimeters=perimeters, times=times)
        
        assert history.n_steps == 2
    
    def test_get_final_perimeter(self):
        """Test getting final perimeter."""
        perimeters = [
            (np.array([0, 1]), np.array([0, 1])),
            (np.array([0, 2]), np.array([0, 2])),
        ]
        times = [0.0, 1.0]
        
        history = FirePerimeterHistory(perimeters=perimeters, times=times)
        
        x, y = history.get_final_perimeter()
        assert np.array_equal(x, np.array([0, 2]))
    
    def test_compute_areas(self):
        """Test area computation."""
        # Square with area 1
        perimeters = [
            (np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])),
        ]
        times = [0.0]
        
        history = FirePerimeterHistory(perimeters=perimeters, times=times)
        
        areas = history.compute_areas()
        assert areas[0] == pytest.approx(1.0)

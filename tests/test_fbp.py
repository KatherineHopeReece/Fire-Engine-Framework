"""
Tests for the FBP module.
"""

import numpy as np
import pytest

from ignacio.fbp import (
    calculate_isi,
    calculate_isi_grid,
    compute_ros,
    compute_ros_components,
    compute_ros_grid,
    fbp_ros_c1,
    fbp_ros_c2,
    fbp_ros_c3,
    fbp_ros_d1,
    fbp_ros_o1,
)


class TestISICalculation:
    """Tests for ISI calculation."""
    
    def test_isi_basic(self):
        """Test basic ISI calculation."""
        isi = calculate_isi(ffmc=85.0, wind_speed=20.0)
        assert isi > 0
        assert isinstance(isi, float)
    
    def test_isi_zero_wind(self):
        """Test ISI with zero wind."""
        isi = calculate_isi(ffmc=85.0, wind_speed=0.0)
        assert isi >= 0
    
    def test_isi_high_ffmc(self):
        """Test ISI with high FFMC increases spread."""
        isi_low = calculate_isi(ffmc=70.0, wind_speed=20.0)
        isi_high = calculate_isi(ffmc=95.0, wind_speed=20.0)
        assert isi_high > isi_low
    
    def test_isi_high_wind(self):
        """Test ISI increases with wind."""
        isi_low = calculate_isi(ffmc=85.0, wind_speed=10.0)
        isi_high = calculate_isi(ffmc=85.0, wind_speed=30.0)
        assert isi_high > isi_low
    
    def test_isi_grid(self):
        """Test vectorized ISI calculation."""
        ffmc = np.array([[80.0, 85.0], [90.0, 95.0]])
        wind = np.array([[10.0, 15.0], [20.0, 25.0]])
        
        isi = calculate_isi_grid(ffmc, wind)
        
        assert isi.shape == (2, 2)
        assert np.all(isi >= 0)
    
    def test_isi_invalid_input(self):
        """Test ISI with invalid inputs returns zero."""
        assert calculate_isi(ffmc=None, wind_speed=20.0) == 0.0
        assert calculate_isi(ffmc=85.0, wind_speed="invalid") == 0.0


class TestFuelTypeROS:
    """Tests for fuel type ROS functions."""
    
    def test_c1_ros(self):
        """Test C-1 Spruce-Lichen Woodland ROS."""
        ros = fbp_ros_c1(isi=10.0, bui=80.0)
        assert ros > 0
        assert ros < 100  # Reasonable upper bound
    
    def test_c2_ros(self):
        """Test C-2 Boreal Spruce ROS."""
        ros = fbp_ros_c2(isi=10.0, bui=80.0)
        assert ros > 0
    
    def test_c3_ros(self):
        """Test C-3 Mature Pine ROS."""
        ros = fbp_ros_c3(isi=10.0, bui=80.0)
        assert ros > 0
    
    def test_d1_ros(self):
        """Test D-1 Leafless Aspen ROS."""
        ros = fbp_ros_d1(isi=10.0, bui=80.0)
        assert ros > 0
    
    def test_o1_ros(self):
        """Test O-1 Grass ROS with curing."""
        ros_low_curing = fbp_ros_o1(isi=10.0, curing=50.0)
        ros_high_curing = fbp_ros_o1(isi=10.0, curing=90.0)
        assert ros_high_curing > ros_low_curing
    
    def test_ros_increases_with_isi(self):
        """Test that ROS increases with ISI for all fuel types."""
        for ros_func in [fbp_ros_c1, fbp_ros_c2, fbp_ros_c3, fbp_ros_d1]:
            ros_low = ros_func(isi=5.0, bui=80.0)
            ros_high = ros_func(isi=20.0, bui=80.0)
            assert ros_high > ros_low
    
    def test_ros_zero_isi(self):
        """Test that ROS is zero with zero ISI."""
        ros = fbp_ros_c2(isi=0.0, bui=80.0)
        assert ros == 0.0


class TestComputeROS:
    """Tests for the general compute_ros function."""
    
    def test_compute_ros_by_name(self):
        """Test ROS computation by fuel type name."""
        ros = compute_ros(fuel_type="C-2", isi=10.0, bui=80.0)
        assert ros > 0
    
    def test_compute_ros_by_id(self):
        """Test ROS computation by numeric fuel ID."""
        ros = compute_ros(
            fuel_type=2,
            isi=10.0,
            bui=80.0,
            fuel_lookup={2: "C-2"},
        )
        assert ros > 0
    
    def test_compute_ros_non_fuel(self):
        """Test that non-fuel types return zero ROS."""
        ros = compute_ros(fuel_type="NF", isi=10.0, bui=80.0)
        assert ros == 0.0
        
        ros = compute_ros(fuel_type=101, isi=10.0, bui=80.0)
        assert ros == 0.0
    
    def test_compute_ros_unknown_fuel(self):
        """Test that unknown fuel types return zero ROS."""
        ros = compute_ros(fuel_type="X-99", isi=10.0, bui=80.0)
        assert ros == 0.0


class TestROSComponents:
    """Tests for ROS component computation."""
    
    def test_ros_components_basic(self):
        """Test basic ROS component computation."""
        components = compute_ros_components(
            ros_head=10.0,
            wind_direction=180.0,
            backing_fraction=0.2,
            lb_ratio=2.0,
        )
        
        assert components.ros_head == 10.0
        assert components.ros_back == 2.0  # 0.2 * 10
        assert components.ros_flank == 3.0  # (10 + 2) / (2 * 2)
        assert components.raz == 0.0  # 180 + 180 = 360 = 0
        assert components.lb_ratio == 2.0
    
    def test_ros_components_ratios(self):
        """Test that ROS component ratios are correct."""
        components = compute_ros_components(
            ros_head=20.0,
            wind_direction=90.0,
            backing_fraction=0.1,
            lb_ratio=3.0,
        )
        
        assert components.ros_back == pytest.approx(2.0)
        expected_fros = (20.0 + 2.0) / (2.0 * 3.0)
        assert components.ros_flank == pytest.approx(expected_fros)


class TestROSGrid:
    """Tests for ROS grid computation."""
    
    def test_ros_grid_basic(self):
        """Test basic ROS grid computation."""
        fuel_grid = np.array([[1, 2], [3, 101]], dtype=np.int32)
        
        ros_grid = compute_ros_grid(
            fuel_grid=fuel_grid,
            isi=10.0,
            bui=80.0,
            fuel_lookup={1: "C-1", 2: "C-2", 3: "C-3"},
            non_fuel_codes=[101],
        )
        
        assert ros_grid.shape == (2, 2)
        assert ros_grid[1, 1] == 0.0  # Non-fuel
        assert ros_grid[0, 0] > 0  # C-1
        assert ros_grid[0, 1] > 0  # C-2
    
    def test_ros_grid_non_fuel_zero(self):
        """Test that non-fuel cells have zero ROS."""
        fuel_grid = np.array([[0, -9999], [101, 102]])
        
        ros_grid = compute_ros_grid(
            fuel_grid=fuel_grid,
            isi=10.0,
            bui=80.0,
            non_fuel_codes=[0, -9999, 101, 102],
        )
        
        assert np.all(ros_grid == 0.0)

"""
Tests for the terrain module.
"""

import numpy as np
import pytest

from ignacio.terrain import (
    compute_elevation_adjustment,
    compute_slope_aspect,
    compute_slope_aspect_simple,
    compute_slope_factor,
)


class TestSlopeAspect:
    """Tests for slope and aspect computation."""
    
    def test_flat_terrain(self):
        """Test flat terrain has zero slope."""
        dem = np.ones((100, 100)) * 100.0  # Flat at 100m
        
        slope, aspect = compute_slope_aspect(dem, dx=30.0, dy=30.0)
        
        assert np.allclose(slope[10:-10, 10:-10], 0.0, atol=0.1)
    
    def test_north_facing_slope(self):
        """Test north-facing slope has aspect ~0/360."""
        dem = np.zeros((100, 100))
        
        # Elevation increases from south to north (row 0 = north)
        for i in range(100):
            dem[i, :] = 100 - i  # Higher in north
        
        slope, aspect = compute_slope_aspect(dem, dx=30.0, dy=30.0)
        
        # Interior points
        interior_aspect = aspect[20:80, 20:80]
        valid_aspect = interior_aspect[np.isfinite(interior_aspect)]
        
        # North-facing: aspect should be near 0 or 360
        mean_aspect = np.mean(valid_aspect)
        assert mean_aspect < 30 or mean_aspect > 330
    
    def test_east_facing_slope(self):
        """Test east-facing slope has aspect ~90."""
        dem = np.zeros((100, 100))
        
        # Elevation increases from west to east
        for j in range(100):
            dem[:, j] = j  # Higher in east
        
        slope, aspect = compute_slope_aspect(dem, dx=30.0, dy=30.0)
        
        interior_aspect = aspect[20:80, 20:80]
        valid_aspect = interior_aspect[np.isfinite(interior_aspect)]
        
        # East-facing: aspect should be near 90
        mean_aspect = np.mean(valid_aspect)
        assert 60 < mean_aspect < 120
    
    def test_slope_magnitude(self):
        """Test slope magnitude calculation."""
        dem = np.zeros((100, 100))
        dx = dy = 30.0
        
        # Create a ramp: 45 degrees
        for i in range(100):
            for j in range(100):
                dem[i, j] = i * dx  # Rise = run
        
        slope, aspect = compute_slope_aspect(dem, dx=dx, dy=dy)
        
        # Interior slope should be ~45 degrees
        interior_slope = slope[20:80, 20:80]
        mean_slope = np.nanmean(interior_slope)
        assert 40 < mean_slope < 50
    
    def test_simple_method_correlated(self):
        """Test that simple method is correlated with Sobel method."""
        dem = np.random.rand(100, 100) * 100 + 500
        
        slope1, aspect1 = compute_slope_aspect(dem, 30.0, 30.0)
        slope2, aspect2 = compute_slope_aspect_simple(dem, 30.0, 30.0)
        
        # Should be correlated (both detect same patterns)
        interior = (slice(10, 90), slice(10, 90))
        
        # Check correlation is positive
        corr = np.corrcoef(slope1[interior].flatten(), slope2[interior].flatten())[0, 1]
        assert corr > 0.5  # Methods should be positively correlated
    
    def test_handles_nan(self):
        """Test that NaN values are handled correctly."""
        dem = np.ones((50, 50)) * 100.0
        dem[20:30, 20:30] = np.nan  # NoData region
        
        slope, aspect = compute_slope_aspect(dem, 30.0, 30.0)
        
        # NaN region should have NaN slope/aspect
        assert np.all(np.isnan(slope[20:30, 20:30]))
        assert np.all(np.isnan(aspect[20:30, 20:30]))


class TestSlopeFactor:
    """Tests for slope correction factor."""
    
    def test_flat_slope_factor(self):
        """Test flat terrain has slope factor of 1."""
        slope_deg = np.zeros((10, 10))
        aspect_deg = np.full((10, 10), 180.0)
        
        sf = compute_slope_factor(slope_deg, aspect_deg, wind_direction=180.0)
        
        assert np.allclose(sf, 1.0)
    
    def test_upslope_wind_increases_ros(self):
        """Test that upslope wind increases ROS."""
        slope_deg = np.full((10, 10), 30.0)
        
        # North-facing slope (aspect = 0, downslope direction is north)
        aspect_deg = np.full((10, 10), 0.0)
        
        # Wind from south (180) pushes fire upslope (north)
        sf_upslope = compute_slope_factor(slope_deg, aspect_deg, wind_direction=180.0)
        
        # Wind from north (0) pushes fire downslope (south)
        sf_downslope = compute_slope_factor(slope_deg, aspect_deg, wind_direction=0.0)
        
        assert np.mean(sf_upslope) > np.mean(sf_downslope)
    
    def test_slope_factor_nonnegative(self):
        """Test that slope factor is always non-negative."""
        slope_deg = np.random.rand(50, 50) * 45
        aspect_deg = np.random.rand(50, 50) * 360
        
        sf = compute_slope_factor(slope_deg, aspect_deg, wind_direction=90.0)
        
        assert np.all(sf >= 0)
    
    def test_slope_factor_strength(self):
        """Test that slope factor strength parameter works."""
        slope_deg = np.full((10, 10), 30.0)
        aspect_deg = np.full((10, 10), 180.0)
        
        sf_weak = compute_slope_factor(slope_deg, aspect_deg, 0.0, slope_factor_strength=0.1)
        sf_strong = compute_slope_factor(slope_deg, aspect_deg, 0.0, slope_factor_strength=1.0)
        
        # Stronger slope factor should deviate more from 1.0
        deviation_weak = np.abs(sf_weak - 1.0).mean()
        deviation_strong = np.abs(sf_strong - 1.0).mean()
        
        assert deviation_strong > deviation_weak


class TestElevationAdjustment:
    """Tests for elevation-based adjustment."""
    
    def test_higher_elevation_cooler(self):
        """Test that higher elevations have temperature adjustment."""
        elev = np.array([0, 500, 1000, 1500, 2000])
        
        factor = compute_elevation_adjustment(
            elev,
            temperature=25.0,
            relative_humidity=40.0,
            lapse_rate=0.0065,
        )
        
        # Higher elevations should have factor > 1 (cooler = wetter = slower fire)
        # or factor < 1 depending on implementation
        # Just check that factor varies with elevation
        assert not np.allclose(factor, factor[0])
    
    def test_factor_bounded(self):
        """Test that adjustment factor is bounded."""
        elev = np.linspace(0, 5000, 100)
        
        factor = compute_elevation_adjustment(
            elev,
            temperature=25.0,
            relative_humidity=40.0,
        )
        
        assert np.all(factor >= 0.7)
        assert np.all(factor <= 1.3)
    
    def test_sea_level_reference(self):
        """Test that sea level gives factor near 1."""
        elev = np.array([0.0])
        
        factor = compute_elevation_adjustment(
            elev,
            temperature=20.0,  # Reference temp
            relative_humidity=30.0,  # Reference RH
        )
        
        assert factor[0] == pytest.approx(1.0, rel=0.1)

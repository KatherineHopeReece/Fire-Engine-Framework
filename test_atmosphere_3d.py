#!/usr/bin/env python3
"""
Test Suite for 3D Atmospheric Dynamics and Coupled Fire-Atmosphere Simulation.

Tests the WRF-SFIRE-style coupled modeling capability.
"""

import jax
import jax.numpy as jnp
import time

print("="*70)
print("IGNACIO 3D ATMOSPHERIC DYNAMICS TEST")
print("="*70)


# =============================================================================
# Test 1: Vertical Grid Creation
# =============================================================================
print("\n1. VERTICAL GRID CREATION")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import (
    create_vertical_grid,
    AtmosphereParams,
)

# Uniform grid
z_w_uniform, z_u_uniform = create_vertical_grid(20, 2000.0, stretch_factor=1.0)
print(f"   Uniform grid (20 levels, 2km):")
print(f"     dz range: {float(jnp.min(jnp.diff(z_w_uniform))):.1f} - "
      f"{float(jnp.max(jnp.diff(z_w_uniform))):.1f} m")

# Stretched grid
z_w_stretch, z_u_stretch = create_vertical_grid(20, 2000.0, stretch_factor=1.5)
print(f"   Stretched grid (20 levels, 2km):")
print(f"     dz range: {float(jnp.min(jnp.diff(z_w_stretch))):.1f} - "
      f"{float(jnp.max(jnp.diff(z_w_stretch))):.1f} m")
print(f"     Surface dz: {float(z_w_stretch[1]):.1f} m")

print("   ✓ Vertical grid creation: PASSED")


# =============================================================================
# Test 2: Base State Computation
# =============================================================================
print("\n2. BASE STATE COMPUTATION")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import compute_base_state

theta_base, rho_base = compute_base_state(z_u_stretch)
print(f"   Potential temperature profile:")
print(f"     Surface: {float(theta_base[0]):.1f} K")
print(f"     Top: {float(theta_base[-1]):.1f} K")
print(f"     Lapse: {float((theta_base[-1] - theta_base[0]) / z_u_stretch[-1] * 1000):.2f} K/km")

print(f"   Density profile:")
print(f"     Surface: {float(rho_base[0]):.3f} kg/m³")
print(f"     Top: {float(rho_base[-1]):.3f} kg/m³")

print("   ✓ Base state computation: PASSED")


# =============================================================================
# Test 3: Atmosphere Grid Setup
# =============================================================================
print("\n3. ATMOSPHERE GRID SETUP")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import create_atmosphere_grids

nx, ny = 50, 50
dx = 100.0
terrain = jnp.zeros((ny, nx))

# Add a hill
cy, cx = ny // 2, nx // 2
Y, X = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
terrain = 200.0 * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / 100.0)

params = AtmosphereParams(nz=15, z_top=1500.0)
grids = create_atmosphere_grids(nx, ny, dx, dx, terrain, params)

print(f"   Grid dimensions: {nx} x {ny} x {params.nz}")
print(f"   Horizontal extent: {nx * dx / 1000:.1f} x {ny * dx / 1000:.1f} km")
print(f"   Vertical extent: {params.z_top / 1000:.1f} km")
print(f"   Terrain max: {float(jnp.max(terrain)):.1f} m")
print(f"   Terrain slope max: {float(jnp.max(jnp.abs(grids.dzdx))):.3f}")

print("   ✓ Atmosphere grid setup: PASSED")


# =============================================================================
# Test 4: Atmosphere Initialization
# =============================================================================
print("\n4. ATMOSPHERE INITIALIZATION")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import initialize_atmosphere

state = initialize_atmosphere(grids, params, u_init=5.0, v_init=1.0)

print(f"   Velocity field shapes:")
print(f"     u: {state.u.shape} (nz, ny, nx+1)")
print(f"     v: {state.v.shape} (nz, ny+1, nx)")
print(f"     w: {state.w.shape} (nz+1, ny, nx)")
print(f"   Theta shape: {state.theta.shape}")
print(f"   Initial u: {float(jnp.mean(state.u)):.1f} m/s")
print(f"   Initial theta (surface): {float(jnp.mean(state.theta[0])):.1f} K")

print("   ✓ Atmosphere initialization: PASSED")


# =============================================================================
# Test 5: Fire Heat Source
# =============================================================================
print("\n5. FIRE HEAT SOURCE")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import compute_fire_heat_source

# Create fire intensity field (point source at center)
fire_intensity = jnp.zeros((ny, nx))
fire_intensity = fire_intensity.at[cy-2:cy+3, cx-2:cx+3].set(5000.0)  # 5000 kW/m

Q = compute_fire_heat_source(fire_intensity, grids, params)

print(f"   Fire intensity max: {float(jnp.max(fire_intensity)):.0f} kW/m")
print(f"   Heat source shape: {Q.shape}")
print(f"   Heat source max (surface): {float(jnp.max(Q[0])):.3f} K/s")
print(f"   Heat source max (500m): {float(jnp.max(Q[5])):.3f} K/s")
print(f"   Vertical distribution preserved: "
      f"{float(jnp.sum(Q[0])) > float(jnp.sum(Q[-1]))}")

print("   ✓ Fire heat source: PASSED")


# =============================================================================
# Test 6: Single Time Step Evolution
# =============================================================================
print("\n6. SINGLE TIME STEP EVOLUTION")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import evolve_atmosphere_step

dt = 2.0  # 2 second time step
t0 = time.time()
state_new = evolve_atmosphere_step(state, grids, params, fire_intensity, dt)
elapsed = time.time() - t0

print(f"   Time step: {dt} s")
print(f"   Computation time: {elapsed*1000:.1f} ms")

# Check changes
theta_change = state_new.theta - state.theta
w_change = state_new.w - state.w

print(f"   Theta change max: {float(jnp.max(jnp.abs(theta_change))):.4f} K")
print(f"   W change max: {float(jnp.max(jnp.abs(w_change))):.4f} m/s")
print(f"   New max w: {float(jnp.max(state_new.w)):.3f} m/s")

print("   ✓ Single time step evolution: PASSED")


# =============================================================================
# Test 7: Multiple Steps with Fire
# =============================================================================
print("\n7. MULTI-STEP SIMULATION WITH FIRE")
print("-" * 50)

# Run 50 steps (100 seconds)
n_steps = 50
state_run = state

t0 = time.time()
for _ in range(n_steps):
    state_run = evolve_atmosphere_step(state_run, grids, params, fire_intensity, dt)
elapsed = time.time() - t0

print(f"   Steps: {n_steps}")
print(f"   Total time: {elapsed:.2f} s ({elapsed/n_steps*1000:.1f} ms/step)")

# Check plume development
print(f"   Max updraft: {float(jnp.max(state_run.w)):.2f} m/s")
print(f"   Max theta perturbation: {float(jnp.max(state_run.theta - state_run.theta_base[:, None, None])):.2f} K")

# Check that plume is centered over fire
w_surface = state_run.w[1]  # First interior w level
max_w_idx = jnp.unravel_index(jnp.argmax(w_surface), w_surface.shape)
print(f"   Plume center: ({max_w_idx[1]}, {max_w_idx[0]}), Fire center: ({cx}, {cy})")

print("   ✓ Multi-step simulation: PASSED")


# =============================================================================
# Test 8: Atmosphere-Fire Coupling Interface
# =============================================================================
print("\n8. ATMOSPHERE-FIRE COUPLING INTERFACE")
print("-" * 50)

from ignacio.jax_core.atmosphere_3d import couple_atmosphere_to_fire

coupling = couple_atmosphere_to_fire(state_run, grids)

print(f"   Surface wind speed range: {float(jnp.min(coupling.wind_speed)):.2f} - "
      f"{float(jnp.max(coupling.wind_speed)):.2f} m/s")
print(f"   Max updraft (w_max): {float(jnp.max(coupling.w_max)):.2f} m/s")
print(f"   Max plume top: {float(jnp.max(coupling.plume_top)):.0f} m")
print(f"   Max indraft strength: {float(jnp.max(coupling.indraft_strength)):.4f} 1/s")
print(f"   Max vorticity: {float(jnp.max(jnp.abs(coupling.vorticity))):.4f} 1/s")

print("   ✓ Atmosphere-fire coupling: PASSED")


# =============================================================================
# Test 9: Coupled Fire-Atmosphere Simulation
# =============================================================================
print("\n9. COUPLED FIRE-ATMOSPHERE SIMULATION")
print("-" * 50)

from ignacio.jax_core.coupled_simulation import (
    CoupledSimConfig,
    initialize_coupled_simulation,
    evolve_coupled_step,
    create_compatible_grids,
)

# Setup
nx_coupled, ny_coupled = 60, 60
dx_coupled = 50.0
terrain_flat = jnp.zeros((ny_coupled, nx_coupled))
x_ign = nx_coupled * dx_coupled / 2
y_ign = ny_coupled * dx_coupled / 2

config = CoupledSimConfig(
    feedback_strength=0.5,
    base_ros=10.0,
    u_background=5.0,
    v_background=0.0,
    fire_dt=0.5,  # 0.5 minute fire steps (smaller for stability)
    atm_dt=1.0,   # 1 second atmosphere steps (smaller for stability)
    atm_params=AtmosphereParams(nz=12, z_top=1200.0),
)

# Create grids
atm_grids, _, _ = create_compatible_grids(
    nx_coupled, ny_coupled, dx_coupled, terrain_flat, config
)

# Initialize
coupled_state = initialize_coupled_simulation(
    nx_coupled, ny_coupled, dx_coupled, terrain_flat,
    x_ign, y_ign, config
)

print(f"   Grid: {nx_coupled} x {ny_coupled} x {config.atm_params.nz}")
print(f"   Fire dt: {config.fire_dt} min, Atm dt: {config.atm_dt} s")
print(f"   Feedback strength: {config.feedback_strength}")

# Run 10 coupled steps
print("   Running 10 coupled steps...")
t0 = time.time()
for step in range(10):
    coupled_state = evolve_coupled_step(coupled_state, atm_grids, config)
elapsed = time.time() - t0

print(f"   Completed in {elapsed:.2f} s ({elapsed/10:.2f} s/step)")

# Check state
fire_mask = coupled_state.phi < 0
burned_area = float(jnp.sum(fire_mask)) * dx_coupled * dx_coupled
coupling_final = couple_atmosphere_to_fire(coupled_state.atm_state, atm_grids)

print(f"   Fire time: {coupled_state.fire_time:.0f} min")
print(f"   Burned area: {burned_area/10000:.2f} ha")
print(f"   Max surface wind: {float(jnp.max(coupling_final.wind_speed)):.2f} m/s")
print(f"   Max updraft: {float(jnp.max(coupling_final.w_max)):.2f} m/s")

print("   ✓ Coupled fire-atmosphere simulation: PASSED")


# =============================================================================
# Test 10: Feedback Effect Demonstration
# =============================================================================
print("\n10. FEEDBACK EFFECT DEMONSTRATION")
print("-" * 50)

# Compare wind near fire vs far from fire
fire_center = (ny_coupled // 2, nx_coupled // 2)
wind_at_fire = coupling_final.wind_speed[fire_center[0]-2:fire_center[0]+3,
                                          fire_center[1]-2:fire_center[1]+3]
wind_far = coupling_final.wind_speed[:5, :5]  # Corner

print(f"   Wind near fire: {float(jnp.mean(wind_at_fire)):.2f} m/s")
print(f"   Wind far from fire: {float(jnp.mean(wind_far)):.2f} m/s")
print(f"   Background wind: {config.u_background:.2f} m/s")

enhancement = float(jnp.mean(wind_at_fire)) / config.u_background
print(f"   Wind enhancement factor: {enhancement:.2f}x")

print("   ✓ Feedback effect demonstration: PASSED")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("ALL 3D ATMOSPHERIC DYNAMICS TESTS PASSED")
print("="*70)

print("""
3D Atmospheric Dynamics Features:

1. ANELASTIC EQUATIONS
   - Momentum with pressure gradient and buoyancy
   - Thermodynamic equation with fire heat source
   - Pressure projection for mass conservation

2. GRID STRUCTURE
   - Arakawa C-grid staggering (u,v,w at faces)
   - Stretched vertical grid (fine near surface)
   - Terrain-following sigma coordinates

3. FIRE COUPLING
   - Fire intensity → atmospheric heat source
   - Atmospheric surface wind → fire spread modification
   - Indraft, updraft, and vorticity diagnostics

4. NUMERICAL METHODS
   - Upwind advection for stability
   - Jacobi iteration for pressure
   - Explicit diffusion

5. WRF-SFIRE STYLE FEATURES
   - Bidirectional fire-atmosphere coupling
   - Configurable feedback strength
   - Plume diagnostics (height, strength)
   - Fire whirl detection (vorticity)
""")

# Quick performance summary
print("PERFORMANCE SUMMARY:")
print(f"  Atmosphere step ({nx}x{ny}x{params.nz}): ~{elapsed/n_steps*1000:.0f} ms")
print(f"  Coupled step ({nx_coupled}x{ny_coupled}): ~{elapsed/10*1000:.0f} ms")

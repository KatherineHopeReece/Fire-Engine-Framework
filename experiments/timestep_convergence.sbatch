#!/usr/bin/env bash
#SBATCH --job-name=timestep-convergence
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=timestep-convergence-%j.out
#SBATCH --error=timestep-convergence-%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
config="${CONFIG:-ignacio.yaml}"
output_dir="${OUTPUT_DIR:-output/timestep_convergence}"
timesteps="${TIMESTEPS:-60 30 15 7.5 3.75}"
integrators="${INTEGRATORS:-euler heun rk4}"
reference_timestep="${REFERENCE_TIMESTEP:-3.75}"
reference_integrator="${REFERENCE_INTEGRATOR:-rk4}"
duration_minutes="${DURATION_MINUTES:-}"

read -r -a timestep_args <<< "$timesteps"
read -r -a integrator_args <<< "$integrators"

command=(
    "$python_bin" -m experiments.timestep_convergence
    --config "$config"
    --output-dir "$output_dir"
    --timesteps "${timestep_args[@]}"
    --integrators "${integrator_args[@]}"
    --reference-timestep "$reference_timestep"
    --reference-integrator "$reference_integrator"
)
if [[ -n "$duration_minutes" ]]; then
    command+=(--duration-minutes "$duration_minutes")
fi

printf 'Running timestep convergence from %s\n' "$repo_root"
printf 'Config: %s\nOutput: %s\n' "$config" "$output_dir"
printf 'Timesteps: %s\nIntegrators: %s\n' "$timesteps" "$integrators"
"${command[@]}"

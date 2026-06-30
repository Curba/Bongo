"""Smoke test: proves the bootstrap environment can run a tiny YAQS simulation.

Mirrors the working pattern in ../yaqs/experiments/generate_ising_tjm_data.py
(branch: char) — same imports, same AnalogSimParams fields, same solver.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from mqt.yaqs import simulator
        from mqt.yaqs.core.data_structures.networks import MPO, MPS
        from mqt.yaqs.core.data_structures.noise_model import NoiseModel
        from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
        from mqt.yaqs.core.libraries.gate_library import Z
    except ImportError as exc:
        print(f"FAIL: could not import mqt.yaqs ({exc}). Is ../yaqs/ installed editable?", file=sys.stderr)
        return 1

    try:
        length = 3
        j_coupling = 1.0
        transverse_field = 1.0

        hamiltonian = MPO.ising(length, j_coupling, transverse_field)
        initial_state = MPS(length, state="zeros")

        noise_processes = [{"name": "z", "sites": [i], "strength": 1e-2} for i in range(length)]
        noise_model = NoiseModel(processes=noise_processes)

        observables = [Observable(Z(), sites=i) for i in range(length)]
        sim_params = AnalogSimParams(
            observables=observables,
            elapsed_time=0.5,
            dt=0.25,
            solver="TJM",
            threshold=1e-3,
            max_bond_dim=4,
            num_traj=5,
            sample_timesteps=True,
            show_progress=False,
        )

        simulator.run(initial_state, hamiltonian, sim_params, noise_model)

        final_value = sim_params.sorted_observables[0].results[-1]
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: YAQS simulation raised an error ({exc})", file=sys.stderr)
        return 1

    print(f"OK: <Z_0>(t_final) = {final_value:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test: prove the bootstrap environment can run a tiny YAQS simulation.

This uses the class-based API from the required YAQS ``main`` branch.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
    except ImportError as exc:
        print(f"FAIL: could not import mqt.yaqs ({exc}). Is ../yaqs/ installed editable?", file=sys.stderr)
        return 1

    try:
        length = 3
        j_coupling = 1.0
        transverse_field = 1.0

        hamiltonian = Hamiltonian.ising(length, J=j_coupling, g=transverse_field)
        initial_state = State(length, initial="zeros", representation="density_matrix")

        noise_processes = [{"name": "pauli_z", "sites": [i], "strength": 1e-2} for i in range(length)]
        noise_model = NoiseModel(processes=noise_processes)

        observables = [Observable("z", sites=i) for i in range(length)]
        sim_params = AnalogSimParams(
            observables=observables,
            elapsed_time=0.1,
            dt=0.1,
            num_traj=1,
            preset="exact",
        )

        result = Simulator(parallel=False, show_progress=False).run(
            initial_state,
            hamiltonian,
            sim_params,
            noise_model,
        )
        final_value = result.expectation_values[0][-1]
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: YAQS simulation raised an error ({exc})", file=sys.stderr)
        return 1

    print(f"OK: <Z_0>(t_final) = {final_value:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

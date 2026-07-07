from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
except ImportError:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.simulator import Simulator


METHOD_TO_REPRESENTATION = {
    # User-facing aliases
    "analog": "mps",
    "tjm": "mps",
    "mps": "mps",

    "mcwf": "vector",
    "vector": "vector",

    "lindblad": "density_matrix",
    "density_matrix": "density_matrix",

    # Convenient alias
    "exact": "density_matrix",
}


PARAM_NAMES = ["gamma_x", "gamma_y", "gamma_z"]
NOISE_PROCESSES = ["pauli_x", "pauli_y", "pauli_z"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    size: tuple[int, ...],
) -> NDArray[np.float64]:
    if low <= 0 or high <= 0:
        raise ValueError("gamma-min and gamma-max must be positive.")
    if low >= high:
        raise ValueError("gamma-min must be smaller than gamma-max.")

    return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def parse_channels(channels_raw: str) -> list[str]:
    channels = [c.strip().lower() for c in channels_raw.split(",") if c.strip()]
    allowed = {"x", "y", "z"}

    if not channels:
        raise ValueError("At least one observable channel is required.")

    unknown = set(channels) - allowed
    if unknown:
        raise ValueError(f"Unsupported observable channel(s): {sorted(unknown)}. Use x,y,z.")

    return channels


def resolve_method(args: argparse.Namespace) -> tuple[str, str]:
    """
    Returns:
        method: user-facing method name
        representation: YAQS State representation

    YAQS analog backend is chosen from State.representation:
        mps            -> TJM
        vector         -> MCWF
        density_matrix -> Lindblad
    """
    method_raw = args.method or args.solver or "tjm"
    method = method_raw.strip().lower()

    if method not in METHOD_TO_REPRESENTATION:
        valid = ", ".join(sorted(METHOD_TO_REPRESENTATION))
        raise ValueError(f"Unsupported simulation method: {method_raw}. Valid methods: {valid}")

    return method, METHOD_TO_REPRESENTATION[method]


def build_observables(length: int, channels: list[str]) -> list[Observable]:
    observables: list[Observable] = []

    # Channel-major order:
    # x0, x1, ..., xL-1, y0, ..., zL-1
    for channel in channels:
        for site in range(length):
            observables.append(Observable(channel, sites=site))

    return observables


def build_noise_model(
    *,
    length: int,
    gamma_x: float,
    gamma_y: float,
    gamma_z: float,
) -> NoiseModel:
    processes: list[dict[str, Any]] = []

    for site in range(length):
        processes.append({"name": "pauli_x", "sites": [site], "strength": float(gamma_x)})
        processes.append({"name": "pauli_y", "sites": [site], "strength": float(gamma_y)})
        processes.append({"name": "pauli_z", "sites": [site], "strength": float(gamma_z)})

    return NoiseModel(processes)


def normalize_initial_state_name(name: str) -> str:
    if name.lower() == "neel":
        return "Neel"
    return name


def build_state(
    *,
    length: int,
    initial_state: str,
    representation: str,
) -> State:
    return State(
        length,
        initial=normalize_initial_state_name(initial_state),
        representation=representation,
    )


def result_to_sample(
    expectation_values: object,
    *,
    channels: list[str],
    length: int,
    expected_time_steps: int,
) -> NDArray[np.float32]:
    """
    Convert YAQS result.expectation_values to one ML sample.

    Output shape:
        (C, L, T)

    Expected order:
        x0, x1, ..., xL-1, y0, ..., zL-1
    """
    expected_observables = len(channels) * length

    rows = [
        np.asarray(row, dtype=np.float32).reshape(-1)
        for row in expectation_values  # type: ignore[union-attr]
    ]

    arr = np.vstack(rows)

    if arr.shape != (expected_observables, expected_time_steps):
        squeezed = np.squeeze(np.asarray(expectation_values, dtype=np.float32))

        if squeezed.shape == (expected_observables, expected_time_steps):
            arr = squeezed
        else:
            raise RuntimeError(
                "Unexpected result.expectation_values shape. "
                f"Got {squeezed.shape}, expected ({expected_observables}, {expected_time_steps})."
            )

    return arr.reshape(len(channels), length, expected_time_steps).astype(np.float32)


def generate_one_sample(
    *,
    length: int,
    elapsed_time: float,
    dt: float,
    channels: list[str],
    gamma_xyz: NDArray[np.float32],
    method: str,
    representation: str,
    preset: str,
    j_coupling: float,
    transverse_field: float,
    initial_state: str,
    svd_threshold: float | None,
    max_bond_dim: int | None,
    num_traj: int,
    order: int,
    tdvp_sweeps: int,
    tdvp_mode: str,
    random_seed: int | None,
    parallel: bool,
    max_workers: int | None,
    show_progress: bool,
) -> NDArray[np.float32]:
    gamma_x, gamma_y, gamma_z = gamma_xyz.tolist()

    hamiltonian = Hamiltonian.ising(length, j_coupling, transverse_field)

    state = build_state(
        length=length,
        initial_state=initial_state,
        representation=representation,
    )

    noise_model = build_noise_model(
        length=length,
        gamma_x=gamma_x,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
    )

    observables = build_observables(length=length, channels=channels)

    # Lindblad/density_matrix is deterministic, so one trajectory is enough.
    effective_num_traj = 1 if representation == "density_matrix" else num_traj

    sim_kwargs: dict[str, Any] = {
        "observables": observables,
        "elapsed_time": elapsed_time,
        "dt": dt,
        "num_traj": effective_num_traj,
        "preset": preset,
        "order": order,
        "sample_timesteps": True,
        "random_seed": random_seed,
        "tdvp_sweeps": tdvp_sweeps,
        "tdvp_mode": tdvp_mode,
    }

    if svd_threshold is not None:
        sim_kwargs["svd_threshold"] = svd_threshold

    if max_bond_dim is not None:
        sim_kwargs["max_bond_dim"] = max_bond_dim

    sim_params = AnalogSimParams(**sim_kwargs)

    sim = Simulator(
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )

    result = sim.run(state, hamiltonian, sim_params, noise_model)

    expected_time_steps = int(round(elapsed_time / dt)) + 1

    return result_to_sample(
        result.expectation_values,
        channels=channels,
        length=length,
        expected_time_steps=expected_time_steps,
    )


def build_dataset_config(
    *,
    args: argparse.Namespace,
    method: str,
    representation: str,
    channels: list[str],
    time_steps: int,
    effective_num_traj: int,
    output_path: Path,
    created_at: str,
) -> dict[str, Any]:
    dataset_id = safe_name(args.dataset_id)

    return {
        "dataset_id": dataset_id,
        "dataset_name": args.dataset_name,
        "description": args.description,
        "created_at": created_at,
        "output": str(output_path),

        "samples": int(args.samples),
        "length": int(args.length),
        "channels": channels,
        "observable_channels": channels,
        "observable_order": "channel_major",
        "input_shape_meaning": "(N, observable_channel, site, time)",

        "elapsed_time": float(args.elapsed_time),
        "dt": float(args.dt),
        "time_steps": int(time_steps),

        "target_names": PARAM_NAMES,
        "target_shape_meaning": "(N, gamma_x/gamma_y/gamma_z)",
        "target_transform": "log10",
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "gamma_sampling": "log_uniform",

        "method": method,
        "yaqs_representation": representation,
        "preset": args.preset,
        "num_traj_requested": int(args.num_traj),
        "num_traj_effective": int(effective_num_traj),

        "initial_state": args.initial_state,
        "j_coupling": float(args.j_coupling),
        "transverse_field": float(args.transverse_field),

        "svd_threshold": args.svd_threshold,
        "max_bond_dim": args.max_bond_dim,
        "order": int(args.order),
        "tdvp_sweeps": int(args.tdvp_sweeps),
        "tdvp_mode": args.tdvp_mode,

        "parallel": bool(args.parallel),
        "max_workers": args.max_workers,
        "seed": args.seed,

        "noise_processes": NOISE_PROCESSES,
        "system_name": "ising_chain_with_global_pauli_lindblad_noise",
    }


def build_dataset_summary(
    *,
    config: dict[str, Any],
    X_save: NDArray[np.float32],
    y_save: NDArray[np.float32],
    gamma_save: NDArray[np.float32],
    times: NDArray[np.float64],
    status: str,
    completed_samples: int,
    total_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    summary = {
        **config,
        "status": status,
        "completed_samples": int(completed_samples),

        "X_shape": list(X_save.shape),
        "y_log_shape": list(y_save.shape),
        "gamma_shape": list(gamma_save.shape),
        "times_shape": list(times.shape),

        "X_min": float(np.min(X_save)) if X_save.size else None,
        "X_max": float(np.max(X_save)) if X_save.size else None,
        "X_mean": float(np.mean(X_save)) if X_save.size else None,
        "X_std": float(np.std(X_save)) if X_save.size else None,

        "y_log_min": float(np.min(y_save)) if y_save.size else None,
        "y_log_max": float(np.max(y_save)) if y_save.size else None,

        "gamma_min_actual": float(np.min(gamma_save)) if gamma_save.size else None,
        "gamma_max_actual": float(np.max(gamma_save)) if gamma_save.size else None,

        "time_min": float(np.min(times)) if times.size else None,
        "time_max": float(np.max(times)) if times.size else None,
        "time_step_mean": float(np.mean(np.diff(times))) if len(times) > 1 else None,
    }

    for idx, name in enumerate(PARAM_NAMES):
        if gamma_save.size:
            summary[f"{name}_min"] = float(np.min(gamma_save[:, idx]))
            summary[f"{name}_max"] = float(np.max(gamma_save[:, idx]))
            summary[f"{name}_log10_min"] = float(np.min(y_save[:, idx]))
            summary[f"{name}_log10_max"] = float(np.max(y_save[:, idx]))

    if total_runtime_seconds is not None:
        summary["total_runtime_seconds"] = float(total_runtime_seconds)
        if completed_samples > 0:
            summary["avg_seconds_per_sample"] = float(total_runtime_seconds / completed_samples)

    return summary


def save_dataset(
    *,
    output_path: Path,
    X_data: NDArray[np.float32],
    y_log: NDArray[np.float32],
    gammas: NDArray[np.float32],
    times: NDArray[np.float64],
    channels: list[str],
    config: dict[str, Any],
    upto: int | None = None,
    status: str = "complete",
    total_runtime_seconds: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if upto is None:
        X_save = X_data
        y_save = y_log
        gamma_save = gammas
        completed_samples = X_data.shape[0]
    else:
        X_save = X_data[:upto]
        y_save = y_log[:upto]
        gamma_save = gammas[:upto]
        completed_samples = upto

    summary = build_dataset_summary(
        config=config,
        X_save=X_save,
        y_save=y_save,
        gamma_save=gamma_save,
        times=times,
        status=status,
        completed_samples=completed_samples,
        total_runtime_seconds=total_runtime_seconds,
    )

    np.savez_compressed(
        output_path,
        X=X_save.astype(np.float32),
        y_log=y_save.astype(np.float32),
        theta=y_save.astype(np.float32),
        gamma=gamma_save.astype(np.float32),
        times=times.astype(np.float64),

        dataset_id=np.array(config["dataset_id"]),
        dataset_name=np.array(config["dataset_name"]),
        created_at=np.array(config["created_at"]),
        status=np.array(status),

        param_names=np.array(PARAM_NAMES),
        observable_channels=np.array(channels),
        target_transform=np.array("log10"),
        config_json=np.array(json.dumps(config)),
        summary_json=np.array(json.dumps(summary)),
    )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def create_dataset(args: argparse.Namespace) -> None:
    method, representation = resolve_method(args)

    rng = np.random.default_rng(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channels = parse_channels(args.channels)

    gammas = log_uniform(
        rng=rng,
        low=args.gamma_min,
        high=args.gamma_max,
        size=(args.samples, 3),
    ).astype(np.float32)

    y_log = np.log10(gammas).astype(np.float32)

    time_steps = int(round(args.elapsed_time / args.dt)) + 1
    times = np.linspace(0.0, args.elapsed_time, time_steps, dtype=np.float64)

    X_data = np.zeros(
        (args.samples, len(channels), args.length, time_steps),
        dtype=np.float32,
    )

    effective_num_traj = 1 if representation == "density_matrix" else args.num_traj
    created_at = now_iso()

    config = build_dataset_config(
        args=args,
        method=method,
        representation=representation,
        channels=channels,
        time_steps=time_steps,
        effective_num_traj=effective_num_traj,
        output_path=output_path,
        created_at=created_at,
    )

    print("=" * 88)
    print("QEL Twin dataset generation")
    print("=" * 88)
    print(f"dataset_id:          {config['dataset_id']}")
    print(f"dataset_name:        {config['dataset_name']}")
    print(f"output:              {output_path}")
    print(f"samples:             {args.samples}")
    print(f"length:              {args.length}")
    print(f"channels:            {channels}")
    print(f"time steps:          {time_steps}")
    print(f"elapsed_time:        {args.elapsed_time}")
    print(f"dt:                  {args.dt}")
    print(f"method:              {method}")
    print(f"YAQS representation: {representation}")
    print(f"preset:              {args.preset}")
    print(f"num_traj requested:  {args.num_traj}")
    print(f"num_traj effective:  {effective_num_traj}")
    print(f"parallel:            {args.parallel}")
    print(f"max_workers:         {args.max_workers}")
    print(f"gamma range:         [{args.gamma_min}, {args.gamma_max}]")
    print("=" * 88)

    if method == "analog":
        print("note: method='analog' is an alias for TJM/MPS analog simulation.")

    if representation == "density_matrix" and args.num_traj != 1:
        print("note: density_matrix/Lindblad is deterministic, so effective num_traj is set to 1.")

    print()

    start_time = time.time()

    for sample_idx in range(args.samples):
        completed = sample_idx

        if completed > 0:
            elapsed = time.time() - start_time
            avg_per_sample = elapsed / completed
            remaining_samples = args.samples - completed
            remaining_seconds = avg_per_sample * remaining_samples
            progress_pct = 100.0 * completed / args.samples

            print(
                f"\n[{sample_idx + 1}/{args.samples}] "
                f"{progress_pct:.1f}% complete | "
                f"elapsed={format_seconds(elapsed)} | "
                f"avg/sample={avg_per_sample:.2f}s | "
                f"remaining={format_seconds(remaining_seconds)}"
            )
        else:
            print(f"\n[{sample_idx + 1}/{args.samples}] starting dataset generation...")

        print(
            f"gamma_x={gammas[sample_idx, 0]:.3e}, "
            f"gamma_y={gammas[sample_idx, 1]:.3e}, "
            f"gamma_z={gammas[sample_idx, 2]:.3e}"
        )

        sample_seed = None if args.seed is None else int(args.seed + sample_idx)

        X_data[sample_idx] = generate_one_sample(
            length=args.length,
            elapsed_time=args.elapsed_time,
            dt=args.dt,
            channels=channels,
            gamma_xyz=gammas[sample_idx],
            method=method,
            representation=representation,
            preset=args.preset,
            j_coupling=args.j_coupling,
            transverse_field=args.transverse_field,
            initial_state=args.initial_state,
            svd_threshold=args.svd_threshold,
            max_bond_dim=args.max_bond_dim,
            num_traj=args.num_traj,
            order=args.order,
            tdvp_sweeps=args.tdvp_sweeps,
            tdvp_mode=args.tdvp_mode,
            random_seed=sample_seed,
            parallel=args.parallel,
            max_workers=args.max_workers,
            show_progress=args.show_progress,
        )

        if args.save_every > 0 and (sample_idx + 1) % args.save_every == 0:
            save_dataset(
                output_path=output_path,
                X_data=X_data,
                y_log=y_log,
                gammas=gammas,
                times=times,
                channels=channels,
                config=config,
                upto=sample_idx + 1,
                status="partial",
                total_runtime_seconds=time.time() - start_time,
            )
            print(f"Partial save: {output_path}")
            print(f"Partial summary: {output_path.with_suffix('.summary.json')}")

    total_elapsed = time.time() - start_time

    save_dataset(
        output_path=output_path,
        X_data=X_data,
        y_log=y_log,
        gammas=gammas,
        times=times,
        channels=channels,
        config=config,
        upto=None,
        status="complete",
        total_runtime_seconds=total_elapsed,
    )

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Saved dataset:  {output_path}")
    print(f"Saved config:   {metadata_path}")
    print(f"Saved summary:  {output_path.with_suffix('.summary.json')}")
    print(f"X shape:        {X_data.shape}")
    print(f"y_log shape:    {y_log.shape}")
    print(f"gamma shape:    {gammas.shape}")
    print(f"times shape:    {times.shape}")
    print(f"Total runtime:  {format_seconds(total_elapsed)}")
    print(f"Avg/sample:     {total_elapsed / args.samples:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create QEL/P3 Lindblad-style dataset with YAQS main API."
    )

    parser.add_argument("--output", type=str, default="data/processed/dataset_p3_lindblad.npz")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--dataset-id", type=str, default="dataset_p3_lindblad")
    parser.add_argument("--dataset-name", type=str, default="P3 Lindblad Global Pauli Noise")
    parser.add_argument(
        "--description",
        type=str,
        default=(
            "Observable time-series dataset for predicting global Lindblad "
            "noise parameters gamma_x, gamma_y, gamma_z."
        ),
    )

    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--channels", type=str, default="x,y,z")

    parser.add_argument("--elapsed-time", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument("--gamma-min", type=float, default=1e-4)
    parser.add_argument("--gamma-max", type=float, default=1e-1)

    parser.add_argument(
        "--method",
        "--simulation-method",
        dest="method",
        type=str,
        default=None,
        help=(
            "Simulation backend/method. Valid: "
            "analog, tjm, mps, mcwf, vector, lindblad, density_matrix, exact. "
            "analog/tjm/mps -> YAQS representation mps. "
            "mcwf/vector -> vector. "
            "lindblad/density_matrix/exact -> density_matrix."
        ),
    )

    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help="Legacy alias for --method. Example: --solver TJM maps to --method tjm.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["fast", "balanced", "accurate", "exact"],
    )

    parser.add_argument("--num-traj", type=int, default=100)
    parser.add_argument("--svd-threshold", type=float, default=None)
    parser.add_argument("--max-bond-dim", type=int, default=None)

    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--tdvp-sweeps", type=int, default=1)
    parser.add_argument(
        "--tdvp-mode",
        type=str,
        default="2site",
        choices=["1site", "2site", "dynamic"],
    )

    parser.add_argument("--j-coupling", type=float, default=1.0)
    parser.add_argument("--transverse-field", type=float, default=1.0)
    parser.add_argument("--initial-state", type=str, default="zeros")

    parser.add_argument("--save-every", type=int, default=25)

    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YAQS trajectory parallelism. Use --no-parallel to disable.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum YAQS worker processes. Default: YAQS decides.",
    )

    parser.add_argument("--show-progress", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_causal_polarization import (
    transition_bandpass,
    transition_bandpass_y_derivative,
    weak_transport_gradient_contraction,
    weak_transport_tensor,
)
from voidscreen.sigma_nonmetricity import weyl_trace_nonmetricity


def relative_error(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-15)


def polarization_density(
    grad_psi: np.ndarray,
    grad_phi: np.ndarray,
    sigma: np.ndarray,
    grad_sigma: np.ndarray,
    *,
    acceleration_scale: float,
    memory_length: float,
    anisotropy: float,
    polarization_scale: float,
) -> float:
    grad_weyl = 0.5 * (grad_psi + grad_phi)
    u = grad_weyl / acceleration_scale
    tensor = weak_transport_tensor(u, anisotropy)
    kinetic = np.einsum("...i,...ij,...j->...", grad_sigma, tensor, grad_sigma)
    y = np.sum(np.square(grad_phi), axis=-1) / acceleration_scale**2
    source = np.square(y) / np.square(1.0 + np.square(y))
    density = polarization_scale * (
        memory_length**2 * kinetic + np.square(sigma) - 2.0 * sigma * source
    )
    return float(np.sum(density))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Sigma v5A weak variation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v5a_weak_field_derivation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v5a_weak_field_derivation",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manufactured = config["manufactured"]
    rng = np.random.default_rng(int(manufactured["random_seed"]))
    samples = int(manufactured["samples"])
    acceleration_scale = float(manufactured["acceleration_scale"])
    memory_length = float(manufactured["memory_length"])
    anisotropy = float(manufactured["anisotropy"])
    polarization_scale = float(manufactured["polarization_scale"])
    step = float(manufactured["finite_difference_step"])

    grad_psi = rng.normal(size=(samples, 3))
    grad_phi = rng.normal(size=(samples, 3))
    sigma = rng.normal(size=samples)
    grad_sigma = rng.normal(size=(samples, 3))
    direction_psi = rng.normal(size=(samples, 3))
    direction_phi = rng.normal(size=(samples, 3))
    direction_sigma = rng.normal(size=samples)
    direction_grad_sigma = rng.normal(size=(samples, 3))

    u = 0.5 * (grad_psi + grad_phi) / acceleration_scale
    chain = weak_transport_gradient_contraction(u, grad_sigma, anisotropy)
    b_flux = memory_length**2 * chain / (2.0 * acceleration_scale)
    y = np.sum(np.square(grad_phi), axis=-1) / acceleration_scale**2
    source = transition_bandpass(np.sqrt(y))
    source_y = transition_bandpass_y_derivative(y)
    phi_source_flux = -4.0 * sigma[:, None] * source_y[:, None] * grad_phi
    phi_source_flux /= acceleration_scale**2
    tensor = weak_transport_tensor(u, anisotropy)
    sigma_gradient_flux = 2.0 * memory_length**2 * np.einsum(
        "...ij,...j->...i", tensor, grad_sigma
    )
    sigma_local_derivative = 2.0 * (sigma - source)

    analytic = polarization_scale * float(
        np.sum(b_flux * direction_psi)
        + np.sum((b_flux + phi_source_flux) * direction_phi)
        + np.sum(sigma_gradient_flux * direction_grad_sigma)
        + np.sum(sigma_local_derivative * direction_sigma)
    )
    plus = polarization_density(
        grad_psi + step * direction_psi,
        grad_phi + step * direction_phi,
        sigma + step * direction_sigma,
        grad_sigma + step * direction_grad_sigma,
        acceleration_scale=acceleration_scale,
        memory_length=memory_length,
        anisotropy=anisotropy,
        polarization_scale=polarization_scale,
    )
    minus = polarization_density(
        grad_psi - step * direction_psi,
        grad_phi - step * direction_phi,
        sigma - step * direction_sigma,
        grad_sigma - step * direction_grad_sigma,
        acceleration_scale=acceleration_scale,
        memory_length=memory_length,
        anisotropy=anisotropy,
        polarization_scale=polarization_scale,
    )
    finite = (plus - minus) / (2.0 * step)
    full_error = relative_error(finite, analytic)

    transport_direction = rng.normal(size=(samples, 3))
    transport_direction /= np.sqrt(np.mean(np.square(transport_direction)))

    def transport_energy(argument: np.ndarray) -> float:
        local_tensor = weak_transport_tensor(argument, anisotropy)
        return float(
            np.sum(
                np.einsum(
                    "...i,...ij,...j->...", grad_sigma, local_tensor, grad_sigma
                )
            )
        )

    finite_transport = (
        transport_energy(u + step * transport_direction)
        - transport_energy(u - step * transport_direction)
    ) / (2.0 * step)
    analytic_transport = float(np.sum(chain * transport_direction))
    transport_error = relative_error(finite_transport, analytic_transport)

    source_direction = rng.normal(size=samples)
    finite_source = float(
        np.sum(
            (
                np.square(y + step * source_direction)
                / np.square(1.0 + np.square(y + step * source_direction))
                - np.square(y - step * source_direction)
                / np.square(1.0 + np.square(y - step * source_direction))
            )
            / (2.0 * step)
        )
    )
    analytic_source = float(np.sum(source_y * source_direction))
    source_error = relative_error(finite_source, analytic_source)

    expected_weyl = 4.0 * np.sum(np.square(grad_psi + grad_phi), axis=-1)
    weyl_error = float(
        np.max(np.abs(weyl_trace_nonmetricity(grad_psi, grad_phi) - expected_weyl))
    )

    gate_config = config["gates"]
    gates = {
        "transport_chain": transport_error
        <= float(gate_config["maximum_transport_chain_relative_error"]),
        "source_derivative": source_error
        <= float(gate_config["maximum_source_derivative_relative_error"]),
        "complete_local_polarization_derivative": full_error
        <= float(
            gate_config["maximum_complete_local_polarization_derivative_relative_error"]
        ),
        "weyl_trace_identity": weyl_error
        <= float(gate_config["maximum_weyl_trace_identity_error"]),
    }
    report = {
        "status": "completed Sigma v5A weak-field variational audit",
        "observational_data_accessed": False,
        "checks": {
            "transport_chain_relative_error": transport_error,
            "source_derivative_relative_error": source_error,
            "complete_local_polarization_derivative_relative_error": full_error,
            "weyl_trace_identity_maximum_absolute_error": weyl_error,
        },
        "gates": {name: bool(value) for name, value in gates.items()},
        "all_gates_pass": bool(all(gates.values())),
        "decision": "continue_full_background_and_constraint_derivation_no_data_fit",
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

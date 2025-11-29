# Physics Discovery Engine - General Relativity Derivation

A GPU-accelerated genetic algorithm that rediscovers Einstein's Field Equations from synthetic spacetime data.

## Prerequisites

```bash
pip install cupy-cuda12x numpy sympy
```

> **Note**: Adjust `cupy-cuda12x` to match your CUDA driver version (e.g., `cupy-cuda11x` for CUDA 11).

## How It Works

### The Strategy

1. **"Universe" Generator**: Creates synthetic spacetime data where GR is true by construction
2. **Blind AI**: Receives raw tensors (g_uv, R_uv, R, T_uv) without knowing relationships
3. **GPU Engine**: Uses genetic algorithm to test billions of tensor combinations

### The Physics

The script generates random metrics and curvature tensors, then computes what the stress-energy tensor T_uv **must be** for Einstein's equation to hold:

```
G_uv = R_uv - 0.5 R g_uv = 8π T_uv
```

The AI then tries to find coefficients [c1, c2, c3, c4] such that:

```
c1*R_uv + c2*R*g_uv + c3*T_uv + c4*g_uv = 0
```

If successful, it should discover c1=1, c2=-0.5, c3=-8π, c4=0.

## Usage

```bash
python gr_derivation.py
```

## Expected Output

```
--- Phase 1: Observing the Universe ---
Generating 100000 spacetime points (The Universe)...
Moving Universe to VRAM...

--- Phase 2: Deriving Laws of Physics on GPU ---
Constructing Term Vectors on GPU...
Gen 0: Best Loss 1.234567e+00
...
Gen 40: Best Loss 1.234567e-12

--- FINAL DISCOVERY ---
Normalized Eq: R_uv + (-0.5000) R g_uv + (-25.1327) T_uv + (0.0000) g_uv = 0

--- COMPARISON TO EINSTEIN ---
Einstein Field Equation: R_uv - 0.5 R g_uv - 8*pi T_uv = 0
Target 'Scalar' Coeff: -0.5
Target 'Matter' Coeff: -25.1327

SUCCESS: The AI has derived General Relativity!
```

## GPU Memory Requirements

- 100k samples × (4,4) tensors × 4 terms × 4 bytes ≈ ~100 MB for data
- Batched evaluation keeps VRAM usage manageable
- RTX 5090 (32GB) can handle larger populations for faster convergence

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 100,000 | Number of spacetime points |
| `population_size` | 5,000 | Candidate equations per generation |
| `generations` | 50 | Evolution iterations |
| `mutation_rate` | 0.1 (decaying) | Noise for exploration |

## Extending This

To search for more complex physics (e.g., with cosmological constant Λ), the genetic search already includes the c4*g_uv term. You could also:

- Add higher-order curvature terms (Riemann tensor, Weyl tensor)
- Include derivative terms for modified gravity theories
- Expand to more sophisticated optimization (CMA-ES, differential evolution)

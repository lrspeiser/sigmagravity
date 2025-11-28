#!/usr/bin/env python3
"""
Σ-GRAVITY COMPREHENSIVE FORMULA DISCOVERY ENGINE
=================================================

Production-ready symbolic regression for discovering gravitational
enhancement formulas from astronomical data.

Optimized for:
- NVIDIA RTX 5090 (32GB VRAM) via CuPy
- 10-core CPU via multiprocessing
- Exhaustive formula space exploration

Target discoveries:
1. Enhancement factor K(R) = g_obs/g_bar - 1
2. Radial Acceleration Relation g_obs = f(g_bar, R)
3. Baryonic Tully-Fisher v = f(M)
4. Coherence window W(R) parameters

Usage:
    python run_sigma_discovery.py                       # Run all targets
    python run_sigma_discovery.py --target K            # Enhancement only
    python run_sigma_discovery.py --target K --exhaustive  # Full search
    python run_sigma_discovery.py --real                # Use real data

Author: Theory Discovery Engine for Σ-Gravity Research
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from copy import deepcopy
from collections import defaultdict
from enum import IntEnum
import random
import time
import json
import pickle
import argparse
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import itertools

warnings.filterwarnings('ignore')

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

try:
    import cupy as cp
    from cupy import cuda
    
    GPU_AVAILABLE = True
    
    # Configure memory pool
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    # Get device info
    device = cuda.Device(0)
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    gpu_mem = device.mem_info[1] / 1e9
    
    print(f"✓ GPU: {gpu_name}")
    print(f"  Memory: {gpu_mem:.1f} GB")
    
    xp = cp  # Array backend
    
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    cp = np
    print("⚠ GPU not available, using CPU")

NUM_CORES = min(10, mp.cpu_count())
print(f"✓ CPU Cores: {NUM_CORES}")


# =============================================================================
# OPERATION DEFINITIONS - COMPREHENSIVE SET
# =============================================================================

class Op(IntEnum):
    """All supported operations for formula discovery"""
    # Terminals
    CONST = 0
    VAR = 1
    
    # Binary arithmetic
    ADD = 10
    SUB = 11
    MUL = 12
    DIV = 13
    POW = 14
    
    # Unary basic
    NEG = 20
    INV = 21
    SQUARE = 22
    CUBE = 23
    SQRT = 24
    CBRT = 25
    
    # Unary transcendental
    LOG = 30      # log10
    LN = 31       # natural log
    EXP = 32
    
    # Trigonometric (rarely needed but included)
    SIN = 40
    COS = 41
    TANH = 42     # Useful for smooth transitions
    
    # Special functions for physics
    SIGMOID = 50  # 1/(1+exp(-x)) - smooth step
    SOFTPLUS = 51 # log(1+exp(x)) - smooth ReLU
    ERF = 52      # Error function


# Operation metadata
OP_ARITY = {
    Op.CONST: 0, Op.VAR: 0,
    Op.ADD: 2, Op.SUB: 2, Op.MUL: 2, Op.DIV: 2, Op.POW: 2,
    Op.NEG: 1, Op.INV: 1, Op.SQUARE: 1, Op.CUBE: 1, Op.SQRT: 1, Op.CBRT: 1,
    Op.LOG: 1, Op.LN: 1, Op.EXP: 1,
    Op.SIN: 1, Op.COS: 1, Op.TANH: 1,
    Op.SIGMOID: 1, Op.SOFTPLUS: 1, Op.ERF: 1,
}

OP_NAMES = {
    Op.CONST: 'c', Op.VAR: 'x',
    Op.ADD: '+', Op.SUB: '-', Op.MUL: '×', Op.DIV: '/', Op.POW: '^',
    Op.NEG: '-', Op.INV: '1/', Op.SQUARE: '²', Op.CUBE: '³', 
    Op.SQRT: '√', Op.CBRT: '∛',
    Op.LOG: 'log', Op.LN: 'ln', Op.EXP: 'exp',
    Op.SIN: 'sin', Op.COS: 'cos', Op.TANH: 'tanh',
    Op.SIGMOID: 'σ', Op.SOFTPLUS: 'sp', Op.ERF: 'erf',
}


# =============================================================================
# PHYSICS-INFORMED CONSTANTS
# =============================================================================

# Constants that appear in gravitational physics
PHYSICS_CONSTANTS = [
    # Integers and simple fractions
    0, 1, 2, 3, 4, 5, 6,
    0.5, 0.25, 0.1, 0.01,
    -1, -0.5,
    
    # Pi multiples (appear in GR)
    np.pi, 2*np.pi, 4*np.pi, 6*np.pi,
    np.pi/2, np.pi/4,
    
    # Common roots
    np.sqrt(2), np.sqrt(3), np.sqrt(np.pi),
    
    # Euler's number
    np.e, 1/np.e,
    
    # MOND acceleration scale (in appropriate units)
    1.2,  # a0 ≈ 1.2 × 10^-10 m/s²
    
    # Common scale radii (kpc)
    3.0, 5.0, 8.0, 10.0, 15.0, 20.0,
    
    # Burr-XII parameters (typical ranges)
    1.5, 2.5, 3.5,
]


# =============================================================================
# EXPRESSION REPRESENTATION
# =============================================================================

@dataclass
class Expression:
    """
    Stack-based expression for GPU-efficient evaluation.
    
    Each instruction is (operation_code, operand_value)
    - CONST: operand is the constant value
    - VAR: operand is the variable index
    - Operations: operand unused (0.0)
    """
    instructions: List[Tuple[int, float]]
    
    def __len__(self):
        return len(self.instructions)
    
    def __hash__(self):
        return hash(tuple(self.instructions))
    
    def __eq__(self, other):
        return self.instructions == other.instructions
    
    def copy(self) -> 'Expression':
        return Expression(list(self.instructions))
    
    def complexity(self) -> int:
        """Weighted complexity score"""
        score = 0
        for op, _ in self.instructions:
            if op in [Op.CONST, Op.VAR]:
                score += 1
            elif op in [Op.ADD, Op.SUB, Op.MUL, Op.NEG]:
                score += 1
            elif op in [Op.DIV, Op.INV, Op.SQRT, Op.SQUARE]:
                score += 2
            elif op in [Op.POW, Op.LOG, Op.LN, Op.EXP]:
                score += 3
            elif op in [Op.TANH, Op.SIGMOID, Op.ERF]:
                score += 4
            else:
                score += 2
        return score
    
    def get_variables(self) -> Set[int]:
        """Get indices of variables used"""
        return {int(val) for op, val in self.instructions if op == Op.VAR}
    
    def to_string(self, var_names: List[str]) -> str:
        """Convert to human-readable infix notation"""
        stack = []
        
        try:
            for op, val in self.instructions:
                if op == Op.CONST:
                    # Format constants nicely
                    if val == 0:
                        stack.append("0")
                    elif abs(val - round(val)) < 0.001 and abs(val) < 100:
                        stack.append(str(int(round(val))))
                    elif abs(val - np.pi) < 0.01:
                        stack.append("π")
                    elif abs(val - 2*np.pi) < 0.01:
                        stack.append("2π")
                    elif abs(val - 4*np.pi) < 0.01:
                        stack.append("4π")
                    elif abs(val - 6*np.pi) < 0.01:
                        stack.append("6π")
                    elif abs(val - np.sqrt(2)) < 0.01:
                        stack.append("√2")
                    elif abs(val - np.e) < 0.01:
                        stack.append("e")
                    else:
                        stack.append(f"{val:.4g}")
                
                elif op == Op.VAR:
                    idx = int(val)
                    stack.append(var_names[idx] if idx < len(var_names) else f"x{idx}")
                
                elif op == Op.ADD:
                    if len(stack) < 2:
                        return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} + {b})")
                elif op == Op.SUB:
                    if len(stack) < 2:
                        return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} - {b})")
                elif op == Op.MUL:
                    if len(stack) < 2:
                        return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    # Clean up display
                    if len(a) < 6 and not any(c in a for c in '+-'):
                        stack.append(f"{a}·{b}")
                    else:
                        stack.append(f"({a} × {b})")
                elif op == Op.DIV:
                    if len(stack) < 2:
                        return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a}/{b})")
                elif op == Op.POW:
                    if len(stack) < 2:
                        return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a}^{b})")
                
                elif op == Op.NEG:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"(-{stack.pop()})")
                elif op == Op.INV:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"(1/{stack.pop()})")
                elif op == Op.SQUARE:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"{stack.pop()}²")
                elif op == Op.CUBE:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"{stack.pop()}³")
                elif op == Op.SQRT:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"√{stack.pop()}")
                elif op == Op.CBRT:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"∛{stack.pop()}")
                
                elif op == Op.LOG:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"log({stack.pop()})")
                elif op == Op.LN:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"ln({stack.pop()})")
                elif op == Op.EXP:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"exp({stack.pop()})")
                
                elif op == Op.SIN:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"sin({stack.pop()})")
                elif op == Op.COS:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"cos({stack.pop()})")
                elif op == Op.TANH:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"tanh({stack.pop()})")
                
                elif op == Op.SIGMOID:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"σ({stack.pop()})")
                elif op == Op.SOFTPLUS:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"softplus({stack.pop()})")
                elif op == Op.ERF:
                    if len(stack) < 1:
                        return "[invalid]"
                    stack.append(f"erf({stack.pop()})")
            
            return stack[0] if stack else "?"
        except Exception:
            return "[invalid]"
    
    def to_latex(self, var_names: List[str]) -> str:
        """Convert to LaTeX notation"""
        s = self.to_string(var_names)
        # Basic conversions
        s = s.replace('×', r'\times')
        s = s.replace('·', r'\cdot')
        s = s.replace('√', r'\sqrt{')
        s = s.replace('²', '^2')
        s = s.replace('³', '^3')
        s = s.replace('π', r'\pi')
        return s


# =============================================================================
# GPU-ACCELERATED EVALUATION
# =============================================================================

def safe_eval_gpu(expr: Expression, X: Dict[str, Any], use_gpu: bool = True) -> Any:
    """
    Evaluate expression with numerical safety.
    Works on both GPU (CuPy) and CPU (NumPy).
    """
    if use_gpu and GPU_AVAILABLE:
        backend = cp
        X_arr = {k: cp.asarray(v, dtype=cp.float64) for k, v in X.items()}
    else:
        backend = np
        X_arr = {k: np.asarray(v, dtype=np.float64) for k, v in X.items()}
    
    var_names = list(X_arr.keys())
    var_arrays = [X_arr[v] for v in var_names]
    n = len(var_arrays[0])
    
    stack = []
    EPS = 1e-30
    CLIP = 1e30
    
    for op, val in expr.instructions:
        # Terminals
        if op == Op.CONST:
            stack.append(backend.full(n, val, dtype=backend.float64))
        elif op == Op.VAR:
            stack.append(var_arrays[int(val)].copy())
        
        # Binary arithmetic
        elif op == Op.ADD:
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == Op.SUB:
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == Op.MUL:
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == Op.DIV:
            b, a = stack.pop(), stack.pop()
            stack.append(backend.where(backend.abs(b) > EPS, a / b, backend.sign(a) * CLIP))
        elif op == Op.POW:
            b, a = stack.pop(), stack.pop()
            b_safe = backend.clip(b, -10, 10)
            a_safe = backend.where(a > 0, a, backend.abs(a) + EPS)
            stack.append(backend.power(a_safe, b_safe))
        
        # Unary basic
        elif op == Op.NEG:
            stack.append(-stack.pop())
        elif op == Op.INV:
            a = stack.pop()
            stack.append(backend.where(backend.abs(a) > EPS, 1.0 / a, CLIP))
        elif op == Op.SQUARE:
            a = stack.pop()
            stack.append(a * a)
        elif op == Op.CUBE:
            a = stack.pop()
            stack.append(a * a * a)
        elif op == Op.SQRT:
            a = stack.pop()
            stack.append(backend.sqrt(backend.abs(a) + EPS))
        elif op == Op.CBRT:
            a = stack.pop()
            stack.append(backend.cbrt(a))
        
        # Transcendental
        elif op == Op.LOG:
            a = stack.pop()
            stack.append(backend.where(a > EPS, backend.log10(a), -30))
        elif op == Op.LN:
            a = stack.pop()
            stack.append(backend.where(a > EPS, backend.log(a), -30))
        elif op == Op.EXP:
            a = stack.pop()
            stack.append(backend.exp(backend.clip(a, -30, 30)))
        
        # Trigonometric
        elif op == Op.SIN:
            stack.append(backend.sin(stack.pop()))
        elif op == Op.COS:
            stack.append(backend.cos(stack.pop()))
        elif op == Op.TANH:
            stack.append(backend.tanh(stack.pop()))
        
        # Special functions
        elif op == Op.SIGMOID:
            a = stack.pop()
            stack.append(1.0 / (1.0 + backend.exp(-backend.clip(a, -30, 30))))
        elif op == Op.SOFTPLUS:
            a = stack.pop()
            stack.append(backend.log(1.0 + backend.exp(backend.clip(a, -30, 30))))
        elif op == Op.ERF:
            a = stack.pop()
            if backend == np:
                from scipy.special import erf
                stack.append(erf(a))
            else:
                # CuPy doesn't have erf, use approximation
                # erf(x) ≈ tanh(sqrt(π) * x * (1 + 0.044715 * x²))
                stack.append(backend.tanh(1.7724538509 * a * (1 + 0.044715 * a * a)))
    
    result = stack[0] if stack else backend.zeros(n, dtype=backend.float64)
    return backend.clip(result, -CLIP, CLIP)


def batch_evaluate_fitness(
    programs: List[Expression],
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    complexity_weight: float = 0.01,
    use_gpu: bool = True
) -> List[Tuple[float, float, int]]:
    """
    Evaluate fitness of multiple programs.
    Returns: List of (mse, r_squared, complexity)
    """
    results = []
    
    y_mean = np.mean(y)
    y_var = np.var(y)
    ss_tot = np.sum((y - y_mean)**2)
    
    for prog in programs:
        try:
            y_pred = safe_eval_gpu(prog, X, use_gpu=use_gpu)
            
            # Convert to numpy if needed
            if GPU_AVAILABLE and use_gpu:
                y_pred = cp.asnumpy(y_pred)
            
            # Check validity
            valid = np.isfinite(y_pred)
            if valid.sum() < len(y) * 0.5:
                results.append((1e30, -1.0, prog.complexity()))
                continue
            
            y_v = y[valid]
            y_pred_v = y_pred[valid]
            
            # Metrics
            residuals = y_v - y_pred_v
            mse = float(np.mean(residuals**2))
            ss_res = float(np.sum(residuals**2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            results.append((mse, r2, prog.complexity()))
            
        except Exception as e:
            results.append((1e30, -1.0, prog.complexity()))
    
    return results


# =============================================================================
# EXPRESSION GENERATION
# =============================================================================

class ExpressionGenerator:
    """
    Generates expressions with physics-aware biases.
    
    For Σ-Gravity, we bias toward:
    - Ratios (M/r, g_bar/a0)
    - Square roots (v = √(GM/r))
    - Power laws (r^α)
    - Smooth transitions (tanh, sigmoid)
    - Window functions ((r/r0)^α / (1 + (r/r0)^β))
    """
    
    def __init__(
        self,
        num_vars: int,
        var_names: List[str],
        max_depth: int = 6,
        physics_mode: str = 'gravity'  # 'gravity', 'general', 'exhaustive'
    ):
        self.num_vars = num_vars
        self.var_names = var_names
        self.max_depth = max_depth
        self.physics_mode = physics_mode
        
        # Set up operation weights based on mode
        self._setup_operations()
        
        # Constants to use
        self.constants = PHYSICS_CONSTANTS
    
    def _setup_operations(self):
        """Configure operation probabilities based on physics mode"""
        
        if self.physics_mode == 'gravity':
            # Gravitational physics: ratios, roots, power laws
            self.binary_ops = [
                (Op.MUL, 3.0),
                (Op.DIV, 3.0),
                (Op.ADD, 2.0),
                (Op.SUB, 1.0),
                (Op.POW, 1.0),
            ]
            self.unary_ops = [
                (Op.SQRT, 3.0),
                (Op.SQUARE, 2.0),
                (Op.INV, 2.0),
                (Op.LOG, 1.5),
                (Op.CUBE, 0.5),
                (Op.TANH, 1.0),
                (Op.EXP, 0.5),
                (Op.SIGMOID, 0.5),
            ]
        
        elif self.physics_mode == 'exhaustive':
            # Include all operations equally
            self.binary_ops = [
                (Op.MUL, 1.0),
                (Op.DIV, 1.0),
                (Op.ADD, 1.0),
                (Op.SUB, 1.0),
                (Op.POW, 1.0),
            ]
            self.unary_ops = [
                (Op.SQRT, 1.0),
                (Op.SQUARE, 1.0),
                (Op.CUBE, 1.0),
                (Op.CBRT, 1.0),
                (Op.INV, 1.0),
                (Op.NEG, 1.0),
                (Op.LOG, 1.0),
                (Op.LN, 1.0),
                (Op.EXP, 1.0),
                (Op.TANH, 1.0),
                (Op.SIGMOID, 1.0),
                (Op.SOFTPLUS, 1.0),
            ]
        
        else:  # general
            self.binary_ops = [
                (Op.MUL, 2.0),
                (Op.DIV, 2.0),
                (Op.ADD, 2.0),
                (Op.SUB, 1.0),
                (Op.POW, 0.5),
            ]
            self.unary_ops = [
                (Op.SQRT, 2.0),
                (Op.SQUARE, 1.5),
                (Op.INV, 1.5),
                (Op.LOG, 1.0),
                (Op.EXP, 0.5),
            ]
        
        # Normalize weights
        self._normalize_weights()
    
    def _normalize_weights(self):
        total_bin = sum(w for _, w in self.binary_ops)
        self.binary_ops = [(op, w/total_bin) for op, w in self.binary_ops]
        
        total_un = sum(w for _, w in self.unary_ops)
        self.unary_ops = [(op, w/total_un) for op, w in self.unary_ops]
    
    def _weighted_choice(self, choices: List[Tuple[Op, float]]) -> Op:
        r = random.random()
        cumsum = 0
        for op, w in choices:
            cumsum += w
            if r < cumsum:
                return op
        return choices[-1][0]
    
    def random_const(self) -> float:
        """Generate a random constant"""
        if random.random() < 0.7:
            return random.choice(self.constants)
        else:
            return random.uniform(-10, 10)
    
    def random_expr(self, max_depth: int = None) -> Expression:
        """Generate random expression using grow method"""
        if max_depth is None:
            max_depth = self.max_depth
        
        instructions = []
        
        def grow(depth: int):
            # Terminal probability increases with depth
            terminal_prob = 0.25 + 0.75 * (1 - depth / self.max_depth)
            
            if depth <= 1 or random.random() < terminal_prob:
                # Terminal: variable or constant
                if random.random() < 0.65:
                    instructions.append((Op.VAR, float(random.randint(0, self.num_vars - 1))))
                else:
                    instructions.append((Op.CONST, self.random_const()))
            else:
                # Operation
                if random.random() < 0.65:
                    # Binary
                    grow(depth - 1)
                    grow(depth - 1)
                    instructions.append((self._weighted_choice(self.binary_ops), 0.0))
                else:
                    # Unary
                    grow(depth - 1)
                    instructions.append((self._weighted_choice(self.unary_ops), 0.0))
        
        grow(max_depth)
        return Expression(instructions)
    
    def from_template(self, template: str) -> Expression:
        """
        Generate expression from a template string.
        
        Templates:
        - "x/y" → x0/x1
        - "sqrt(x/y)" → √(x0/x1)
        - "x^a * y^b / (1 + (x/c)^d)" → power law with window
        """
        instructions = []
        
        # Simple templates
        templates = {
            "x/y": [(Op.VAR, 0), (Op.VAR, 1), (Op.DIV, 0)],
            "sqrt(x/y)": [(Op.VAR, 0), (Op.VAR, 1), (Op.DIV, 0), (Op.SQRT, 0)],
            "sqrt(x)": [(Op.VAR, 0), (Op.SQRT, 0)],
            "1/x": [(Op.VAR, 0), (Op.INV, 0)],
            "log(x)": [(Op.VAR, 0), (Op.LOG, 0)],
            "x*y": [(Op.VAR, 0), (Op.VAR, 1), (Op.MUL, 0)],
            "x+y": [(Op.VAR, 0), (Op.VAR, 1), (Op.ADD, 0)],
            "x^0.5": [(Op.VAR, 0), (Op.SQRT, 0)],
            "x^0.25": [(Op.VAR, 0), (Op.SQRT, 0), (Op.SQRT, 0)],
            
            # Σ-Gravity specific templates
            "sqrt(R)/(1+R)": [
                (Op.VAR, 0), (Op.SQRT, 0),  # sqrt(R)
                (Op.CONST, 1), (Op.VAR, 0), (Op.ADD, 0),  # 1 + R
                (Op.DIV, 0)  # sqrt(R) / (1 + R)
            ],
            "R^a/(1+R^b)": [
                (Op.VAR, 0), (Op.CONST, 0.5), (Op.POW, 0),  # R^0.5
                (Op.CONST, 1), (Op.VAR, 0), (Op.ADD, 0),  # 1 + R
                (Op.DIV, 0)
            ],
            "1+sqrt(c/g)": [
                (Op.CONST, 1),
                (Op.CONST, 1.2), (Op.VAR, 1), (Op.DIV, 0), (Op.SQRT, 0),  # sqrt(1.2/g)
                (Op.ADD, 0)  # 1 + sqrt(1.2/g)
            ],
        }
        
        if template in templates:
            return Expression(templates[template])
        else:
            return self.random_expr()


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def mutate_expr(
    expr: Expression,
    generator: ExpressionGenerator,
    rate: float = 0.15
) -> Expression:
    """Mutate an expression"""
    new_instr = list(expr.instructions)
    
    for i in range(len(new_instr)):
        if random.random() > rate:
            continue
        
        op, val = new_instr[i]
        
        if op == Op.CONST:
            # Mutate constant
            if random.random() < 0.3:
                new_instr[i] = (Op.CONST, generator.random_const())
            elif random.random() < 0.5:
                new_instr[i] = (Op.CONST, val * random.uniform(0.5, 2.0))
            else:
                new_instr[i] = (Op.CONST, val + random.uniform(-1, 1))
        
        elif op == Op.VAR:
            # Change variable
            new_instr[i] = (Op.VAR, float(random.randint(0, generator.num_vars - 1)))
        
        elif OP_ARITY.get(op, 0) == 2:
            # Change binary operation
            new_op = generator._weighted_choice(generator.binary_ops)
            new_instr[i] = (new_op, 0.0)
        
        elif OP_ARITY.get(op, 0) == 1:
            # Change unary operation
            new_op = generator._weighted_choice(generator.unary_ops)
            new_instr[i] = (new_op, 0.0)
    
    return Expression(new_instr)


def crossover_expr(
    p1: Expression,
    p2: Expression,
    max_size: int = 40
) -> Tuple[Expression, Expression]:
    """Single-point crossover"""
    if len(p1) < 2 or len(p2) < 2:
        return p1.copy(), p2.copy()
    
    pt1 = random.randint(1, len(p1) - 1)
    pt2 = random.randint(1, len(p2) - 1)
    
    c1 = Expression(list(p1.instructions[:pt1]) + list(p2.instructions[pt2:]))
    c2 = Expression(list(p2.instructions[:pt2]) + list(p1.instructions[pt1:]))
    
    # Limit size
    c1.instructions = c1.instructions[:max_size]
    c2.instructions = c2.instructions[:max_size]
    
    return c1, c2


def subtree_crossover(
    p1: Expression,
    p2: Expression,
    max_size: int = 40
) -> Tuple[Expression, Expression]:
    """More sophisticated subtree crossover"""
    # Find valid crossover points (operation nodes)
    ops1 = [(i, op) for i, (op, _) in enumerate(p1.instructions) if OP_ARITY.get(op, 0) > 0]
    ops2 = [(i, op) for i, (op, _) in enumerate(p2.instructions) if OP_ARITY.get(op, 0) > 0]
    
    if not ops1 or not ops2:
        return crossover_expr(p1, p2, max_size)
    
    # Select crossover points
    pt1, _ = random.choice(ops1)
    pt2, _ = random.choice(ops2)
    
    # Create children
    c1 = Expression(list(p1.instructions[:pt1]) + list(p2.instructions[pt2:]))
    c2 = Expression(list(p2.instructions[:pt2]) + list(p1.instructions[pt1:]))
    
    c1.instructions = c1.instructions[:max_size]
    c2.instructions = c2.instructions[:max_size]
    
    return c1, c2


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'Expression', 'Op', 'OP_ARITY', 'OP_NAMES',
    'PHYSICS_CONSTANTS', 'ExpressionGenerator',
    'safe_eval_gpu', 'batch_evaluate_fitness',
    'mutate_expr', 'crossover_expr', 'subtree_crossover',
    'GPU_AVAILABLE', 'NUM_CORES', 'xp', 'cp',
]


if __name__ == "__main__":
    print("\nΣ-Gravity Discovery Engine - Core Module")
    print("Import this module or run run_sigma_discovery.py")

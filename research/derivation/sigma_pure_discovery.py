#!/usr/bin/env python3
"""
Σ-GRAVITY PURE FIRST-PRINCIPLES DISCOVERY ENGINE
=================================================

NO TEMPLATES. NO PHYSICS ASSUMPTIONS. PURE DATA-DRIVEN DISCOVERY.

Goal: Discover the field equations that explain Σ-Gravity coherence
effects from astronomical data alone.

Optimized for massive scale:
- RTX 5090 (32GB VRAM) - GPU batch evaluation
- 10-core CPU - parallel evolution
- Billions of candidate formulas
- Aggressive pruning of unpromising branches

Author: Σ-Gravity Research - First Principles Derivation
"""

import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from enum import IntEnum
import random
import time
import json
import gc
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# GPU CONFIGURATION - MAXIMIZE UTILIZATION
# =============================================================================

try:
    import cupy as cp
    from cupy import cuda
    
    GPU_AVAILABLE = True
    
    # Configure for maximum throughput
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    device = cuda.Device(0)
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    gpu_mem = device.mem_info[1] / 1e9
    
    print(f"✓ GPU: {gpu_name}")
    print(f"  Memory: {gpu_mem:.1f} GB")
    print(f"  Mode: MAXIMUM THROUGHPUT")
    
    xp = cp
    
except ImportError:
    GPU_AVAILABLE = False
    xp = np
    cp = np
    print("⚠ GPU not available - falling back to CPU")

NUM_CORES = min(10, mp.cpu_count())
print(f"✓ CPU Cores: {NUM_CORES}")


# =============================================================================
# COMPREHENSIVE OPERATION SET - ALL MATHEMATICAL PRIMITIVES
# =============================================================================

class Op(IntEnum):
    """Complete set of mathematical operations"""
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
    ABS = 26
    
    # Transcendental
    LOG = 30      # log10
    LN = 31       # natural log
    EXP = 32
    LOG2 = 33
    
    # Trigonometric
    SIN = 40
    COS = 41
    TAN = 42
    TANH = 43
    SINH = 44
    COSH = 45
    ATAN = 46
    
    # Special physics functions
    SIGMOID = 50  # 1/(1+exp(-x))
    SOFTPLUS = 51 # log(1+exp(x))
    ERF = 52      # Error function
    HEAVISIDE = 53  # Step function (smoothed)


OP_ARITY = {
    Op.CONST: 0, Op.VAR: 0,
    Op.ADD: 2, Op.SUB: 2, Op.MUL: 2, Op.DIV: 2, Op.POW: 2,
    Op.NEG: 1, Op.INV: 1, Op.SQUARE: 1, Op.CUBE: 1, Op.SQRT: 1, 
    Op.CBRT: 1, Op.ABS: 1,
    Op.LOG: 1, Op.LN: 1, Op.EXP: 1, Op.LOG2: 1,
    Op.SIN: 1, Op.COS: 1, Op.TAN: 1, Op.TANH: 1, Op.SINH: 1, 
    Op.COSH: 1, Op.ATAN: 1,
    Op.SIGMOID: 1, Op.SOFTPLUS: 1, Op.ERF: 1, Op.HEAVISIDE: 1,
}

# NO PHYSICS CONSTANTS - discover them from data
# Only mathematical constants and small integers
BASE_CONSTANTS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    -1, -2, -3,
    0.5, 0.25, 0.1, 0.01, 0.001,
    1.5, 2.5, 3.5,
    np.pi, 2*np.pi, np.pi/2,
    np.e, np.sqrt(2), np.sqrt(3),
]


# =============================================================================
# EXPRESSION REPRESENTATION
# =============================================================================

@dataclass  
class Expression:
    """Stack-based expression for GPU-efficient evaluation."""
    instructions: List[Tuple[int, float]]
    
    def __len__(self):
        return len(self.instructions)
    
    def __hash__(self):
        return hash(tuple(self.instructions))
    
    def copy(self) -> 'Expression':
        return Expression(list(self.instructions))
    
    def complexity(self) -> int:
        """Compute complexity score"""
        score = 0
        for op, _ in self.instructions:
            if op in [Op.CONST, Op.VAR]:
                score += 1
            elif op in [Op.ADD, Op.SUB, Op.MUL, Op.NEG, Op.ABS]:
                score += 1
            elif op in [Op.DIV, Op.INV, Op.SQRT, Op.SQUARE]:
                score += 2
            elif op in [Op.POW, Op.LOG, Op.LN, Op.EXP, Op.CUBE, Op.CBRT]:
                score += 3
            elif op in [Op.SIN, Op.COS, Op.TANH, Op.SIGMOID]:
                score += 4
            else:
                score += 3
        return score
    
    def to_string(self, var_names: List[str]) -> str:
        """Convert to human-readable notation"""
        stack = []
        try:
            for op, val in self.instructions:
                if op == Op.CONST:
                    if val == 0:
                        stack.append("0")
                    elif abs(val - round(val)) < 0.001 and abs(val) < 100:
                        stack.append(str(int(round(val))))
                    elif abs(val - np.pi) < 0.01:
                        stack.append("π")
                    elif abs(val - np.e) < 0.01:
                        stack.append("e")
                    else:
                        stack.append(f"{val:.4g}")
                
                elif op == Op.VAR:
                    idx = int(val)
                    stack.append(var_names[idx] if idx < len(var_names) else f"x{idx}")
                
                # Binary ops
                elif op == Op.ADD:
                    if len(stack) < 2: return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} + {b})")
                elif op == Op.SUB:
                    if len(stack) < 2: return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a} - {b})")
                elif op == Op.MUL:
                    if len(stack) < 2: return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"{a}·{b}" if len(str(a)) < 8 else f"({a} × {b})")
                elif op == Op.DIV:
                    if len(stack) < 2: return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a}/{b})")
                elif op == Op.POW:
                    if len(stack) < 2: return "[invalid]"
                    b, a = stack.pop(), stack.pop()
                    stack.append(f"({a}^{b})")
                
                # Unary ops
                elif op == Op.NEG:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"(-{stack.pop()})")
                elif op == Op.INV:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"(1/{stack.pop()})")
                elif op == Op.SQUARE:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"{stack.pop()}²")
                elif op == Op.CUBE:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"{stack.pop()}³")
                elif op == Op.SQRT:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"√{stack.pop()}")
                elif op == Op.CBRT:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"∛{stack.pop()}")
                elif op == Op.ABS:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"|{stack.pop()}|")
                
                # Transcendental
                elif op == Op.LOG:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"log({stack.pop()})")
                elif op == Op.LN:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"ln({stack.pop()})")
                elif op == Op.EXP:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"exp({stack.pop()})")
                elif op == Op.LOG2:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"log₂({stack.pop()})")
                
                # Trigonometric
                elif op == Op.SIN:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"sin({stack.pop()})")
                elif op == Op.COS:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"cos({stack.pop()})")
                elif op == Op.TAN:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"tan({stack.pop()})")
                elif op == Op.TANH:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"tanh({stack.pop()})")
                elif op == Op.SINH:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"sinh({stack.pop()})")
                elif op == Op.COSH:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"cosh({stack.pop()})")
                elif op == Op.ATAN:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"atan({stack.pop()})")
                
                # Special
                elif op == Op.SIGMOID:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"σ({stack.pop()})")
                elif op == Op.SOFTPLUS:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"softplus({stack.pop()})")
                elif op == Op.ERF:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"erf({stack.pop()})")
                elif op == Op.HEAVISIDE:
                    if len(stack) < 1: return "[invalid]"
                    stack.append(f"H({stack.pop()})")
            
            return stack[0] if stack else "?"
        except Exception:
            return "[invalid]"


# =============================================================================
# GPU-ACCELERATED EVALUATION
# =============================================================================

def safe_eval_gpu(expr: Expression, X: Dict[str, any], use_gpu: bool = True) -> any:
    """Evaluate expression with full numerical safety."""
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
    
    try:
        for op, val in expr.instructions:
            if op == Op.CONST:
                stack.append(backend.full(n, val, dtype=backend.float64))
            elif op == Op.VAR:
                idx = int(val)
                if idx < len(var_arrays):
                    stack.append(var_arrays[idx].copy())
                else:
                    return backend.full(n, np.nan)
            
            # Binary arithmetic
            elif op == Op.ADD:
                if len(stack) < 2: return backend.full(n, np.nan)
                b, a = stack.pop(), stack.pop()
                stack.append(a + b)
            elif op == Op.SUB:
                if len(stack) < 2: return backend.full(n, np.nan)
                b, a = stack.pop(), stack.pop()
                stack.append(a - b)
            elif op == Op.MUL:
                if len(stack) < 2: return backend.full(n, np.nan)
                b, a = stack.pop(), stack.pop()
                stack.append(a * b)
            elif op == Op.DIV:
                if len(stack) < 2: return backend.full(n, np.nan)
                b, a = stack.pop(), stack.pop()
                stack.append(backend.where(backend.abs(b) > EPS, a / b, backend.sign(a) * CLIP))
            elif op == Op.POW:
                if len(stack) < 2: return backend.full(n, np.nan)
                b, a = stack.pop(), stack.pop()
                b_safe = backend.clip(b, -10, 10)
                a_safe = backend.where(a > 0, a, backend.abs(a) + EPS)
                stack.append(backend.power(a_safe, b_safe))
            
            # Unary basic
            elif op == Op.NEG:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(-stack.pop())
            elif op == Op.INV:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.where(backend.abs(a) > EPS, 1.0 / a, CLIP))
            elif op == Op.SQUARE:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(a * a)
            elif op == Op.CUBE:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(a * a * a)
            elif op == Op.SQRT:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.sqrt(backend.abs(a) + EPS))
            elif op == Op.CBRT:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.cbrt(a))
            elif op == Op.ABS:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(backend.abs(stack.pop()))
            
            # Transcendental
            elif op == Op.LOG:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.where(a > EPS, backend.log10(a), -30))
            elif op == Op.LN:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.where(a > EPS, backend.log(a), -30))
            elif op == Op.EXP:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.exp(backend.clip(a, -30, 30)))
            elif op == Op.LOG2:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.where(a > EPS, backend.log2(a), -30))
            
            # Trigonometric
            elif op == Op.SIN:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(backend.sin(stack.pop()))
            elif op == Op.COS:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(backend.cos(stack.pop()))
            elif op == Op.TAN:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.clip(backend.tan(a), -CLIP, CLIP))
            elif op == Op.TANH:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(backend.tanh(stack.pop()))
            elif op == Op.SINH:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.clip(backend.sinh(backend.clip(a, -30, 30)), -CLIP, CLIP))
            elif op == Op.COSH:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.cosh(backend.clip(a, -30, 30)))
            elif op == Op.ATAN:
                if len(stack) < 1: return backend.full(n, np.nan)
                stack.append(backend.arctan(stack.pop()))
            
            # Special functions
            elif op == Op.SIGMOID:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(1.0 / (1.0 + backend.exp(-backend.clip(a, -30, 30))))
            elif op == Op.SOFTPLUS:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                stack.append(backend.log(1.0 + backend.exp(backend.clip(a, -30, 30))))
            elif op == Op.ERF:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                # Approximation: erf(x) ≈ tanh(√π * x * (1 + 0.044715 * x²))
                stack.append(backend.tanh(1.7724538509 * a * (1 + 0.044715 * a * a)))
            elif op == Op.HEAVISIDE:
                if len(stack) < 1: return backend.full(n, np.nan)
                a = stack.pop()
                # Smooth approximation: σ(10*x)
                stack.append(1.0 / (1.0 + backend.exp(-10 * backend.clip(a, -3, 3))))
        
        result = stack[0] if stack else backend.zeros(n, dtype=backend.float64)
        return backend.clip(result, -CLIP, CLIP)
    
    except Exception:
        if use_gpu and GPU_AVAILABLE:
            return cp.full(n, np.nan)
        return np.full(n, np.nan)


def batch_evaluate_fitness(
    programs: List[Expression],
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    use_gpu: bool = True
) -> List[Tuple[float, float, int]]:
    """Batch evaluate fitness with GPU acceleration."""
    results = []
    
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    
    for prog in programs:
        try:
            y_pred = safe_eval_gpu(prog, X, use_gpu=use_gpu)
            
            if GPU_AVAILABLE and use_gpu:
                y_pred = cp.asnumpy(y_pred)
            
            valid = np.isfinite(y_pred)
            if valid.sum() < len(y) * 0.5:
                results.append((1e30, -1.0, prog.complexity()))
                continue
            
            y_v = y[valid]
            y_pred_v = y_pred[valid]
            
            residuals = y_v - y_pred_v
            mse = float(np.mean(residuals**2))
            ss_res = float(np.sum(residuals**2))
            
            # Use adjusted ss_tot for valid points only
            ss_tot_valid = np.sum((y_v - np.mean(y_v))**2)
            r2 = 1.0 - ss_res / ss_tot_valid if ss_tot_valid > 0 else 0.0
            
            results.append((mse, r2, prog.complexity()))
            
        except Exception:
            results.append((1e30, -1.0, prog.complexity()))
    
    return results


# =============================================================================
# PURE RANDOM EXPRESSION GENERATOR - NO TEMPLATES
# =============================================================================

class PureExpressionGenerator:
    """
    Generate expressions from first principles.
    NO PHYSICS ASSUMPTIONS. NO TEMPLATES.
    """
    
    def __init__(
        self,
        num_vars: int,
        var_names: List[str],
        max_depth: int = 8,
        allow_transcendental: bool = True,
        allow_trig: bool = True,
    ):
        self.num_vars = num_vars
        self.var_names = var_names
        self.max_depth = max_depth
        self.allow_transcendental = allow_transcendental
        self.allow_trig = allow_trig
        
        # All operations equally weighted - no physics bias
        self.binary_ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.POW]
        
        self.unary_ops = [
            Op.NEG, Op.INV, Op.SQUARE, Op.CUBE, Op.SQRT, Op.CBRT, Op.ABS
        ]
        
        if allow_transcendental:
            self.unary_ops.extend([Op.LOG, Op.LN, Op.EXP])
        
        if allow_trig:
            self.unary_ops.extend([Op.TANH, Op.SIGMOID, Op.ATAN])
        
        # Constants discovered from data, not physics
        self.constants = BASE_CONSTANTS.copy()
    
    def add_discovered_constant(self, c: float):
        """Add a constant discovered to be useful"""
        if c not in self.constants and np.isfinite(c):
            self.constants.append(c)
    
    def random_const(self) -> float:
        """Generate random constant"""
        r = random.random()
        if r < 0.5:
            # Use known useful constants
            return random.choice(self.constants)
        elif r < 0.8:
            # Small random value
            return random.uniform(-10, 10)
        else:
            # Larger random value
            return random.uniform(-100, 100)
    
    def random_expr(self, max_depth: int = None) -> Expression:
        """Generate purely random expression"""
        if max_depth is None:
            max_depth = self.max_depth
        
        instructions = []
        
        def grow(depth: int):
            # Probability of terminal increases with depth
            terminal_prob = 0.2 + 0.6 * (1 - depth / self.max_depth)
            
            if depth <= 1 or random.random() < terminal_prob:
                # Terminal: variable or constant
                if random.random() < 0.6:
                    instructions.append((Op.VAR, float(random.randint(0, self.num_vars - 1))))
                else:
                    instructions.append((Op.CONST, self.random_const()))
            else:
                # Operation
                if random.random() < 0.55:
                    # Binary operation
                    grow(depth - 1)
                    grow(depth - 1)
                    instructions.append((random.choice(self.binary_ops), 0.0))
                else:
                    # Unary operation
                    grow(depth - 1)
                    instructions.append((random.choice(self.unary_ops), 0.0))
        
        grow(max_depth)
        return Expression(instructions)


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def mutate_expr(expr: Expression, generator: PureExpressionGenerator, rate: float = 0.2) -> Expression:
    """Mutate expression with higher rate for exploration"""
    new_instr = list(expr.instructions)
    
    for i in range(len(new_instr)):
        if random.random() > rate:
            continue
        
        op, val = new_instr[i]
        
        if op == Op.CONST:
            r = random.random()
            if r < 0.25:
                new_instr[i] = (Op.CONST, generator.random_const())
            elif r < 0.5:
                new_instr[i] = (Op.CONST, val * random.uniform(0.5, 2.0))
            elif r < 0.75:
                new_instr[i] = (Op.CONST, val + random.uniform(-1, 1))
            else:
                # Fine-tune
                new_instr[i] = (Op.CONST, val * random.uniform(0.9, 1.1))
        
        elif op == Op.VAR:
            new_instr[i] = (Op.VAR, float(random.randint(0, generator.num_vars - 1)))
        
        elif OP_ARITY.get(op, 0) == 2:
            new_instr[i] = (random.choice(generator.binary_ops), 0.0)
        
        elif OP_ARITY.get(op, 0) == 1:
            new_instr[i] = (random.choice(generator.unary_ops), 0.0)
    
    return Expression(new_instr)


def crossover_expr(p1: Expression, p2: Expression, max_size: int = 50) -> Tuple[Expression, Expression]:
    """Crossover two expressions"""
    if len(p1) < 2 or len(p2) < 2:
        return p1.copy(), p2.copy()
    
    pt1 = random.randint(1, len(p1) - 1)
    pt2 = random.randint(1, len(p2) - 1)
    
    c1 = Expression(list(p1.instructions[:pt1]) + list(p2.instructions[pt2:]))
    c2 = Expression(list(p2.instructions[:pt2]) + list(p1.instructions[pt1:]))
    
    c1.instructions = c1.instructions[:max_size]
    c2.instructions = c2.instructions[:max_size]
    
    return c1, c2


# =============================================================================
# DISCOVERY RESULT
# =============================================================================

@dataclass
class DiscoveryResult:
    """Stores a discovered formula"""
    expression: Expression
    formula_string: str
    mse: float
    r_squared: float
    complexity: int
    generation: int = 0
    
    def to_dict(self) -> dict:
        return {
            'formula': self.formula_string,
            'mse': self.mse,
            'r_squared': self.r_squared,
            'complexity': self.complexity,
            'generation': self.generation,
        }


# =============================================================================
# MASSIVE SCALE PURE DISCOVERY ENGINE
# =============================================================================

class PureDiscoveryEngine:
    """
    First-principles formula discovery.
    NO TEMPLATES. NO PHYSICS ASSUMPTIONS. MASSIVE SCALE.
    """
    
    def __init__(
        self,
        var_names: List[str],
        population_size: int = 10000,      # HUGE population
        max_generations: int = 500,         # Many generations
        num_islands: int = 10,              # Many islands for diversity
        migration_rate: float = 0.15,
        migration_interval: int = 20,
        tournament_size: int = 10,
        crossover_rate: float = 0.75,
        mutation_rate: float = 0.25,        # Higher mutation for exploration
        elitism: int = 50,                  # Keep more elites
        complexity_weight: float = 0.001,   # Low penalty for complexity
        max_depth: int = 10,                # Allow deep expressions
        seed: int = None,
    ):
        self.var_names = var_names
        self.num_vars = len(var_names)
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_islands = num_islands
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.complexity_weight = complexity_weight
        self.max_depth = max_depth
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if GPU_AVAILABLE:
                cp.random.seed(seed)
        
        self.generator = PureExpressionGenerator(
            num_vars=self.num_vars,
            var_names=var_names,
            max_depth=max_depth,
        )
        
        self.islands: List[List[Expression]] = []
        self.best_overall: Optional[DiscoveryResult] = None
        self.pareto_front: List[DiscoveryResult] = []
        self.history: List[dict] = []
        self.all_evaluated: Dict[str, DiscoveryResult] = {}
        self.total_evaluations = 0
    
    def _init_population(self, island_size: int) -> List[Expression]:
        """Initialize with PURELY RANDOM expressions - NO TEMPLATES"""
        pop = []
        
        # Generate random expressions at various depths
        for _ in range(island_size):
            depth = random.randint(2, self.max_depth)
            pop.append(self.generator.random_expr(max_depth=depth))
        
        return pop
    
    def _compute_score(self, mse: float, r2: float, complexity: int, y_scale: float) -> float:
        """Score favoring accuracy with mild complexity penalty"""
        if mse >= 1e30 or r2 < -1:
            return 1e30
        normalized_mse = mse / (y_scale**2 + 1e-10)
        return np.log(normalized_mse + 1e-10) + self.complexity_weight * complexity
    
    def _tournament_select(self, pop: List[Expression], scores: List[float]) -> int:
        tournament = random.sample(range(len(pop)), min(self.tournament_size, len(pop)))
        return min(tournament, key=lambda i: scores[i])
    
    def _update_pareto_front(self, result: DiscoveryResult):
        """Track Pareto-optimal solutions"""
        dominated = []
        is_dominated = False
        
        for i, existing in enumerate(self.pareto_front):
            if (existing.mse <= result.mse and existing.complexity <= result.complexity and
                (existing.mse < result.mse or existing.complexity < result.complexity)):
                is_dominated = True
                break
            
            if (result.mse <= existing.mse and result.complexity <= existing.complexity and
                (result.mse < existing.mse or result.complexity < existing.complexity)):
                dominated.append(i)
        
        if not is_dominated:
            self.pareto_front = [r for i, r in enumerate(self.pareto_front) if i not in dominated]
            self.pareto_front.append(result)
            
            if len(self.pareto_front) > 200:
                self.pareto_front.sort(key=lambda x: x.mse)
                self.pareto_front = self.pareto_front[:200]
    
    def evolve(
        self,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        verbose: bool = True,
        early_stop_r2: float = 0.999,
    ) -> Tuple[Expression, float, float]:
        """
        Run massive-scale evolutionary search.
        NO TEMPLATES - pure discovery from data.
        """
        start_time = time.time()
        
        y_scale = float(np.std(y))
        y_mean = float(np.mean(y))
        
        if verbose:
            print(f"\n{'='*75}")
            print(f"  PURE FIRST-PRINCIPLES DISCOVERY")
            print(f"  NO TEMPLATES. NO PHYSICS ASSUMPTIONS.")
            print(f"{'='*75}")
            print(f"\n  Data: {len(y):,} points")
            print(f"  Variables: {self.var_names}")
            print(f"  Target: mean={y_mean:.4f}, std={y_scale:.4f}")
            print(f"\n  Population: {self.population_size:,} × {self.num_islands} islands = {self.population_size * self.num_islands:,}")
            print(f"  Generations: {self.max_generations}")
            print(f"  Max depth: {self.max_depth}")
            print()
        
        # Initialize islands
        island_size = self.population_size // self.num_islands
        self.islands = []
        
        for i in range(self.num_islands):
            island = self._init_population(island_size)
            self.islands.append(island)
        
        if verbose:
            print(f"  Initialized {self.num_islands} islands × {island_size} individuals")
            print(f"  Starting evolution...\n")
        
        # Main evolution loop
        for generation in range(self.max_generations):
            gen_start = time.time()
            
            gen_best_result = None
            gen_best_score = float('inf')
            gen_evaluations = 0
            
            # Evolve each island
            for island_idx, island in enumerate(self.islands):
                # Batch evaluate
                fitness_results = batch_evaluate_fitness(island, X, y, use_gpu=GPU_AVAILABLE)
                gen_evaluations += len(island)
                
                # Compute scores
                scores = []
                for (mse, r2, compl), expr in zip(fitness_results, island):
                    score = self._compute_score(mse, r2, compl, y_scale)
                    scores.append(score)
                    
                    formula_str = expr.to_string(self.var_names)
                    if formula_str != "[invalid]" and formula_str not in self.all_evaluated:
                        result = DiscoveryResult(
                            expression=expr.copy(),
                            formula_string=formula_str,
                            mse=mse,
                            r_squared=r2,
                            complexity=compl,
                            generation=generation
                        )
                        self.all_evaluated[formula_str] = result
                        
                        if r2 > 0.3:  # Lower threshold for Pareto
                            self._update_pareto_front(result)
                    
                    if score < gen_best_score and formula_str != "[invalid]":
                        gen_best_score = score
                        gen_best_result = DiscoveryResult(
                            expression=expr.copy(),
                            formula_string=formula_str,
                            mse=mse,
                            r_squared=r2,
                            complexity=compl,
                            generation=generation
                        )
                
                # Selection and reproduction
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
                new_island = [island[i].copy() for i in sorted_indices[:self.elitism]]
                
                while len(new_island) < island_size:
                    p1_idx = self._tournament_select(island, scores)
                    
                    if random.random() < self.crossover_rate:
                        p2_idx = self._tournament_select(island, scores)
                        c1, c2 = crossover_expr(island[p1_idx], island[p2_idx])
                        c1 = mutate_expr(c1, self.generator, self.mutation_rate * 0.5)
                        c2 = mutate_expr(c2, self.generator, self.mutation_rate * 0.5)
                        new_island.extend([c1, c2])
                    else:
                        child = mutate_expr(island[p1_idx], self.generator, self.mutation_rate)
                        new_island.append(child)
                    
                    # Occasionally inject fresh random individuals
                    if random.random() < 0.02:
                        new_island.append(self.generator.random_expr())
                
                self.islands[island_idx] = new_island[:island_size]
            
            self.total_evaluations += gen_evaluations
            
            # Migration
            if generation > 0 and generation % self.migration_interval == 0:
                num_migrants = max(1, int(island_size * self.migration_rate))
                for i in range(self.num_islands):
                    source = (i + 1) % self.num_islands
                    migrants = self.islands[source][:num_migrants]
                    self.islands[i] = migrants + self.islands[i][:-num_migrants]
            
            # Update best
            if gen_best_result and (self.best_overall is None or gen_best_result.r_squared > self.best_overall.r_squared):
                self.best_overall = gen_best_result
            
            gen_time = time.time() - gen_start
            
            # History
            self.history.append({
                'generation': generation,
                'best_r2': gen_best_result.r_squared if gen_best_result else -1,
                'best_mse': gen_best_result.mse if gen_best_result else 1e30,
                'pareto_size': len(self.pareto_front),
                'unique_evaluated': len(self.all_evaluated),
                'total_evaluations': self.total_evaluations,
            })
            
            # Progress
            if verbose and (generation % 5 == 0 or generation == self.max_generations - 1):
                formula = gen_best_result.formula_string[:50] if gen_best_result else "?"
                r2 = gen_best_result.r_squared if gen_best_result else -1
                print(f"  Gen {generation:4d} | R²={r2:.6f} | "
                      f"Pareto={len(self.pareto_front):3d} | "
                      f"Unique={len(self.all_evaluated):,} | "
                      f"{gen_time:.1f}s | {formula}")
            
            # Early stopping
            if gen_best_result and gen_best_result.r_squared > early_stop_r2:
                if verbose:
                    print(f"\n  ✓ Converged at generation {generation} (R² > {early_stop_r2})")
                break
            
            # Memory cleanup
            if generation % 50 == 0:
                gc.collect()
                if GPU_AVAILABLE:
                    mempool.free_all_blocks()
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*75}")
            print(f"  DISCOVERY COMPLETE")
            print(f"{'='*75}")
            print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Total evaluations: {self.total_evaluations:,}")
            print(f"  Unique formulas: {len(self.all_evaluated):,}")
            print(f"  Pareto front: {len(self.pareto_front)} solutions")
        
        return (
            self.best_overall.expression if self.best_overall else None,
            self.best_overall.mse if self.best_overall else 1e30,
            self.best_overall.r_squared if self.best_overall else -1
        )
    
    def get_pareto_front(self) -> List[DiscoveryResult]:
        return sorted(self.pareto_front, key=lambda x: -x.r_squared)
    
    def get_top_formulas(self, n: int = 50) -> List[DiscoveryResult]:
        results = list(self.all_evaluated.values())
        results.sort(key=lambda x: -x.r_squared)
        return results[:n]
    
    def export_results(self, filepath: str):
        results = {
            'best': self.best_overall.to_dict() if self.best_overall else None,
            'pareto_front': [r.to_dict() for r in self.get_pareto_front()],
            'top_formulas': [r.to_dict() for r in self.get_top_formulas(100)],
            'history': self.history,
            'config': {
                'variables': self.var_names,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'num_islands': self.num_islands,
                'total_evaluations': self.total_evaluations,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results exported to {filepath}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'Expression', 'Op', 'OP_ARITY',
    'PureExpressionGenerator', 'PureDiscoveryEngine',
    'DiscoveryResult',
    'safe_eval_gpu', 'batch_evaluate_fitness',
    'GPU_AVAILABLE', 'NUM_CORES',
]

if __name__ == "__main__":
    print("\nΣ-Gravity Pure Discovery Engine")
    print("Run run_pure_discovery.py to start")

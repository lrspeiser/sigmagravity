"""
Tensor Symbolic Regression
==========================

Discovers tensor field equations from spacetime curvature data.

Unlike scalar symbolic regression, this searches over tensor expressions:
- Ricci tensor R_μν
- Ricci scalar R
- Metric tensor g_μν
- Stress-energy tensor T_μν
- Einstein tensor G_μν = R_μν - ½Rg_μν

The goal: given curvature data, discover that G_μν = 8πT_μν
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from time import time
import copy


@dataclass
class TensorData:
    """
    Container for spacetime tensor data at multiple points.
    
    All tensors are stored as (N, 4, 4) arrays where N is the number
    of spacetime points sampled.
    """
    g: np.ndarray      # Metric tensor g_μν
    g_inv: np.ndarray  # Inverse metric g^μν
    R_ud: np.ndarray   # Ricci tensor R_μν (mixed indices for contraction)
    R: np.ndarray      # Ricci scalar (N,)
    T: np.ndarray      # Stress-energy tensor T_μν
    
    # Derived quantities (computed on demand)
    _G: Optional[np.ndarray] = None  # Einstein tensor
    
    @property
    def G(self) -> np.ndarray:
        """Einstein tensor G_μν = R_μν - ½Rg_μν"""
        if self._G is None:
            self._G = self.R_ud - 0.5 * self.R[:, None, None] * self.g
        return self._G
    
    @property
    def n_points(self) -> int:
        return self.g.shape[0]


class TensorTerm:
    """
    Represents a term in a tensor equation.
    
    A term is: coefficient × tensor_expression
    
    Tensor expressions can be:
    - 'g': metric g_μν
    - 'R_ud': Ricci tensor R_μν
    - 'R*g': Scalar curvature times metric
    - 'T': Stress-energy tensor
    - 'G': Einstein tensor (R_μν - ½Rg_μν)
    - 'I': Identity (Kronecker delta)
    """
    
    TENSOR_TYPES = ['g', 'R_ud', 'R*g', 'T', 'G', 'I']
    
    def __init__(self, coefficient: float, tensor_type: str):
        self.coefficient = coefficient
        self.tensor_type = tensor_type
    
    def evaluate(self, data: TensorData) -> np.ndarray:
        """Evaluate this term on tensor data. Returns (N, 4, 4) array."""
        if self.tensor_type == 'g':
            tensor = data.g
        elif self.tensor_type == 'R_ud':
            tensor = data.R_ud
        elif self.tensor_type == 'R*g':
            tensor = data.R[:, None, None] * data.g
        elif self.tensor_type == 'T':
            tensor = data.T
        elif self.tensor_type == 'G':
            tensor = data.G
        elif self.tensor_type == 'I':
            tensor = np.tile(np.eye(4), (data.n_points, 1, 1))
        else:
            raise ValueError(f"Unknown tensor type: {self.tensor_type}")
        
        return self.coefficient * tensor
    
    def to_string(self) -> str:
        coef_str = f"{self.coefficient:.4g}"
        if abs(self.coefficient - 1.0) < 0.001:
            coef_str = ""
        elif abs(self.coefficient + 1.0) < 0.001:
            coef_str = "-"
        else:
            coef_str = f"{self.coefficient:.4g}·"
        
        tensor_str = {
            'g': 'g_μν',
            'R_ud': 'R_μν', 
            'R*g': 'Rg_μν',
            'T': 'T_μν',
            'G': 'G_μν',
            'I': 'δ_μν',
        }.get(self.tensor_type, self.tensor_type)
        
        return f"{coef_str}{tensor_str}"
    
    def copy(self) -> 'TensorTerm':
        return TensorTerm(self.coefficient, self.tensor_type)


class TensorEquation:
    """
    Represents a tensor equation as a sum of terms.
    
    LHS = Σ(coefficient_i × tensor_i) = 0
    
    For Einstein's equation:
    G_μν - 8πT_μν = 0
    or equivalently:
    R_μν - 0.5·Rg_μν - 8πT_μν = 0
    """
    
    def __init__(self, terms: List[TensorTerm]):
        self.terms = terms
        self._fitness: Optional[float] = None
    
    @property
    def fitness(self) -> Optional[float]:
        return self._fitness
    
    @fitness.setter
    def fitness(self, value: float):
        self._fitness = value
    
    def evaluate(self, data: TensorData) -> np.ndarray:
        """
        Evaluate equation LHS on data.
        Returns (N, 4, 4) tensor that should be zero if equation is correct.
        """
        result = np.zeros((data.n_points, 4, 4))
        for term in self.terms:
            result += term.evaluate(data)
        return result
    
    def compute_residual(self, data: TensorData) -> float:
        """Compute mean squared residual (should be 0 for valid equation)."""
        lhs = self.evaluate(data)
        return np.mean(lhs ** 2)
    
    def to_string(self) -> str:
        if not self.terms:
            return "0"
        
        parts = []
        for i, term in enumerate(self.terms):
            s = term.to_string()
            if i > 0 and not s.startswith('-'):
                s = '+ ' + s
            parts.append(s)
        
        return ' '.join(parts) + ' = 0'
    
    def copy(self) -> 'TensorEquation':
        return TensorEquation([t.copy() for t in self.terms])
    
    @staticmethod
    def random(n_terms: int = 3, 
               allowed_tensors: List[str] = None,
               coef_range: Tuple[float, float] = (-30, 30)) -> 'TensorEquation':
        """Generate a random tensor equation."""
        if allowed_tensors is None:
            allowed_tensors = ['R_ud', 'R*g', 'T', 'g']
        
        terms = []
        for _ in range(n_terms):
            coef = random.uniform(*coef_range)
            # Favor simple coefficients
            if random.random() < 0.3:
                coef = random.choice([1, -1, 2, -2, 0.5, -0.5, 8*np.pi, -8*np.pi])
            tensor_type = random.choice(allowed_tensors)
            terms.append(TensorTerm(coef, tensor_type))
        
        return TensorEquation(terms)
    
    def mutate(self, coef_range: Tuple[float, float] = (-30, 30),
               allowed_tensors: List[str] = None) -> 'TensorEquation':
        """Create mutated copy."""
        if allowed_tensors is None:
            allowed_tensors = ['R_ud', 'R*g', 'T', 'g']
        
        new_eq = self.copy()
        
        mutation_type = random.choice(['coef', 'tensor', 'add', 'remove'])
        
        if mutation_type == 'coef' and new_eq.terms:
            # Modify a coefficient
            term = random.choice(new_eq.terms)
            if random.random() < 0.5:
                term.coefficient *= random.uniform(0.8, 1.2)
            else:
                term.coefficient += random.uniform(-1, 1)
        
        elif mutation_type == 'tensor' and new_eq.terms:
            # Change tensor type
            term = random.choice(new_eq.terms)
            term.tensor_type = random.choice(allowed_tensors)
        
        elif mutation_type == 'add' and len(new_eq.terms) < 5:
            # Add a term
            coef = random.uniform(*coef_range)
            tensor_type = random.choice(allowed_tensors)
            new_eq.terms.append(TensorTerm(coef, tensor_type))
        
        elif mutation_type == 'remove' and len(new_eq.terms) > 1:
            # Remove a term
            new_eq.terms.pop(random.randint(0, len(new_eq.terms) - 1))
        
        new_eq._fitness = None
        return new_eq
    
    @staticmethod
    def crossover(parent1: 'TensorEquation', 
                  parent2: 'TensorEquation') -> 'TensorEquation':
        """Create offspring by combining terms from parents."""
        all_terms = parent1.terms + parent2.terms
        n_terms = random.randint(1, min(5, len(all_terms)))
        selected = random.sample(all_terms, n_terms)
        child = TensorEquation([t.copy() for t in selected])
        child._fitness = None
        return child


class TensorRegressor:
    """
    Genetic algorithm for discovering tensor field equations.
    """
    
    def __init__(self,
                 population_size: int = 1000,
                 max_generations: int = 100,
                 tournament_size: int = 5,
                 crossover_prob: float = 0.6,
                 mutation_prob: float = 0.3,
                 elitism: int = 5,
                 parsimony_coef: float = 0.001,
                 allowed_tensors: List[str] = None):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.parsimony_coef = parsimony_coef
        self.allowed_tensors = allowed_tensors or ['R_ud', 'R*g', 'T', 'g']
        
        self.population: List[TensorEquation] = []
        self.best_equation: Optional[TensorEquation] = None
        self.best_fitness = float('inf')
        self.history = []
    
    def _init_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            n_terms = random.randint(2, 4)
            eq = TensorEquation.random(n_terms, self.allowed_tensors)
            self.population.append(eq)
    
    def _compute_fitness(self, equation: TensorEquation, 
                         data: TensorData) -> float:
        """Compute fitness (lower is better)."""
        try:
            residual = equation.compute_residual(data)
            complexity = len(equation.terms)
            fitness = residual + self.parsimony_coef * complexity
            
            if not np.isfinite(fitness):
                return 1e20
            return fitness
        except Exception:
            return 1e20
    
    def _tournament_select(self) -> TensorEquation:
        """Tournament selection."""
        contestants = random.sample(self.population, self.tournament_size)
        return min(contestants, key=lambda x: x.fitness or float('inf'))
    
    def _evolve(self, data: TensorData):
        """Evolve one generation."""
        # Evaluate fitness
        for eq in self.population:
            if eq.fitness is None:
                eq.fitness = self._compute_fitness(eq, data)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness)
        
        # Track best
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_equation = self.population[0].copy()
        
        # Elitism
        new_pop = [eq.copy() for eq in self.population[:self.elitism]]
        
        # Generate rest
        while len(new_pop) < self.population_size:
            if random.random() < self.crossover_prob:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child = TensorEquation.crossover(p1, p2)
            else:
                parent = self._tournament_select()
                child = parent.mutate(allowed_tensors=self.allowed_tensors)
            
            if random.random() < self.mutation_prob:
                child = child.mutate(allowed_tensors=self.allowed_tensors)
            
            new_pop.append(child)
        
        self.population = new_pop[:self.population_size]
    
    def fit(self, data: TensorData, verbose: bool = True) -> 'TensorRegressor':
        """
        Run tensor symbolic regression.
        
        Args:
            data: TensorData containing curvature information
            verbose: Print progress
        
        Returns:
            self
        """
        if verbose:
            print(f"Starting tensor symbolic regression...")
            print(f"  Data points: {data.n_points}")
            print(f"  Population: {self.population_size}")
            print(f"  Allowed tensors: {self.allowed_tensors}")
        
        self._init_population()
        
        start_time = time()
        
        for gen in range(self.max_generations):
            self._evolve(data)
            
            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'best_equation': self.best_equation.to_string() if self.best_equation else None,
            })
            
            if verbose and gen % 10 == 0:
                eq_str = self.best_equation.to_string() if self.best_equation else "None"
                print(f"  Gen {gen:3d}: Loss = {self.best_fitness:.6e}  |  {eq_str}")
            
            if self.best_fitness < 1e-12:
                if verbose:
                    print(f"  Converged at generation {gen}!")
                break
        
        elapsed = time() - start_time
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s")
        
        return self
    
    def get_equation(self) -> str:
        """Get discovered equation as string."""
        if self.best_equation is None:
            return "No equation discovered"
        return self.best_equation.to_string()


# ============================================
# CURVATURE DATA GENERATORS
# ============================================

def generate_schwarzschild_data(n_points: int = 1000,
                                r_range: Tuple[float, float] = (3, 100),
                                M: float = 1.0) -> TensorData:
    """
    Generate curvature data from Schwarzschild spacetime.
    
    The Schwarzschild metric (in Schwarzschild coordinates):
    ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dΩ²
    
    This is a vacuum solution: T_μν = 0, so G_μν = 0
    """
    print(f"Generating Schwarzschild curvature data (M={M})...")
    
    # Sample radial positions (avoid horizon at r=2M)
    r = np.random.uniform(r_range[0] * M, r_range[1] * M, n_points)
    
    # Metric components
    f = 1 - 2*M/r  # g_tt and 1/g_rr
    
    # Build metric tensor g_μν (diagonal in Schwarzschild coords)
    g = np.zeros((n_points, 4, 4))
    g[:, 0, 0] = -f           # g_tt
    g[:, 1, 1] = 1/f          # g_rr
    g[:, 2, 2] = r**2         # g_θθ
    g[:, 3, 3] = r**2         # g_φφ (at θ=π/2)
    
    # Inverse metric
    g_inv = np.zeros((n_points, 4, 4))
    g_inv[:, 0, 0] = -1/f
    g_inv[:, 1, 1] = f
    g_inv[:, 2, 2] = 1/r**2
    g_inv[:, 3, 3] = 1/r**2
    
    # For Schwarzschild (vacuum): R_μν = 0, R = 0
    # This tests if the AI can find that in vacuum, the equation is just R_μν = 0
    R_ud = np.zeros((n_points, 4, 4))
    R = np.zeros(n_points)
    
    # Vacuum: T_μν = 0
    T = np.zeros((n_points, 4, 4))
    
    return TensorData(g=g, g_inv=g_inv, R_ud=R_ud, R=R, T=T)


def generate_flrw_data(n_points: int = 1000,
                       a_range: Tuple[float, float] = (0.1, 10),
                       k: float = 0,
                       rho_0: float = 1.0) -> TensorData:
    """
    Generate curvature data from FLRW cosmology.
    
    The FLRW metric:
    ds² = -dt² + a(t)²[dr²/(1-kr²) + r²dΩ²]
    
    For matter-dominated universe with ρ = ρ₀/a³:
    - Friedmann equation: (ȧ/a)² = 8πρ/3
    - G_μν = 8πT_μν with T_μν = diag(-ρ, p, p, p)
    """
    print(f"Generating FLRW cosmological curvature data...")
    
    # Sample scale factors
    a = np.random.uniform(*a_range, n_points)
    
    # Matter density (matter-dominated: ρ ∝ 1/a³)
    rho = rho_0 / a**3
    
    # Hubble parameter from Friedmann: H² = 8πρ/3 (for flat k=0)
    H_squared = 8 * np.pi * rho / 3
    H = np.sqrt(H_squared)
    
    # ä/a from acceleration equation: ä/a = -4π(ρ + 3p)/3
    # For dust (p=0): ä/a = -4πρ/3
    a_ddot_over_a = -4 * np.pi * rho / 3
    
    # Metric (comoving coordinates, flat space)
    g = np.zeros((n_points, 4, 4))
    g[:, 0, 0] = -1
    g[:, 1, 1] = a**2
    g[:, 2, 2] = a**2
    g[:, 3, 3] = a**2
    
    g_inv = np.zeros((n_points, 4, 4))
    g_inv[:, 0, 0] = -1
    g_inv[:, 1, 1] = 1/a**2
    g_inv[:, 2, 2] = 1/a**2
    g_inv[:, 3, 3] = 1/a**2
    
    # Ricci tensor for FLRW
    # R_00 = -3ä/a
    # R_ij = (aä + 2ȧ²)δ_ij = a²(ä/a + 2H²)δ_ij
    R_ud = np.zeros((n_points, 4, 4))
    R_ud[:, 0, 0] = -3 * a_ddot_over_a
    spatial_R = a**2 * (a_ddot_over_a + 2 * H_squared)
    R_ud[:, 1, 1] = spatial_R
    R_ud[:, 2, 2] = spatial_R
    R_ud[:, 3, 3] = spatial_R
    
    # Ricci scalar: R = g^μν R_μν = -R_00 + 3R_ii/a²
    R = -R_ud[:, 0, 0] * (-1) + 3 * spatial_R / a**2
    # Simplifies to: R = 6(ä/a + H²)
    
    # Stress-energy for dust (p=0)
    T = np.zeros((n_points, 4, 4))
    T[:, 0, 0] = rho  # T_00 = ρ (energy density)
    # T_ij = p g_ij = 0 for dust
    
    return TensorData(g=g, g_inv=g_inv, R_ud=R_ud, R=R, T=T)


def generate_mixed_data(n_points: int = 2000) -> TensorData:
    """
    Generate mixed curvature data from multiple spacetimes.
    
    Combines vacuum (Schwarzschild) and matter (FLRW) solutions
    to give the AI data where both G_μν = 0 and G_μν = 8πT_μν apply.
    """
    print("Generating mixed spacetime data...")
    
    # Half vacuum, half matter
    n_vac = n_points // 2
    n_mat = n_points - n_vac
    
    vac_data = generate_schwarzschild_data(n_vac)
    mat_data = generate_flrw_data(n_mat)
    
    # Concatenate
    return TensorData(
        g=np.concatenate([vac_data.g, mat_data.g]),
        g_inv=np.concatenate([vac_data.g_inv, mat_data.g_inv]),
        R_ud=np.concatenate([vac_data.R_ud, mat_data.R_ud]),
        R=np.concatenate([vac_data.R, mat_data.R]),
        T=np.concatenate([vac_data.T, mat_data.T]),
    )


def generate_perfect_fluid_data(n_points: int = 1000) -> TensorData:
    """
    Generate data where Einstein's equation is exactly satisfied.
    
    We construct arbitrary (g, R_μν, R) and then DEFINE T_μν
    such that G_μν = 8πT_μν.
    
    This is for testing that the regressor can find the relationship.
    """
    print("Generating perfect fluid curvature data (GR enforced)...")
    
    # Random metrics (perturbations of Minkowski)
    eta = np.diag([-1, 1, 1, 1])
    perturbations = np.random.uniform(-0.1, 0.1, (n_points, 4, 4))
    perturbations = (perturbations + perturbations.transpose(0, 2, 1)) / 2
    
    g = np.tile(eta, (n_points, 1, 1)) + perturbations
    g_inv = np.linalg.inv(g)
    
    # Random Ricci tensor (symmetric)
    R_ud = np.random.randn(n_points, 4, 4)
    R_ud = (R_ud + R_ud.transpose(0, 2, 1)) / 2
    
    # Ricci scalar
    R = np.einsum('nij,nij->n', g_inv, R_ud)
    
    # Compute Einstein tensor G_μν = R_μν - ½Rg_μν
    G = R_ud - 0.5 * R[:, None, None] * g
    
    # Define T_μν such that G_μν = 8πT_μν
    T = G / (8 * np.pi)
    
    return TensorData(g=g, g_inv=g_inv, R_ud=R_ud, R=R, T=T)


# ============================================
# DIRECT COUPLING DISCOVERY
# ============================================

def discover_coupling_constant(data: TensorData) -> Tuple[float, float]:
    """
    Directly discover the coupling constant κ in G_μν = κ T_μν.
    
    This is a simpler search: given that we know the structure is
    "geometry = constant × matter", find the constant.
    
    Returns:
        (kappa, residual) - the discovered coupling and fit quality
    """
    # Compute Einstein tensor G_μν = R_μν - 0.5 R g_μν
    G = data.R_ud - 0.5 * data.R[:, None, None] * data.g
    T = data.T
    
    # Flatten for regression
    G_flat = G.reshape(-1)
    T_flat = T.reshape(-1)
    
    # Remove near-zero entries (numerical noise)
    mask = np.abs(T_flat) > 1e-10
    if np.sum(mask) < 10:
        return 0.0, float('inf')
    
    G_masked = G_flat[mask]
    T_masked = T_flat[mask]
    
    # Least squares: G = κT => κ = (T·G)/(T·T)
    kappa = np.dot(T_masked, G_masked) / np.dot(T_masked, T_masked)
    
    # Residual
    residual = np.mean((G_masked - kappa * T_masked)**2)
    
    return kappa, residual


def discover_trace_reversal(data: TensorData) -> Tuple[float, float, float]:
    """
    Discover the trace reversal coefficient in R_μν = α R g_μν + κ T_μν.
    
    The Einstein equation has α = 0.5. This searches for that coefficient.
    
    Returns:
        (alpha, kappa, residual)
    """
    # We want to find α, κ such that:
    # R_μν - α R g_μν - κ T_μν = 0
    # 
    # Rewrite as: R_μν = α (R g_μν) + κ T_μν
    # This is linear regression with two regressors
    
    R_ud = data.R_ud.reshape(data.n_points, -1)  # (N, 16)
    Rg = (data.R[:, None, None] * data.g).reshape(data.n_points, -1)  # (N, 16)
    T = data.T.reshape(data.n_points, -1)  # (N, 16)
    
    # Stack regressors: X = [Rg, T], shape (N*16, 2)
    # Target: y = R_ud, shape (N*16,)
    y = R_ud.flatten()
    X = np.column_stack([Rg.flatten(), T.flatten()])
    
    # Remove near-zero rows
    row_norm = np.abs(X).sum(axis=1)
    mask = row_norm > 1e-10
    y_masked = y[mask]
    X_masked = X[mask]
    
    if len(y_masked) < 10:
        return 0.0, 0.0, float('inf')
    
    # Least squares: (X'X)^{-1} X'y
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(X_masked, y_masked, rcond=None)
        alpha, kappa = coeffs
        
        # Compute full residual
        pred = X_masked @ coeffs
        residual = np.mean((y_masked - pred)**2)
        
        return alpha, kappa, residual
    except:
        return 0.0, 0.0, float('inf')


# ============================================
# MAIN DISCOVERY FUNCTION
# ============================================

def discover_field_equations(data_type: str = 'perfect_fluid',
                            n_points: int = 2000) -> TensorRegressor:
    """
    Attempt to discover Einstein's field equations from curvature data.
    
    Uses multiple approaches:
    1. Direct coupling: Find κ in G_μν = κT_μν
    2. Trace reversal: Find α in R_μν = αRg_μν + κT_μν
    3. Full tensor regression: Search over all tensor combinations
    
    Args:
        data_type: 'schwarzschild', 'flrw', 'mixed', or 'perfect_fluid'
        n_points: Number of spacetime points to sample
    
    Returns:
        Trained TensorRegressor
    """
    print("\n" + "="*60)
    print("TENSOR FIELD EQUATION DISCOVERY")
    print("="*60)
    print(f"\nData type: {data_type}")
    print(f"Goal: Discover G_μν = 8πT_μν from curvature data")
    print(f"      (equivalently: R_μν - 0.5·Rg_μν - 8π·T_μν = 0)")
    
    # Generate data
    if data_type == 'schwarzschild':
        data = generate_schwarzschild_data(n_points)
        print("\nNote: Schwarzschild is vacuum (T=0), so expect G_μν = 0")
    elif data_type == 'flrw':
        data = generate_flrw_data(n_points)
        print("\nNote: FLRW has matter, expect G_μν = 8πT_μν")
    elif data_type == 'mixed':
        data = generate_mixed_data(n_points)
        print("\nNote: Mixed vacuum + matter data")
    else:  # perfect_fluid
        data = generate_perfect_fluid_data(n_points)
        print("\nNote: Data constructed to satisfy GR exactly")
    
    # ========================================
    # Method 1: Direct coupling discovery
    # ========================================
    print("\n" + "-"*60)
    print("METHOD 1: Direct Coupling Discovery")
    print("-"*60)
    print("  Searching for κ in: G_μν = κ T_μν")
    
    kappa, residual = discover_coupling_constant(data)
    print(f"\n  Discovered κ = {kappa:.6f}")
    print(f"  Expected κ = 8π = {8*np.pi:.6f}")
    print(f"  Error: {abs(kappa - 8*np.pi):.6f} ({100*abs(kappa - 8*np.pi)/(8*np.pi):.2f}%)")
    print(f"  Residual: {residual:.6e}")
    
    # ========================================
    # Method 2: Trace reversal discovery
    # ========================================
    print("\n" + "-"*60)
    print("METHOD 2: Trace Reversal Discovery")
    print("-"*60)
    print("  Searching for α, κ in: R_μν = α·Rg_μν + κ·T_μν")
    
    alpha, kappa2, residual2 = discover_trace_reversal(data)
    print(f"\n  Discovered α = {alpha:.6f}")
    print(f"  Expected α = 0.5")
    print(f"  Discovered κ = {kappa2:.6f}")
    print(f"  Expected κ = 8π = {8*np.pi:.6f}")
    print(f"  Residual: {residual2:.6e}")
    
    # ========================================
    # Method 3: Full tensor regression
    # ========================================
    print("\n" + "-"*60)
    print("METHOD 3: Genetic Tensor Regression")
    print("-"*60)
    
    regressor = TensorRegressor(
        population_size=2000,
        max_generations=100,
        allowed_tensors=['R_ud', 'R*g', 'T'],  # Removed 'g' to avoid trivial solutions
        parsimony_coef=0.01,  # Higher parsimony to prefer simpler equations
    )
    
    regressor.fit(data, verbose=True)
    
    # Results
    print("\n" + "="*60)
    print("FINAL DISCOVERED FIELD EQUATION:")
    print("="*60)
    print(f"\n  {regressor.get_equation()}")
    
    print("\n  Expected (Einstein): R_μν - 0.5·Rg_μν - 8π·T_μν = 0")
    print(f"  (where 8π ≈ {8*np.pi:.4f})")
    
    # Verify
    if regressor.best_equation:
        residual = regressor.best_equation.compute_residual(data)
        print(f"\n  Final residual: {residual:.6e}")
    
    return regressor


if __name__ == "__main__":
    import sys
    
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'perfect_fluid'
    discover_field_equations(data_type)

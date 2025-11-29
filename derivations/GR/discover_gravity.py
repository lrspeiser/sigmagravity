"""
Physics Discovery Engine - Real Data Version
=============================================

Discovers gravitational laws from ACTUAL observational data using
genetic programming / symbolic regression.

This is NOT circular - we feed in real measurements and let the AI
discover the equations that explain them.

Discoveries attempted:
1. Kepler's Third Law: T² ∝ a³
2. Newton's Law: F = GMm/r² (or equivalently a = GM/r²)
3. Galaxy rotation anomaly: g_obs vs g_baryon relationship

Usage:
    python discover_gravity.py [mode]
    
    Modes:
        kepler   - Discover Kepler's Third Law from planetary data
        newton   - Discover Newton's gravitational law
        galaxy   - Discover relationship in galaxy rotation curves
        all      - Run all discovery modes
"""

import numpy as np
import random
from time import time
from typing import Dict, List, Tuple, Optional
import argparse

from expression_tree import ExpressionTree, simplify_constants
from data_sources import (
    get_kepler_test_data,
    get_acceleration_data_for_discovery,
    get_galaxy_rotation_data,
    get_planetary_orbits,
    compute_orbital_observables,
    G, AU, M_SUN, YEAR
)


class SymbolicRegressor:
    """
    Genetic Programming engine for discovering equations from data.
    """
    
    def __init__(self, 
                 variable_names: List[str],
                 population_size: int = 2000,
                 max_generations: int = 100,
                 max_depth: int = 5,
                 tournament_size: int = 7,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 elitism: int = 10,
                 parsimony_coef: float = 0.001):
        """
        Args:
            variable_names: List of input variable names
            population_size: Number of candidate equations per generation
            max_generations: Maximum evolution iterations
            max_depth: Maximum expression tree depth
            tournament_size: Selection tournament size
            crossover_prob: Probability of crossover vs mutation
            mutation_prob: Probability of mutation
            elitism: Number of best individuals to preserve
            parsimony_coef: Penalty for equation complexity (Occam's razor)
        """
        self.variable_names = variable_names
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.parsimony_coef = parsimony_coef
        
        self.population: List[ExpressionTree] = []
        self.best_individual: Optional[ExpressionTree] = None
        self.best_fitness = float('inf')
        self.history = []
        
    def _initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            tree = ExpressionTree.random_tree(
                self.variable_names, 
                max_depth=self.max_depth,
                p_const=0.4
            )
            self.population.append(tree)
    
    def _compute_fitness(self, individual: ExpressionTree, 
                        X: Dict[str, np.ndarray], 
                        y: np.ndarray) -> float:
        """
        Compute fitness (lower is better).
        
        Fitness = MSE + parsimony_penalty
        """
        try:
            predictions = individual.evaluate(X)
            
            # Mean squared error
            residuals = predictions - y
            mse = np.mean(residuals**2)
            
            # Parsimony pressure (prefer simpler equations)
            complexity = individual.size()
            parsimony_penalty = self.parsimony_coef * complexity
            
            fitness = mse + parsimony_penalty
            
            # Penalize invalid results
            if not np.isfinite(fitness):
                fitness = 1e20
                
            return fitness
            
        except Exception:
            return 1e20
    
    def _tournament_select(self) -> ExpressionTree:
        """Select individual via tournament selection."""
        contestants = random.sample(self.population, self.tournament_size)
        return min(contestants, key=lambda x: x.fitness or float('inf'))
    
    def _evolve_generation(self, X: Dict[str, np.ndarray], y: np.ndarray):
        """Create next generation through selection, crossover, mutation."""
        
        # Evaluate fitness for all
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self._compute_fitness(ind, X, y)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness)
        
        # Track best
        if self.population[0].fitness < self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0].copy()
        
        # Elitism: keep best individuals
        new_population = [ind.copy() for ind in self.population[:self.elitism]]
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_prob:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = ExpressionTree.crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._tournament_select()
                child = parent.mutate()
            
            # Additional random mutation
            if random.random() < self.mutation_prob:
                child = child.mutate()
            
            # Limit depth
            if child.depth() <= self.max_depth + 2:
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, 
            verbose: bool = True) -> 'SymbolicRegressor':
        """
        Run symbolic regression to find equation fitting data.
        
        Args:
            X: Dictionary mapping variable names to data arrays
            y: Target values to predict
            verbose: Print progress
        
        Returns:
            self
        """
        if verbose:
            print(f"Starting symbolic regression...")
            print(f"  Variables: {list(X.keys())}")
            print(f"  Data points: {len(y)}")
            print(f"  Population: {self.population_size}")
            print(f"  Max generations: {self.max_generations}")
        
        self._initialize_population()
        
        start_time = time()
        
        for gen in range(self.max_generations):
            self._evolve_generation(X, y)
            
            self.history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'best_equation': self.best_individual.to_string() if self.best_individual else None,
                'mean_fitness': np.mean([ind.fitness for ind in self.population if ind.fitness]),
            })
            
            if verbose and gen % 10 == 0:
                simplified = simplify_constants(self.best_individual) if self.best_individual else None
                eq_str = simplified.to_string() if simplified else "None"
                print(f"  Gen {gen:3d}: Loss = {self.best_fitness:.6e}  |  {eq_str[:60]}")
            
            # Early stopping if very good fit
            if self.best_fitness < 1e-10:
                if verbose:
                    print(f"  Converged at generation {gen}!")
                break
        
        elapsed = time() - start_time
        if verbose:
            print(f"\nCompleted in {elapsed:.1f}s")
        
        return self
    
    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using best discovered equation."""
        if self.best_individual is None:
            raise ValueError("Must fit before predicting")
        return self.best_individual.evaluate(X)
    
    def get_equation(self, simplify: bool = True) -> str:
        """Get the discovered equation as string."""
        if self.best_individual is None:
            return "No equation discovered"
        if simplify:
            return simplify_constants(self.best_individual).to_string()
        return self.best_individual.to_string()


# ============================================
# DISCOVERY MODES
# ============================================

def discover_kepler():
    """
    Attempt to discover Kepler's Third Law: T² = a³
    
    Given: Semi-major axis (a) and orbital period (T)
    Goal: Find the relationship between them
    """
    print("\n" + "="*60)
    print("DISCOVERY MODE: Kepler's Third Law")
    print("="*60)
    print("\nObjective: Discover relationship between orbital period (T)")
    print("           and semi-major axis (a) for planets.")
    print("\nExpected: T² ∝ a³  or equivalently  T = a^1.5")
    
    # Get real planetary data
    data = get_kepler_test_data()
    
    print(f"\nInput data (8 planets):")
    print(f"  a (AU):    {data['a']}")
    print(f"  T (years): {data['T']}")
    
    # Setup: predict T from a
    X = {'a': data['a']}
    y = data['T']
    
    # Run symbolic regression
    regressor = SymbolicRegressor(
        variable_names=['a'],
        population_size=3000,
        max_generations=100,
        max_depth=4,
        parsimony_coef=0.01,  # Prefer simpler equations
    )
    
    regressor.fit(X, y, verbose=True)
    
    # Results
    print("\n" + "-"*40)
    print("DISCOVERED EQUATION:")
    print("-"*40)
    eq = regressor.get_equation()
    print(f"  T = {eq}")
    
    # Verify
    predictions = regressor.predict(X)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    print(f"\n  R² = {r2:.6f}")
    
    # Compare to known law
    print("\n  Comparison to Kepler's Law (T = a^1.5):")
    kepler_pred = data['a']**1.5
    for i, name in enumerate(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']):
        print(f"    {name:10s}: T_actual={data['T'][i]:.3f}, T_discovered={predictions[i]:.3f}, T_kepler={kepler_pred[i]:.3f}")
    
    return regressor


def discover_newton():
    """
    Attempt to discover Newton's gravitational law: a = k/r²
    
    Given: Distance from Sun (r), measured acceleration (a)
    Goal: Find the relationship
    """
    print("\n" + "="*60)
    print("DISCOVERY MODE: Newton's Law of Gravitation")
    print("="*60)
    print("\nObjective: Discover how gravitational acceleration depends")
    print("           on distance from the Sun.")
    print("\nExpected: a ∝ 1/r²  or equivalently  a = k * r^(-2)")
    
    # Get data
    data = get_acceleration_data_for_discovery()
    
    print(f"\nInput data (8 planets):")
    print(f"  r (AU):         {data['r']}")
    print(f"  a_norm (rel):   {data['a_normalized']}")
    
    # Setup: predict normalized acceleration from r
    X = {'r': data['r']}
    y = data['a_normalized']
    
    # Run symbolic regression
    regressor = SymbolicRegressor(
        variable_names=['r'],
        population_size=3000,
        max_generations=100,
        max_depth=4,
        parsimony_coef=0.01,
    )
    
    regressor.fit(X, y, verbose=True)
    
    # Results
    print("\n" + "-"*40)
    print("DISCOVERED EQUATION:")
    print("-"*40)
    eq = regressor.get_equation()
    print(f"  a = {eq}")
    
    # Verify
    predictions = regressor.predict(X)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    print(f"\n  R² = {r2:.6f}")
    
    # Compare to Newton
    print("\n  Comparison to Newton's Law (a ∝ 1/r²):")
    newton_pred = 1 / data['r']**2
    newton_pred_norm = newton_pred / newton_pred[2]  # Normalize to Earth
    for i, name in enumerate(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']):
        print(f"    {name:10s}: a_actual={y[i]:.4f}, a_discovered={predictions[i]:.4f}, a_newton={newton_pred_norm[i]:.4f}")
    
    return regressor


def discover_galaxy_rotation():
    """
    Attempt to discover the relationship between observed and baryonic
    acceleration in galaxy rotation curves.
    
    This probes the "dark matter problem" - the AI should find that
    g_obs ≠ g_baryon at low accelerations.
    
    Known relationship (MOND): g_obs = g_baryon / (1 - exp(-sqrt(g_baryon/a0)))
    Or equivalently: g_obs ≈ sqrt(g_baryon * a0) when g_baryon << a0
    """
    print("\n" + "="*60)
    print("DISCOVERY MODE: Galaxy Rotation Anomaly")
    print("="*60)
    print("\nObjective: Discover how observed acceleration (g_obs) relates")
    print("           to baryonic acceleration (g_baryon) in galaxies.")
    print("\nThis is the DARK MATTER / MODIFIED GRAVITY problem!")
    print("If Newton is correct: g_obs = g_baryon")
    print("Reality shows: g_obs > g_baryon at large radii")
    
    # Get data
    data = get_galaxy_rotation_data()
    
    print(f"\nInput data ({len(data['R'])} points from 3 galaxies):")
    print(f"  g_baryon range: {data['g_baryon'].min():.2e} to {data['g_baryon'].max():.2e} m/s²")
    print(f"  g_obs range:    {data['g_obs'].min():.2e} to {data['g_obs'].max():.2e} m/s²")
    print(f"  Ratio range:    {data['g_ratio'].min():.2f} to {data['g_ratio'].max():.2f}")
    
    # Normalize for numerical stability
    g_scale = 1e-10  # Typical galactic acceleration scale
    g_bar_norm = data['g_baryon'] / g_scale
    g_obs_norm = data['g_obs'] / g_scale
    
    # Setup: predict g_obs from g_baryon
    X = {'g_bar': g_bar_norm}
    y = g_obs_norm
    
    # Run symbolic regression
    regressor = SymbolicRegressor(
        variable_names=['g_bar'],
        population_size=3000,
        max_generations=150,
        max_depth=5,
        parsimony_coef=0.005,
    )
    
    regressor.fit(X, y, verbose=True)
    
    # Results
    print("\n" + "-"*40)
    print("DISCOVERED EQUATION:")
    print("-"*40)
    eq = regressor.get_equation()
    print(f"  g_obs = {eq}")
    print(f"  (where g values are in units of {g_scale:.0e} m/s²)")
    
    # Verify
    predictions = regressor.predict(X)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    print(f"\n  R² = {r2:.6f}")
    
    # Compare to simple Newton (g_obs = g_baryon)
    newton_residual = np.mean((g_obs_norm - g_bar_norm)**2)
    discovered_residual = np.mean((g_obs_norm - predictions)**2)
    print(f"\n  Newton residual (g_obs = g_baryon): {newton_residual:.4f}")
    print(f"  Discovered residual:                {discovered_residual:.4f}")
    print(f"  Improvement factor: {newton_residual/discovered_residual:.1f}x")
    
    return regressor


def main():
    parser = argparse.ArgumentParser(description='Physics Discovery Engine')
    parser.add_argument('mode', nargs='?', default='all',
                       choices=['kepler', 'newton', 'galaxy', 'all'],
                       help='Discovery mode to run')
    args = parser.parse_args()
    
    print("="*60)
    print("       PHYSICS DISCOVERY ENGINE")
    print("       Discovering Laws from Real Data")
    print("="*60)
    
    results = {}
    
    if args.mode in ['kepler', 'all']:
        results['kepler'] = discover_kepler()
    
    if args.mode in ['newton', 'all']:
        results['newton'] = discover_newton()
    
    if args.mode in ['galaxy', 'all']:
        results['galaxy'] = discover_galaxy_rotation()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF DISCOVERIES")
    print("="*60)
    
    for name, reg in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Equation: {reg.get_equation()}")
        print(f"  Fitness:  {reg.best_fitness:.6e}")


if __name__ == "__main__":
    main()

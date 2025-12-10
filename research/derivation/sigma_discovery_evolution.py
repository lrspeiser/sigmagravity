#!/usr/bin/env python3
"""
Σ-GRAVITY DISCOVERY ENGINE - Part 2: Evolution & Template Seeding
==================================================================

This module contains:
1. Main evolution engine with island model
2. Physics-informed template seeding
3. Exhaustive formula enumeration
4. Pareto front tracking
5. Result analysis and export
"""

import numpy as np
import random
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import itertools

# Import core components
from sigma_discovery_engine import (
    Expression, Op, OP_ARITY, PHYSICS_CONSTANTS,
    ExpressionGenerator, safe_eval_gpu, batch_evaluate_fitness,
    mutate_expr, crossover_expr, subtree_crossover,
    GPU_AVAILABLE, NUM_CORES, xp, cp
)


# =============================================================================
# PHYSICS-INFORMED TEMPLATE LIBRARY
# =============================================================================

class TemplateLibrary:
    """
    Pre-defined formula templates based on physics.
    
    These templates represent known or hypothesized functional forms
    that might describe gravitational enhancement.
    """
    
    @staticmethod
    def get_sigma_gravity_templates(var_names: List[str]) -> List[Expression]:
        """
        Templates specifically for Σ-Gravity enhancement factor K(R).
        
        The theory predicts K follows a coherence window function.
        """
        templates = []
        
        # Assume var_names[0] = R (radius), var_names[1] = g (log g_bar)
        R_idx = 0
        g_idx = 1 if len(var_names) > 1 else 0
        
        # ===================
        # POWER LAW TEMPLATES
        # ===================
        
        # K = c * R^α
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.CONST, alpha), (Op.POW, 0),
            ]))
        
        # K = c * sqrt(R)
        templates.append(Expression([
            (Op.VAR, R_idx), (Op.SQRT, 0),
        ]))
        
        # K = c / R
        templates.append(Expression([
            (Op.CONST, 1.0), (Op.VAR, R_idx), (Op.DIV, 0),
        ]))
        
        # ===================
        # WINDOW FUNCTION TEMPLATES (Core Σ-Gravity)
        # ===================
        
        # K = A * sqrt(R) / (1 + R/R_max)
        for R_max in [10, 15, 20, 25]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.SQRT, 0),                    # sqrt(R)
                (Op.CONST, 1.0),                                   # 1
                (Op.VAR, R_idx), (Op.CONST, R_max), (Op.DIV, 0),  # R/R_max
                (Op.ADD, 0),                                       # 1 + R/R_max
                (Op.DIV, 0),                                       # sqrt(R) / (1 + R/R_max)
            ]))
        
        # K = A * R^α / (1 + (R/R0)^β)
        for alpha in [0.5, 1.0]:
            for R0 in [5, 10, 15]:
                templates.append(Expression([
                    (Op.VAR, R_idx), (Op.CONST, alpha), (Op.POW, 0),  # R^α
                    (Op.CONST, 1.0),                                   # 1
                    (Op.VAR, R_idx), (Op.CONST, R0), (Op.DIV, 0),     # R/R0
                    (Op.ADD, 0),                                       # 1 + R/R0
                    (Op.DIV, 0),                                       # R^α / (1 + R/R0)
                ]))
        
        # Burr-XII type: K = A * (R/R0)^α / (1 + (R/R0)^β)^((α+γ)/β)
        # Simplified: K = A * x^α / (1 + x)^(α+1) where x = R/R0
        for R0 in [3, 5, 8]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.CONST, R0), (Op.DIV, 0),         # x = R/R0
                (Op.SQRT, 0),                                          # x^0.5
                (Op.CONST, 1.0),
                (Op.VAR, R_idx), (Op.CONST, R0), (Op.DIV, 0),         # x = R/R0
                (Op.ADD, 0),                                           # 1 + x
                (Op.CONST, 1.5), (Op.POW, 0),                         # (1 + x)^1.5
                (Op.DIV, 0),                                           # x^0.5 / (1 + x)^1.5
            ]))
        
        # ===================
        # MOND-LIKE TEMPLATES
        # ===================
        
        # K = sqrt(a0/g_bar) (MOND deep regime)
        templates.append(Expression([
            (Op.CONST, -10.0),  # log10(a0) ≈ -10
            (Op.VAR, g_idx),    # log10(g_bar)
            (Op.SUB, 0),        # log10(a0) - log10(g_bar) = log10(a0/g_bar)
            (Op.CONST, 0.5), (Op.MUL, 0),  # 0.5 * log10(a0/g_bar)
            (Op.CONST, 10.0), (Op.POW, 0),  # 10^(...) = sqrt(a0/g_bar) in linear
        ]))
        
        # K = 1 + sqrt(a0/g_bar) - 1 = sqrt(a0/g_bar) (simplified)
        for a0_log in [-10, -10.1, -9.9]:
            templates.append(Expression([
                (Op.CONST, a0_log), (Op.VAR, g_idx), (Op.SUB, 0),  # log ratio
                (Op.CONST, 0.5), (Op.MUL, 0),  # half for sqrt
            ]))
        
        # ===================
        # LOGARITHMIC TEMPLATES
        # ===================
        
        # K = c * log(R/R0)
        for R0 in [1, 3, 5]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.CONST, R0), (Op.DIV, 0),
                (Op.LOG, 0),
            ]))
        
        # K = c * log(R) * f(R)
        templates.append(Expression([
            (Op.VAR, R_idx), (Op.LOG, 0),
            (Op.CONST, 1.0), (Op.VAR, R_idx), (Op.CONST, 20.0), (Op.DIV, 0), (Op.ADD, 0),
            (Op.INV, 0),
            (Op.MUL, 0),  # log(R) / (1 + R/20)
        ]))
        
        # ===================
        # EXPONENTIAL CUTOFF TEMPLATES
        # ===================
        
        # K = A * R^α * exp(-R/R_max)
        for R_max in [15, 20, 25]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.SQRT, 0),                      # sqrt(R)
                (Op.VAR, R_idx), (Op.CONST, R_max), (Op.DIV, 0),   # R/R_max
                (Op.NEG, 0),                                        # -R/R_max
                (Op.EXP, 0),                                        # exp(-R/R_max)
                (Op.MUL, 0),                                        # sqrt(R) * exp(-R/R_max)
            ]))
        
        # ===================
        # TANH TRANSITION TEMPLATES
        # ===================
        
        # K = A * tanh(R/R0)
        for R0 in [5, 10, 15]:
            templates.append(Expression([
                (Op.VAR, R_idx), (Op.CONST, R0), (Op.DIV, 0),
                (Op.TANH, 0),
            ]))
        
        # K = A * R * (1 - tanh((R-R0)/w))
        templates.append(Expression([
            (Op.VAR, R_idx),                                        # R
            (Op.CONST, 1.0),
            (Op.VAR, R_idx), (Op.CONST, 15.0), (Op.SUB, 0),        # R - 15
            (Op.CONST, 5.0), (Op.DIV, 0),                          # (R-15)/5
            (Op.TANH, 0),                                           # tanh((R-15)/5)
            (Op.SUB, 0),                                            # 1 - tanh(...)
            (Op.MUL, 0),                                            # R * (1 - tanh(...))
        ]))
        
        return templates
    
    @staticmethod
    def get_rar_templates(var_names: List[str]) -> List[Expression]:
        """Templates for Radial Acceleration Relation"""
        templates = []
        
        # g_idx should be log10(g_bar)
        g_idx = 0
        R_idx = 1 if len(var_names) > 1 else 0
        
        # Linear: g_obs = g_bar (no dark matter baseline)
        templates.append(Expression([
            (Op.VAR, g_idx),
        ]))
        
        # MOND simple: g_obs = g_bar + const
        templates.append(Expression([
            (Op.VAR, g_idx), (Op.CONST, 0.3), (Op.ADD, 0),
        ]))
        
        # MOND interpolating function forms
        # In log space: log(g_obs) = log(g_bar) + log(ν)
        # where ν = 1/(1-exp(-sqrt(g_bar/a0)))
        
        # Simplified: log(g_obs) ≈ g + f(g)
        for offset in [0.1, 0.2, 0.3, 0.5]:
            templates.append(Expression([
                (Op.VAR, g_idx),
                (Op.CONST, -10.0), (Op.VAR, g_idx), (Op.SUB, 0),  # a0/g
                (Op.SQRT, 0),
                (Op.CONST, offset), (Op.MUL, 0),
                (Op.ADD, 0),
            ]))
        
        return templates
    
    @staticmethod
    def get_btf_templates(var_names: List[str]) -> List[Expression]:
        """Templates for Baryonic Tully-Fisher relation"""
        templates = []
        
        # M_idx should be log10(M_baryon)
        M_idx = 0
        
        # BTF: log(v) = 0.25 * log(M) + const
        for const in [0.3, 0.4, 0.5, 0.6]:
            templates.append(Expression([
                (Op.CONST, 0.25), (Op.VAR, M_idx), (Op.MUL, 0),
                (Op.CONST, const), (Op.ADD, 0),
            ]))
        
        # Pure power law: v ∝ M^α
        for alpha in [0.2, 0.25, 0.3]:
            templates.append(Expression([
                (Op.CONST, alpha), (Op.VAR, M_idx), (Op.MUL, 0),
            ]))
        
        return templates


# =============================================================================
# EXHAUSTIVE FORMULA ENUMERATION
# =============================================================================

class FormulaEnumerator:
    """
    Systematically enumerate all possible formulas up to a given complexity.
    
    This ensures we don't miss any simple formulas that might fit the data.
    """
    
    def __init__(self, num_vars: int, var_names: List[str]):
        self.num_vars = num_vars
        self.var_names = var_names
        
        # Operations to include
        self.binary_ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.POW]
        self.unary_ops = [Op.SQRT, Op.SQUARE, Op.INV, Op.LOG, Op.EXP, Op.TANH]
        
        # Key constants
        self.constants = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, np.pi]
    
    def enumerate_depth(self, max_depth: int) -> List[Expression]:
        """Enumerate all expressions up to given depth"""
        
        if max_depth == 1:
            # Terminals only
            exprs = []
            for i in range(self.num_vars):
                exprs.append(Expression([(Op.VAR, float(i))]))
            for c in self.constants:
                exprs.append(Expression([(Op.CONST, c)]))
            return exprs
        
        # Get expressions of smaller depth
        smaller = self.enumerate_depth(max_depth - 1)
        
        # New expressions at this depth
        new_exprs = []
        
        # Apply unary operations
        for expr in smaller:
            for op in self.unary_ops:
                new_instr = list(expr.instructions) + [(op, 0.0)]
                new_exprs.append(Expression(new_instr))
        
        # Apply binary operations (combine pairs)
        # Limit combinations to avoid explosion
        if len(smaller) > 100:
            smaller_sample = random.sample(smaller, 100)
        else:
            smaller_sample = smaller
        
        for e1 in smaller_sample:
            for e2 in smaller_sample:
                for op in self.binary_ops:
                    new_instr = list(e1.instructions) + list(e2.instructions) + [(op, 0.0)]
                    new_exprs.append(Expression(new_instr))
        
        return smaller + new_exprs
    
    def enumerate_targeted(self, max_complexity: int = 15) -> List[Expression]:
        """
        Targeted enumeration for Σ-Gravity formulas.
        
        Focus on forms likely to appear in gravitational physics:
        - x^α forms
        - x/(1+x) forms  
        - sqrt(x) forms
        - log(x) forms
        """
        exprs = []
        
        R_idx = 0
        g_idx = 1 if self.num_vars > 1 else 0
        
        # Single variable forms
        for var_idx in range(self.num_vars):
            # x
            exprs.append(Expression([(Op.VAR, var_idx)]))
            
            # sqrt(x)
            exprs.append(Expression([(Op.VAR, var_idx), (Op.SQRT, 0)]))
            
            # x^2
            exprs.append(Expression([(Op.VAR, var_idx), (Op.SQUARE, 0)]))
            
            # 1/x
            exprs.append(Expression([(Op.VAR, var_idx), (Op.INV, 0)]))
            
            # log(x)
            exprs.append(Expression([(Op.VAR, var_idx), (Op.LOG, 0)]))
            
            # x^0.25
            exprs.append(Expression([
                (Op.VAR, var_idx), (Op.SQRT, 0), (Op.SQRT, 0)
            ]))
            
            # Power laws with various exponents
            for alpha in [0.25, 0.33, 0.5, 0.67, 0.75, 1.5, 2]:
                exprs.append(Expression([
                    (Op.VAR, var_idx), (Op.CONST, alpha), (Op.POW, 0)
                ]))
        
        # Two-variable forms (if we have 2+ variables)
        if self.num_vars >= 2:
            for i, j in itertools.permutations(range(self.num_vars), 2):
                # x/y
                exprs.append(Expression([
                    (Op.VAR, i), (Op.VAR, j), (Op.DIV, 0)
                ]))
                
                # x*y
                exprs.append(Expression([
                    (Op.VAR, i), (Op.VAR, j), (Op.MUL, 0)
                ]))
                
                # sqrt(x/y)
                exprs.append(Expression([
                    (Op.VAR, i), (Op.VAR, j), (Op.DIV, 0), (Op.SQRT, 0)
                ]))
                
                # sqrt(x*y)
                exprs.append(Expression([
                    (Op.VAR, i), (Op.VAR, j), (Op.MUL, 0), (Op.SQRT, 0)
                ]))
        
        # Window function forms: x / (1 + x/c)
        for c in [5, 10, 15, 20, 25]:
            exprs.append(Expression([
                (Op.VAR, R_idx),
                (Op.CONST, 1.0),
                (Op.VAR, R_idx), (Op.CONST, c), (Op.DIV, 0),
                (Op.ADD, 0),
                (Op.DIV, 0),
            ]))
            
            # sqrt(x) / (1 + x/c)
            exprs.append(Expression([
                (Op.VAR, R_idx), (Op.SQRT, 0),
                (Op.CONST, 1.0),
                (Op.VAR, R_idx), (Op.CONST, c), (Op.DIV, 0),
                (Op.ADD, 0),
                (Op.DIV, 0),
            ]))
        
        # Exponential cutoff: x * exp(-x/c)
        for c in [10, 15, 20, 25]:
            exprs.append(Expression([
                (Op.VAR, R_idx),
                (Op.VAR, R_idx), (Op.CONST, c), (Op.DIV, 0),
                (Op.NEG, 0), (Op.EXP, 0),
                (Op.MUL, 0),
            ]))
        
        return exprs


# =============================================================================
# MAIN EVOLUTION ENGINE
# =============================================================================

@dataclass
class DiscoveryResult:
    """Stores a discovered formula and its metrics"""
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


class SigmaGravityDiscoveryEngine:
    """
    Main engine for discovering Σ-Gravity formulas.
    
    Features:
    - GPU-accelerated fitness evaluation
    - Island model with migration
    - Physics-informed template seeding
    - Exhaustive enumeration for simple formulas
    - Pareto front tracking (accuracy vs simplicity)
    - Multiple restart strategy
    """
    
    def __init__(
        self,
        var_names: List[str],
        population_size: int = 3000,
        max_generations: int = 150,
        num_islands: int = 6,
        migration_rate: float = 0.1,
        migration_interval: int = 15,
        tournament_size: int = 7,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.15,
        elitism: int = 15,
        complexity_weight: float = 0.005,
        max_depth: int = 6,
        physics_mode: str = 'gravity',
        use_templates: bool = True,
        exhaustive_depth: int = 3,
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
        self.physics_mode = physics_mode
        self.use_templates = use_templates
        self.exhaustive_depth = exhaustive_depth
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if GPU_AVAILABLE:
                cp.random.seed(seed)
        
        # Expression generator
        self.generator = ExpressionGenerator(
            num_vars=self.num_vars,
            var_names=var_names,
            max_depth=max_depth,
            physics_mode=physics_mode
        )
        
        # Template library
        self.templates = TemplateLibrary()
        
        # Enumerator for exhaustive search
        self.enumerator = FormulaEnumerator(self.num_vars, var_names)
        
        # Results tracking
        self.islands: List[List[Expression]] = []
        self.best_overall: Optional[DiscoveryResult] = None
        self.pareto_front: List[DiscoveryResult] = []
        self.history: List[dict] = []
        self.all_evaluated: Dict[str, DiscoveryResult] = {}  # Cache
    
    def _init_population(self, island_size: int, target: str = 'K') -> List[Expression]:
        """Initialize population with templates and random expressions"""
        pop = []
        
        # Add physics templates
        if self.use_templates:
            if target == 'K':
                templates = self.templates.get_sigma_gravity_templates(self.var_names)
            elif target == 'rar':
                templates = self.templates.get_rar_templates(self.var_names)
            elif target == 'btf':
                templates = self.templates.get_btf_templates(self.var_names)
            else:
                templates = self.templates.get_sigma_gravity_templates(self.var_names)
            
            pop.extend(templates)
            print(f"    Added {len(templates)} physics templates")
        
        # Add exhaustive enumeration
        if self.exhaustive_depth > 0:
            enumerated = self.enumerator.enumerate_targeted()
            pop.extend(enumerated)
            print(f"    Added {len(enumerated)} enumerated formulas")
        
        # Fill rest with random
        while len(pop) < island_size:
            pop.append(self.generator.random_expr())
        
        return pop[:island_size]
    
    def _evaluate_population(
        self,
        pop: List[Expression],
        X: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> List[Tuple[float, float, int]]:
        """Evaluate fitness of population"""
        return batch_evaluate_fitness(
            pop, X, y,
            complexity_weight=self.complexity_weight,
            use_gpu=GPU_AVAILABLE
        )
    
    def _compute_score(self, mse: float, r2: float, complexity: int, y_scale: float) -> float:
        """Compute combined fitness score (lower is better)"""
        normalized_mse = mse / (y_scale**2 + 1e-10)
        return np.log(normalized_mse + 1e-10) + self.complexity_weight * complexity
    
    def _tournament_select(
        self,
        pop: List[Expression],
        scores: List[float]
    ) -> int:
        """Tournament selection"""
        tournament = random.sample(range(len(pop)), min(self.tournament_size, len(pop)))
        return min(tournament, key=lambda i: scores[i])
    
    def _update_pareto_front(self, result: DiscoveryResult):
        """Update Pareto front with new result"""
        dominated = []
        is_dominated = False
        
        for i, existing in enumerate(self.pareto_front):
            # Check if existing dominates new
            if (existing.mse <= result.mse and existing.complexity <= result.complexity and
                (existing.mse < result.mse or existing.complexity < result.complexity)):
                is_dominated = True
                break
            
            # Check if new dominates existing
            if (result.mse <= existing.mse and result.complexity <= existing.complexity and
                (result.mse < existing.mse or result.complexity < existing.complexity)):
                dominated.append(i)
        
        if not is_dominated:
            # Remove dominated solutions
            self.pareto_front = [r for i, r in enumerate(self.pareto_front) if i not in dominated]
            self.pareto_front.append(result)
            
            # Keep front manageable
            if len(self.pareto_front) > 100:
                self.pareto_front.sort(key=lambda x: x.mse)
                self.pareto_front = self.pareto_front[:100]
    
    def evolve(
        self,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        target: str = 'K',
        verbose: bool = True
    ) -> Tuple[Expression, float, float]:
        """
        Run evolutionary search.
        
        Returns: (best_expression, mse, r_squared)
        """
        start_time = time.time()
        
        y_scale = float(np.std(y))
        y_mean = float(np.mean(y))
        
        if verbose:
            print(f"  Target statistics: mean={y_mean:.4f}, std={y_scale:.4f}")
        
        # Initialize islands
        island_size = self.population_size // self.num_islands
        self.islands = []
        
        if verbose:
            print(f"  Initializing {self.num_islands} islands × {island_size} individuals...")
        
        for i in range(self.num_islands):
            island = self._init_population(island_size, target)
            self.islands.append(island)
        
        # Main evolution loop
        for generation in range(self.max_generations):
            gen_start = time.time()
            
            gen_best_result = None
            gen_best_score = float('inf')
            
            # Evolve each island
            for island_idx, island in enumerate(self.islands):
                # Evaluate
                fitness_results = self._evaluate_population(island, X, y)
                
                # Compute scores
                scores = []
                for (mse, r2, compl), expr in zip(fitness_results, island):
                    score = self._compute_score(mse, r2, compl, y_scale)
                    scores.append(score)
                    
                    # Track result
                    formula_str = expr.to_string(self.var_names)
                    if formula_str not in self.all_evaluated:
                        result = DiscoveryResult(
                            expression=expr.copy(),
                            formula_string=formula_str,
                            mse=mse,
                            r_squared=r2,
                            complexity=compl,
                            generation=generation
                        )
                        self.all_evaluated[formula_str] = result
                        
                        # Update Pareto front
                        if r2 > 0.5:
                            self._update_pareto_front(result)
                    
                    # Track best
                    if score < gen_best_score:
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
                
                # Elitism
                new_island = [island[i].copy() for i in sorted_indices[:self.elitism]]
                
                # Generate offspring
                while len(new_island) < island_size:
                    p1_idx = self._tournament_select(island, scores)
                    
                    if random.random() < self.crossover_rate:
                        p2_idx = self._tournament_select(island, scores)
                        
                        if random.random() < 0.5:
                            c1, c2 = crossover_expr(island[p1_idx], island[p2_idx])
                        else:
                            c1, c2 = subtree_crossover(island[p1_idx], island[p2_idx])
                        
                        c1 = mutate_expr(c1, self.generator, self.mutation_rate * 0.5)
                        c2 = mutate_expr(c2, self.generator, self.mutation_rate * 0.5)
                        new_island.extend([c1, c2])
                    else:
                        child = mutate_expr(island[p1_idx], self.generator, self.mutation_rate)
                        new_island.append(child)
                
                self.islands[island_idx] = new_island[:island_size]
            
            # Migration between islands
            if generation > 0 and generation % self.migration_interval == 0:
                num_migrants = max(1, int(island_size * self.migration_rate))
                for i in range(self.num_islands):
                    source = (i + 1) % self.num_islands
                    migrants = self.islands[source][:num_migrants]
                    self.islands[i] = migrants + self.islands[i][:-num_migrants]
            
            # Update overall best
            if self.best_overall is None or gen_best_result.mse < self.best_overall.mse:
                self.best_overall = gen_best_result
            
            gen_time = time.time() - gen_start
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_r2': gen_best_result.r_squared,
                'best_mse': gen_best_result.mse,
                'best_complexity': gen_best_result.complexity,
                'pareto_size': len(self.pareto_front),
                'unique_evaluated': len(self.all_evaluated),
            })
            
            # Progress output
            if verbose and (generation % 10 == 0 or generation == self.max_generations - 1):
                formula = gen_best_result.formula_string[:45]
                print(f"  Gen {generation:3d} | R²={gen_best_result.r_squared:.5f} | "
                      f"MSE={gen_best_result.mse:.2e} | Pareto={len(self.pareto_front):2d} | "
                      f"{gen_time:.2f}s | {formula}")
            
            # Early stopping
            if gen_best_result.r_squared > 0.9999:
                if verbose:
                    print(f"  ✓ Converged at generation {generation}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Unique formulas evaluated: {len(self.all_evaluated):,}")
        
        return (
            self.best_overall.expression,
            self.best_overall.mse,
            self.best_overall.r_squared
        )
    
    def get_pareto_front(self) -> List[DiscoveryResult]:
        """Get sorted Pareto front"""
        return sorted(self.pareto_front, key=lambda x: -x.r_squared)
    
    def get_top_formulas(self, n: int = 20) -> List[DiscoveryResult]:
        """Get top n formulas by R²"""
        all_results = list(self.all_evaluated.values())
        all_results.sort(key=lambda x: -x.r_squared)
        return all_results[:n]
    
    def export_results(self, filepath: str):
        """Export results to JSON"""
        results = {
            'best': self.best_overall.to_dict() if self.best_overall else None,
            'pareto_front': [r.to_dict() for r in self.get_pareto_front()],
            'top_formulas': [r.to_dict() for r in self.get_top_formulas(50)],
            'history': self.history,
            'config': {
                'variables': self.var_names,
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'num_islands': self.num_islands,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Results exported to {filepath}")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'TemplateLibrary',
    'FormulaEnumerator',
    'DiscoveryResult',
    'SigmaGravityDiscoveryEngine',
]


if __name__ == "__main__":
    print("\nΣ-Gravity Discovery Engine - Evolution Module")
    print("Run run_sigma_discovery.py for complete analysis")

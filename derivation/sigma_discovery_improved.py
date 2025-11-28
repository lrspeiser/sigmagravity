#!/usr/bin/env python3
"""
Σ-GRAVITY IMPROVED DISCOVERY ENGINE
====================================

Uses niched genetic programming for diversity preservation.
Prevents premature convergence to local optima.

Usage:
    python sigma_discovery_improved.py --stars 50000 --generations 50
"""

import numpy as np
from scipy.optimize import minimize
import time
import json
import argparse
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')


# ============== EXPRESSION TREE ==============

@dataclass
class Node:
    """Node in expression tree."""
    op: str
    children: List['Node'] = field(default_factory=list)
    value: float = 0.0  # For constants
    
    def copy(self):
        return Node(op=self.op, children=[c.copy() for c in self.children], value=self.value)
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)
    
    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)
    
    def to_string(self) -> str:
        if self.op == 'R':
            return 'R'
        elif self.op == 'g':
            return 'g'
        elif self.op == 'const':
            return f'{self.value:.3g}'
        elif self.op == 'add':
            return f'({self.children[0].to_string()} + {self.children[1].to_string()})'
        elif self.op == 'sub':
            return f'({self.children[0].to_string()} - {self.children[1].to_string()})'
        elif self.op == 'mul':
            return f'({self.children[0].to_string()} × {self.children[1].to_string()})'
        elif self.op == 'div':
            return f'({self.children[0].to_string()} / {self.children[1].to_string()})'
        elif self.op == 'sqrt':
            return f'√{self.children[0].to_string()}'
        elif self.op == 'pow':
            return f'{self.children[0].to_string()}^{self.children[1].to_string()}'
        elif self.op == 'exp':
            return f'exp({self.children[0].to_string()})'
        elif self.op == 'log':
            return f'log({self.children[0].to_string()})'
        elif self.op == 'tanh':
            return f'tanh({self.children[0].to_string()})'
        elif self.op == 'abs':
            return f'|{self.children[0].to_string()}|'
        return f'{self.op}(?)'
    
    def get_structure(self) -> str:
        """Get structure signature for niching."""
        if self.op in ['R', 'g', 'const']:
            return self.op
        elif len(self.children) == 1:
            return f'{self.op}({self.children[0].get_structure()})'
        else:
            sigs = sorted(c.get_structure() for c in self.children)
            return f'{self.op}({",".join(sigs)})'


BINARY_OPS = ['add', 'sub', 'mul', 'div']
UNARY_OPS = ['sqrt', 'tanh', 'log', 'abs']
TERMINALS = ['R', 'g', 'const']


def random_tree(max_depth: int = 4, current_depth: int = 0) -> Node:
    """Generate random expression tree."""
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
        # Terminal
        op = random.choice(TERMINALS)
        if op == 'const':
            return Node(op='const', value=random.uniform(0.01, 10))
        return Node(op=op)
    
    if random.random() < 0.7:
        # Binary
        op = random.choice(BINARY_OPS)
        return Node(op=op, children=[
            random_tree(max_depth, current_depth + 1),
            random_tree(max_depth, current_depth + 1)
        ])
    else:
        # Unary
        op = random.choice(UNARY_OPS)
        return Node(op=op, children=[random_tree(max_depth, current_depth + 1)])


def evaluate_tree(node: Node, R: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Evaluate expression tree."""
    eps = 1e-10
    
    if node.op == 'R':
        return R.copy()
    elif node.op == 'g':
        return g.copy()
    elif node.op == 'const':
        return np.full_like(R, node.value)
    elif node.op == 'add':
        return evaluate_tree(node.children[0], R, g) + evaluate_tree(node.children[1], R, g)
    elif node.op == 'sub':
        return evaluate_tree(node.children[0], R, g) - evaluate_tree(node.children[1], R, g)
    elif node.op == 'mul':
        return evaluate_tree(node.children[0], R, g) * evaluate_tree(node.children[1], R, g)
    elif node.op == 'div':
        denom = evaluate_tree(node.children[1], R, g)
        return evaluate_tree(node.children[0], R, g) / np.maximum(np.abs(denom), eps)
    elif node.op == 'sqrt':
        return np.sqrt(np.maximum(evaluate_tree(node.children[0], R, g), eps))
    elif node.op == 'pow':
        base = np.maximum(evaluate_tree(node.children[0], R, g), eps)
        exp = np.clip(evaluate_tree(node.children[1], R, g), -3, 3)
        return np.power(base, exp)
    elif node.op == 'exp':
        return np.exp(np.clip(evaluate_tree(node.children[0], R, g), -20, 20))
    elif node.op == 'log':
        return np.log(np.maximum(evaluate_tree(node.children[0], R, g), eps))
    elif node.op == 'tanh':
        return np.tanh(evaluate_tree(node.children[0], R, g))
    elif node.op == 'abs':
        return np.abs(evaluate_tree(node.children[0], R, g))
    return np.zeros_like(R)


def get_constants(node: Node) -> List[Node]:
    """Get all constant nodes."""
    consts = []
    if node.op == 'const':
        consts.append(node)
    for c in node.children:
        consts.extend(get_constants(c))
    return consts


def optimize_constants(tree: Node, R: np.ndarray, g: np.ndarray, K: np.ndarray) -> float:
    """Optimize constants in tree. Returns MSE."""
    consts = get_constants(tree)
    if not consts:
        pred = evaluate_tree(tree, R, g)
        valid = np.isfinite(pred)
        if valid.sum() < len(K) * 0.5:
            return 1e30
        return float(np.mean((K[valid] - pred[valid])**2))
    
    def objective(params):
        for i, c in enumerate(consts):
            c.value = params[i]
        pred = evaluate_tree(tree, R, g)
        valid = np.isfinite(pred)
        if valid.sum() < len(K) * 0.5:
            return 1e30
        return float(np.mean((K[valid] - pred[valid])**2))
    
    x0 = [c.value for c in consts]
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100})
        for i, c in enumerate(consts):
            c.value = result.x[i]
        return result.fun
    except:
        return objective(x0)


# ============== GENETIC OPERATORS ==============

def mutate(tree: Node, prob: float = 0.3) -> Node:
    """Mutate tree."""
    tree = tree.copy()
    
    def mutate_node(node: Node, depth: int = 0):
        if random.random() < prob:
            if node.op == 'const':
                node.value *= random.uniform(0.5, 2.0)
            elif node.op in TERMINALS:
                node.op = random.choice(TERMINALS)
                if node.op == 'const':
                    node.value = random.uniform(0.01, 10)
            elif node.op in UNARY_OPS:
                node.op = random.choice(UNARY_OPS)
            elif node.op in BINARY_OPS:
                if random.random() < 0.5:
                    node.op = random.choice(BINARY_OPS)
                else:
                    # Subtree replacement
                    new_tree = random_tree(max_depth=3)
                    node.op = new_tree.op
                    node.children = new_tree.children
                    node.value = new_tree.value
                    return
        
        for c in node.children:
            mutate_node(c, depth + 1)
    
    mutate_node(tree)
    return tree


def crossover(tree1: Node, tree2: Node) -> Tuple[Node, Node]:
    """Crossover two trees."""
    tree1, tree2 = tree1.copy(), tree2.copy()
    
    def get_subtrees(node: Node, depth: int = 0) -> List[Tuple[Node, Node, int]]:
        """Get (parent, child, index) tuples."""
        subtrees = []
        for i, c in enumerate(node.children):
            subtrees.append((node, c, i))
            subtrees.extend(get_subtrees(c, depth + 1))
        return subtrees
    
    subs1 = get_subtrees(tree1)
    subs2 = get_subtrees(tree2)
    
    if subs1 and subs2:
        p1, c1, i1 = random.choice(subs1)
        p2, c2, i2 = random.choice(subs2)
        p1.children[i1], p2.children[i2] = c2.copy(), c1.copy()
    
    return tree1, tree2


# ============== NICHED EVOLUTION ==============

@dataclass
class Individual:
    tree: Node
    mse: float = 1e30
    r2: float = -1.0
    structure: str = ""
    
    def __post_init__(self):
        self.structure = self.tree.get_structure()


class NichedDiscoveryEngine:
    """Discovery engine with structure-based niching."""
    
    def __init__(self, pop_size: int = 200, n_niches: int = 20, max_depth: int = 5):
        self.pop_size = pop_size
        self.n_niches = n_niches
        self.max_depth = max_depth
        self.population: List[Individual] = []
        self.best_by_niche: Dict[str, Individual] = {}
        
    def initialize(self, R: np.ndarray, g: np.ndarray, K: np.ndarray):
        """Initialize population with random trees and window templates."""
        self.R, self.g, self.K = R, g, K
        
        # Seed with window function templates
        templates = self._create_template_trees()
        
        # Random trees
        random_trees = [random_tree(max_depth=self.max_depth) for _ in range(self.pop_size - len(templates))]
        
        all_trees = templates + random_trees
        self.population = []
        
        for tree in all_trees:
            mse = optimize_constants(tree, R, g, K)
            r2 = self._compute_r2(tree)
            self.population.append(Individual(tree=tree, mse=mse, r2=r2))
        
        self._update_niches()
    
    def _create_template_trees(self) -> List[Node]:
        """Create window function template trees."""
        templates = []
        
        # sqrt(R)
        templates.append(Node(op='mul', children=[
            Node(op='const', value=1.0),
            Node(op='sqrt', children=[Node(op='R')])
        ]))
        
        # R^alpha
        templates.append(Node(op='mul', children=[
            Node(op='const', value=0.1),
            Node(op='pow', children=[Node(op='R'), Node(op='const', value=1.5)])
        ]))
        
        # sqrt(R) / (1 + R/R0)
        templates.append(Node(op='div', children=[
            Node(op='mul', children=[
                Node(op='const', value=2.0),
                Node(op='sqrt', children=[Node(op='R')])
            ]),
            Node(op='add', children=[
                Node(op='const', value=1.0),
                Node(op='div', children=[Node(op='R'), Node(op='const', value=15.0)])
            ])
        ]))
        
        # tanh((R - R1) / w) + c
        templates.append(Node(op='add', children=[
            Node(op='mul', children=[
                Node(op='const', value=1.0),
                Node(op='tanh', children=[
                    Node(op='div', children=[
                        Node(op='sub', children=[Node(op='R'), Node(op='const', value=7.0)]),
                        Node(op='const', value=2.0)
                    ])
                ])
            ]),
            Node(op='const', value=1.0)
        ]))
        
        # R / (c + R)
        templates.append(Node(op='div', children=[
            Node(op='mul', children=[Node(op='const', value=3.0), Node(op='R')]),
            Node(op='add', children=[Node(op='const', value=5.0), Node(op='R')])
        ]))
        
        # Linear baseline
        templates.append(Node(op='add', children=[
            Node(op='mul', children=[Node(op='const', value=0.3), Node(op='R')]),
            Node(op='const', value=-1.0)
        ]))
        
        return templates
    
    def _compute_r2(self, tree: Node) -> float:
        pred = evaluate_tree(tree, self.R, self.g)
        valid = np.isfinite(pred)
        if valid.sum() < len(self.K) * 0.5:
            return -1.0
        K_v, pred_v = self.K[valid], pred[valid]
        ss_tot = np.sum((K_v - np.mean(K_v))**2)
        ss_res = np.sum((K_v - pred_v)**2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def _update_niches(self):
        """Update best individual per niche."""
        niche_groups = defaultdict(list)
        for ind in self.population:
            niche_groups[ind.structure].append(ind)
        
        self.best_by_niche = {}
        for struct, inds in niche_groups.items():
            best = min(inds, key=lambda x: x.mse)
            self.best_by_niche[struct] = best
    
    def evolve_generation(self) -> Individual:
        """Run one generation of evolution."""
        # Selection: tournament within niches
        new_pop = []
        
        # Elitism: keep best from each niche
        elites = sorted(self.best_by_niche.values(), key=lambda x: x.mse)[:self.n_niches]
        new_pop.extend([Individual(tree=e.tree.copy(), mse=e.mse, r2=e.r2) for e in elites])
        
        # Generate offspring
        while len(new_pop) < self.pop_size:
            # Tournament selection
            tournament = random.sample(self.population, min(5, len(self.population)))
            parent1 = min(tournament, key=lambda x: x.mse)
            tournament = random.sample(self.population, min(5, len(self.population)))
            parent2 = min(tournament, key=lambda x: x.mse)
            
            # Crossover
            if random.random() < 0.7:
                child1, child2 = crossover(parent1.tree, parent2.tree)
            else:
                child1, child2 = parent1.tree.copy(), parent2.tree.copy()
            
            # Mutation
            child1 = mutate(child1, prob=0.2)
            child2 = mutate(child2, prob=0.2)
            
            # Evaluate
            for child in [child1, child2]:
                if child.depth() <= self.max_depth and child.size() <= 30:
                    mse = optimize_constants(child, self.R, self.g, self.K)
                    r2 = self._compute_r2(child)
                    new_pop.append(Individual(tree=child, mse=mse, r2=r2))
        
        self.population = new_pop[:self.pop_size]
        self._update_niches()
        
        return min(self.population, key=lambda x: x.mse)
    
    def run(self, generations: int, verbose: bool = True) -> List[Individual]:
        """Run evolution for specified generations."""
        best_ever = None
        history = []
        
        for gen in range(generations):
            best = self.evolve_generation()
            
            if best_ever is None or best.mse < best_ever.mse:
                best_ever = Individual(tree=best.tree.copy(), mse=best.mse, r2=best.r2)
            
            if verbose and (gen % 5 == 0 or gen == generations - 1):
                n_structures = len(self.best_by_niche)
                print(f"  Gen {gen+1:3d}: R²={best.r2:.5f} | Niches={n_structures:2d} | {best.tree.to_string()[:60]}")
            
            history.append({
                'gen': gen,
                'best_r2': best.r2,
                'best_formula': best.tree.to_string(),
                'n_niches': len(self.best_by_niche)
            })
        
        # Get Pareto front
        pareto = self._get_pareto_front()
        
        return pareto, best_ever, history
    
    def _get_pareto_front(self) -> List[Individual]:
        """Get Pareto front (complexity vs accuracy)."""
        candidates = sorted(self.best_by_niche.values(), key=lambda x: x.mse)[:50]
        pareto = []
        
        for ind in candidates:
            dominated = False
            for other in candidates:
                if other.mse < ind.mse and other.tree.size() <= ind.tree.size():
                    dominated = True
                    break
            if not dominated:
                pareto.append(ind)
        
        return sorted(pareto, key=lambda x: x.mse)


def load_gaia_data(filepath: str = None, max_stars: int = 100000, seed: int = 42):
    """Load Gaia enhancement factor data."""
    import pandas as pd
    
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "gaia" / "outputs" / "mw_rar_starlevel_full.csv"
    
    df = pd.read_csv(filepath)
    
    if 'K' not in df.columns:
        if 'log10_g_obs' in df.columns and 'log10_g_bar' in df.columns:
            df['K'] = 10**(df['log10_g_obs'] - df['log10_g_bar']) - 1
    
    valid = (df['K'] > -0.5) & (df['K'] < 50) & df['K'].notna()
    df = df[valid]
    
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=seed)
    
    R = df['R_kpc'].values if 'R_kpc' in df.columns else df['R'].values
    g = df['log10_g_bar'].values if 'log10_g_bar' in df.columns else df['g_bar'].values
    K = df['K'].values
    
    return R, g, K


def main():
    parser = argparse.ArgumentParser(description='Σ-Gravity Improved Discovery')
    parser.add_argument('--data', type=str, help='Path to data CSV')
    parser.add_argument('--stars', type=int, default=50000, help='Max stars')
    parser.add_argument('--generations', type=int, default=30, help='Evolution generations')
    parser.add_argument('--pop-size', type=int, default=200, help='Population size')
    parser.add_argument('--export', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*70)
    print("  Σ-GRAVITY NICHED DISCOVERY ENGINE")
    print("="*70)
    
    print(f"\n  Loading Gaia data...")
    try:
        R, g, K = load_gaia_data(args.data, max_stars=args.stars)
        print(f"  ✓ Loaded {len(K):,} stars")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return
    
    print(f"\n  Data: {len(K):,} points")
    print(f"  R range: [{R.min():.2f}, {R.max():.2f}] kpc")
    print(f"  K mean: {np.mean(K):.3f} ± {np.std(K):.3f}")
    
    print(f"\n  Running niched evolution (pop={args.pop_size}, gen={args.generations})...")
    print("  " + "-"*65)
    
    engine = NichedDiscoveryEngine(pop_size=args.pop_size, n_niches=20, max_depth=5)
    engine.initialize(R, g, K)
    
    start = time.time()
    pareto, best, history = engine.run(generations=args.generations, verbose=True)
    elapsed = time.time() - start
    
    print("\n" + "="*70)
    print("  PARETO FRONT (complexity vs accuracy)")
    print("="*70)
    
    for i, ind in enumerate(pareto[:10], 1):
        print(f"\n  {i}. R² = {ind.r2:.6f} | Size = {ind.tree.size()}")
        print(f"     {ind.tree.to_string()}")
    
    print(f"\n  Total time: {elapsed:.1f}s")
    
    # Export
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    export_path = args.export or str(output_dir / "niched_discovery_results.json")
    
    output = {
        'best': {'r2': best.r2, 'formula': best.tree.to_string(), 'mse': best.mse},
        'pareto_front': [{'r2': p.r2, 'formula': p.tree.to_string(), 'size': p.tree.size()} for p in pareto],
        'history': history,
        'data_info': {'n_points': len(K), 'K_mean': float(np.mean(K)), 'K_std': float(np.std(K))},
    }
    
    with open(export_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to {export_path}")


if __name__ == '__main__':
    main()

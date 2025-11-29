"""
Expression Tree for Symbolic Regression
========================================

Represents mathematical expressions as trees that can be:
- Evaluated on data
- Mutated (modified randomly)
- Crossed over (combined with other expressions)
- Simplified and displayed as equations

This is the core of genetic programming for equation discovery.
"""

import numpy as np
import copy
import random
from typing import List, Dict, Optional, Tuple, Union

# Operator definitions
BINARY_OPS = {
    '+': lambda x, y: x + y,
    '-': lambda x, y: x - y,
    '*': lambda x, y: x * y,
    '/': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, x),  # Protected division
    '^': lambda x, y: np.where((x > 0) | (y == np.floor(y)), 
                                np.power(np.abs(x) + 1e-10, np.clip(y, -10, 10)), 
                                np.abs(x) ** np.clip(y, -10, 10)),  # Protected power
}

UNARY_OPS = {
    'neg': lambda x: -x,
    'sqrt': lambda x: np.sqrt(np.abs(x) + 1e-10),
    'square': lambda x: x ** 2,
    'cube': lambda x: x ** 3,
    'inv': lambda x: np.where(np.abs(x) > 1e-10, 1.0 / x, 1e10),  # Protected inverse
    'log': lambda x: np.log(np.abs(x) + 1e-10),
    'exp': lambda x: np.exp(np.clip(x, -20, 20)),
    'sin': np.sin,
    'cos': np.cos,
}

# String representations for printing
BINARY_OP_STRS = {
    '+': '({} + {})',
    '-': '({} - {})',
    '*': '({} * {})',
    '/': '({} / {})',
    '^': '({}^{})',
}

UNARY_OP_STRS = {
    'neg': '(-{})',
    'sqrt': 'sqrt({})',
    'square': '({}²)',
    'cube': '({}³)',
    'inv': '(1/{})',
    'log': 'log({})',
    'exp': 'exp({})',
    'sin': 'sin({})',
    'cos': 'cos({})',
}


class ExprNode:
    """A node in an expression tree."""
    
    def __init__(self, node_type: str, value=None, children: List['ExprNode'] = None):
        """
        Args:
            node_type: 'const', 'var', 'binary', or 'unary'
            value: The constant value, variable name, or operator name
            children: Child nodes (0 for terminals, 1 for unary, 2 for binary)
        """
        self.node_type = node_type
        self.value = value
        self.children = children or []
        
    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate expression on given variable values."""
        if self.node_type == 'const':
            # Return constant broadcast to match variable shapes
            first_var = next(iter(variables.values()))
            return np.full_like(first_var, self.value, dtype=np.float64)
        
        elif self.node_type == 'var':
            return variables[self.value].astype(np.float64)
        
        elif self.node_type == 'unary':
            child_val = self.children[0].evaluate(variables)
            return UNARY_OPS[self.value](child_val)
        
        elif self.node_type == 'binary':
            left_val = self.children[0].evaluate(variables)
            right_val = self.children[1].evaluate(variables)
            return BINARY_OPS[self.value](left_val, right_val)
        
        else:
            raise ValueError(f"Unknown node type: {self.node_type}")
    
    def to_string(self) -> str:
        """Convert expression to readable string."""
        if self.node_type == 'const':
            if abs(self.value - round(self.value)) < 0.001:
                return str(int(round(self.value)))
            return f"{self.value:.4g}"
        
        elif self.node_type == 'var':
            return self.value
        
        elif self.node_type == 'unary':
            child_str = self.children[0].to_string()
            return UNARY_OP_STRS[self.value].format(child_str)
        
        elif self.node_type == 'binary':
            left_str = self.children[0].to_string()
            right_str = self.children[1].to_string()
            return BINARY_OP_STRS[self.value].format(left_str, right_str)
        
        return "?"
    
    def copy(self) -> 'ExprNode':
        """Deep copy the expression tree."""
        return copy.deepcopy(self)
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)
    
    def size(self) -> int:
        """Count total nodes."""
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)
    
    def get_all_nodes(self) -> List[Tuple['ExprNode', Optional['ExprNode'], int]]:
        """Get all nodes with their parents and child indices."""
        nodes = [(self, None, -1)]
        for i, child in enumerate(self.children):
            child_nodes = child.get_all_nodes()
            # Update parent info for direct children
            child_nodes[0] = (child_nodes[0][0], self, i)
            nodes.extend(child_nodes)
        return nodes


class ExpressionTree:
    """
    A complete expression tree with genetic operators.
    """
    
    def __init__(self, root: ExprNode, variable_names: List[str]):
        self.root = root
        self.variable_names = variable_names
        self._fitness = None
        
    @property
    def fitness(self):
        return self._fitness
    
    @fitness.setter
    def fitness(self, value):
        self._fitness = value
    
    def evaluate(self, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate expression on data."""
        try:
            result = self.root.evaluate(variables)
            # Handle infinities and NaNs
            result = np.where(np.isfinite(result), result, 1e10)
            return result
        except Exception:
            return np.full(len(next(iter(variables.values()))), 1e10)
    
    def to_string(self) -> str:
        return self.root.to_string()
    
    def copy(self) -> 'ExpressionTree':
        new_tree = ExpressionTree(self.root.copy(), self.variable_names.copy())
        new_tree._fitness = self._fitness
        return new_tree
    
    def depth(self) -> int:
        return self.root.depth()
    
    def size(self) -> int:
        return self.root.size()
    
    @staticmethod
    def random_tree(variable_names: List[str], max_depth: int = 4, 
                    p_const: float = 0.3, const_range: Tuple[float, float] = (-10, 10),
                    use_unary: bool = True) -> 'ExpressionTree':
        """Generate a random expression tree."""
        
        def random_node(depth: int) -> ExprNode:
            # Terminal node if max depth reached or randomly chosen
            if depth >= max_depth or (depth > 1 and random.random() < 0.3):
                if random.random() < p_const:
                    # Constant - favor simple values
                    if random.random() < 0.5:
                        val = random.choice([0.5, 1, 2, 3, -1, -2, np.pi, 2*np.pi])
                    else:
                        val = random.uniform(*const_range)
                    return ExprNode('const', val)
                else:
                    return ExprNode('var', random.choice(variable_names))
            
            # Operator node
            if use_unary and random.random() < 0.3:
                op = random.choice(list(UNARY_OPS.keys()))
                child = random_node(depth + 1)
                return ExprNode('unary', op, [child])
            else:
                op = random.choice(list(BINARY_OPS.keys()))
                left = random_node(depth + 1)
                right = random_node(depth + 1)
                return ExprNode('binary', op, [left, right])
        
        root = random_node(0)
        return ExpressionTree(root, variable_names)
    
    def mutate(self, p_const: float = 0.3, const_range: Tuple[float, float] = (-10, 10)) -> 'ExpressionTree':
        """Create a mutated copy of this tree."""
        new_tree = self.copy()
        
        # Choose mutation type
        mutation_type = random.choice(['subtree', 'point', 'constant', 'operator'])
        
        nodes = new_tree.root.get_all_nodes()
        if not nodes:
            return new_tree
        
        node, parent, child_idx = random.choice(nodes)
        
        if mutation_type == 'subtree':
            # Replace subtree with random tree
            new_subtree = ExpressionTree.random_tree(
                self.variable_names, max_depth=3, p_const=p_const, const_range=const_range
            ).root
            if parent is None:
                new_tree.root = new_subtree
            else:
                parent.children[child_idx] = new_subtree
                
        elif mutation_type == 'point':
            # Change node type/value
            if node.node_type == 'const':
                node.value = random.uniform(*const_range)
            elif node.node_type == 'var':
                node.value = random.choice(self.variable_names)
            elif node.node_type == 'unary':
                node.value = random.choice(list(UNARY_OPS.keys()))
            elif node.node_type == 'binary':
                node.value = random.choice(list(BINARY_OPS.keys()))
                
        elif mutation_type == 'constant':
            # Find and modify a constant
            const_nodes = [n for n, _, _ in nodes if n.node_type == 'const']
            if const_nodes:
                c = random.choice(const_nodes)
                c.value *= random.uniform(0.5, 2.0)  # Scale
                c.value += random.uniform(-1, 1)     # Shift
                
        elif mutation_type == 'operator':
            # Change an operator
            op_nodes = [n for n, _, _ in nodes if n.node_type in ['unary', 'binary']]
            if op_nodes:
                op = random.choice(op_nodes)
                if op.node_type == 'unary':
                    op.value = random.choice(list(UNARY_OPS.keys()))
                else:
                    op.value = random.choice(list(BINARY_OPS.keys()))
        
        new_tree._fitness = None
        return new_tree
    
    @staticmethod
    def crossover(parent1: 'ExpressionTree', parent2: 'ExpressionTree') -> 'ExpressionTree':
        """Create offspring by swapping subtrees."""
        child = parent1.copy()
        
        nodes1 = child.root.get_all_nodes()
        nodes2 = parent2.root.get_all_nodes()
        
        if not nodes1 or not nodes2:
            return child
        
        # Select random nodes
        node1, parent1_node, child_idx1 = random.choice(nodes1)
        node2, _, _ = random.choice(nodes2)
        
        # Swap subtree from parent2 into child
        new_subtree = node2.copy() if hasattr(node2, 'copy') else copy.deepcopy(node2)
        
        if parent1_node is None:
            child.root = new_subtree
        else:
            parent1_node.children[child_idx1] = new_subtree
        
        child._fitness = None
        return child


def simplify_constants(expr: ExpressionTree) -> ExpressionTree:
    """
    Attempt to simplify constant expressions.
    E.g., (2 * 3) -> 6, (x + 0) -> x
    """
    # This is a basic implementation - a full CAS would be better
    new_expr = expr.copy()
    
    def simplify_node(node: ExprNode) -> ExprNode:
        if not node.children:
            return node
        
        # Recursively simplify children
        node.children = [simplify_node(c) for c in node.children]
        
        # Check if all children are constants
        if all(c.node_type == 'const' for c in node.children):
            try:
                if node.node_type == 'unary':
                    result = UNARY_OPS[node.value](node.children[0].value)
                elif node.node_type == 'binary':
                    result = BINARY_OPS[node.value](
                        node.children[0].value, 
                        node.children[1].value
                    )
                if np.isfinite(result):
                    return ExprNode('const', float(result))
            except Exception:
                pass
        
        return node
    
    new_expr.root = simplify_node(new_expr.root)
    return new_expr

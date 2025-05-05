import random
import math
from textblob import TextBlob
from evaluate import evaluate_response_with_textblob

class ConversationState:
    def __init__(self, message, history=None, score=None):
        self.message = message
        self.history = history or []
        self.score = score

    def is_terminal(self):
        return len(self.history) >= 3

    def generate_children(self, generate_variants_fn):
        variants = generate_variants_fn(self.message)
        return [ConversationState(msg, self.history + [self.message]) 
                for msg in variants]


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.unexplored_actions = None

    def expand(self, generate_variants_fn):
        """Expand the node by generating children states"""
        if not self.unexplored_actions:
            child_states = self.state.generate_children(generate_variants_fn)
            self.unexplored_actions = child_states
        
        if self.unexplored_actions:
            child_state = self.unexplored_actions.pop(0)
            child_node = MCTSNode(child_state, self)
            self.children.append(child_node)
            return child_node
        return None

    def is_fully_expanded(self):
        """Check if all possible actions from this state have been explored"""
        return self.unexplored_actions is not None and len(self.unexplored_actions) == 0

    def best_child(self, c_param=1.4):
        """Select the best child node using UCB1 formula"""
        if not self.children:
            return None
            
        ucb_values = []
        for child in self.children:
            exploitation = child.value / (child.visits + 1e-5)
            exploration = c_param * ((2 * math.log(self.visits + 1e-5)) / (child.visits + 1e-5)) ** 0.5
            ucb_values.append(exploitation + exploration)
            
        return self.children[ucb_values.index(max(ucb_values))]

    def backpropagate(self, result):
        """Update node statistics after a simulation"""
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)
    
    def get_all_leaf_nodes(self):
        """Returns all leaf nodes in the subtree rooted at this node."""
        if not self.children:
            return [self]
        
        leaf_nodes = []
        for child in self.children:
            leaf_nodes.extend(child.get_all_leaf_nodes())
        return leaf_nodes


def rollout(state, evaluate_fn):
    """Simulate from the given state to a terminal state and evaluate"""
    score = evaluate_fn(state.message)
    if isinstance(score, dict):
        return score["final_score"]
    return score


def mcts_search(initial_state, generate_variants_fn, evaluate_fn, iterations=10, return_all=False):
    """Run MCTS algorithm to find the best message variant"""
    root = MCTSNode(initial_state)
    
    for i in range(iterations):
        # Selection: Find the most promising node to expand
        node = selection(root)
        
        # Expansion: Create a new child node
        if not node.state.is_terminal():
            node = expansion(node, generate_variants_fn)
        
        # Simulation: Simulate from the new node to get a result
        result = rollout(node.state, evaluate_fn)
        
        # Backpropagation: Update statistics in the tree
        node.backpropagate(result)
    
    if return_all:
        all_nodes = []
        for node in root.get_all_leaf_nodes():
            if node.state.score is None:
                raw_score = evaluate_fn(node.state.message)
                if isinstance(raw_score, dict):
                    node.state.score = raw_score
                else:
                    sentiment = TextBlob(node.state.message).sentiment
                    node.state.score = {
                        "final_score": raw_score,
                        "polarity": sentiment.polarity,
                        "subjectivity": sentiment.subjectivity
                    }
            
            all_nodes.append({
                'message': node.state.message,
                'final_score': node.state.score.get("final_score", 0),
                'polarity': node.state.score.get("polarity", 0),
                'subjectivity': node.state.score.get("subjectivity", 0),
                'visits': node.visits,
                'value': node.value
            })
        
        best_messages = sorted(all_nodes, key=lambda x: x['visits'] > 0 and x['value'] / x['visits'] or 0, reverse=True)
        return best_messages[0]['message'] if best_messages else initial_state.message, all_nodes
    
    best_child = max(root.children, key=lambda c: c.visits) if root.children else root
    return best_child.state.message


def selection(node):
    """Select a node to expand using UCB1"""
    current = node
    while not current.state.is_terminal() and current.is_fully_expanded():
        best_child = current.best_child()
        if best_child is None:
            break
        current = best_child
    return current


def expansion(node, generate_variants_fn):
    """Expand the selected node"""
    if node.is_fully_expanded():
        return node
    
    new_node = node.expand(generate_variants_fn)
    return new_node if new_node else node
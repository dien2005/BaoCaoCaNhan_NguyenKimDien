import pygame
from collections import deque
import heapq
import time
from dataclasses import dataclass, field
import itertools
from typing import List, Tuple, Optional, Dict, Callable, Set
import random
import math
import numpy as np

# Khởi tạo Pygame
pygame.init()
clock = pygame.time.Clock()

# Cài đặt giao diện
WINDOW_SIZE = (800, 700)
TILE_SIZE = 120
BOARD_SIZE = 3
PADDING = 20
COLORS = {
    'GREY': (200, 200, 200),
    'DARK_GREY': (100, 100, 100),
    'BLUE': (70, 130, 180),
    'DARK_BLUE': (50, 110, 160),
    'LIGHT_BLUE': (135, 206, 250),
    'BUTTON_HOVER': (100, 149, 237),
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0)
}

WINDOW = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("8-Puzzle Solver")

FONTS = {
    'tile': pygame.font.SysFont("Arial", 60, bold=True),
    'button': pygame.font.SysFont("Arial", 32, bold=True),
    'status': pygame.font.SysFont("Arial", 28),
    'metrics': pygame.font.SysFont("Arial", 24)
}

@dataclass
class PuzzleState:
    state: List[int]
    path: List[Tuple[int, int]]
    cost: int = 0
    id: int = field(default_factory=lambda: next(PuzzleState._id_counter))
    
    _id_counter = itertools.count()
    
    def __hash__(self):
        return hash(tuple(self.state))
    
    def __lt__(self, other):
        return self.id < other.id

@dataclass
class BeliefState:
    states: Set[PuzzleState]
    path: List[Tuple[int, int]]
    cost: int = 0
    id: int = field(default_factory=lambda: next(BeliefState._id_counter))
    
    _id_counter = itertools.count()
    
    def __hash__(self):
        return hash(frozenset(tuple(s.state) for s in self.states))
    
    def __lt__(self, other):
        return self.id < other.id

class PuzzleSolver:
    GOAL_STATE = list(range(1, 9)) + [0]
    MOVES = [(-3, lambda i: i >= 3),  # Lên
             (3, lambda i: i < 6),   # Xuống
             (-1, lambda i: i % 3 > 0),  # Trái
             (1, lambda i: i % 3 < 2)]   # Phải

    def is_solvable(self, state: List[int]) -> bool:
        inversions = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] != 0 and state[j] != 0 and state[i] > state[j]:
                    inversions += 1
        zero_row_from_bottom = 3 - (state.index(0) // 3)
        return (inversions % 2) == (zero_row_from_bottom % 2)

    def get_valid_moves(self, zero_idx: int) -> List[int]:
        return [move for move, condition in self.MOVES if condition(zero_idx)]

    def manhattan_distance(self, state: List[int]) -> int:
        goal_pos = {val: (i % 3, i // 3) for i, val in enumerate(range(9))}
        return sum(abs((state.index(val) % 3) - goal_pos[val][0]) + 
                  abs((state.index(val) // 3) - goal_pos[val][1]) 
                  for val in state if val != 0)

    def generate_new_states(self, current: PuzzleState, zero_idx: int) -> List[PuzzleState]:
        new_states = []
        for move in self.get_valid_moves(zero_idx):
            new_idx = zero_idx + move
            new_state = current.state.copy()
            new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
            new_path = current.path + [(zero_idx, new_idx)]
            new_states.append(PuzzleState(new_state, new_path, current.cost + 1))
        return new_states

    def generate_random_solvable_state(self) -> List[int]:
        while True:
            state = list(range(9))
            random.shuffle(state)
            if self.is_solvable(state):
                return state

    def generate_random_solvable_state_with_first_tile(self, first_tile: int = 2) -> List[int]:
        while True:
            state = [first_tile] + random.sample([i for i in range(9) if i != first_tile], 8)
            if self.is_solvable(state):
                return state

    def solve(self, start_state: List[int], algorithm: str) -> Tuple[Optional[List], Dict]:
        if not self.is_solvable(start_state):
            return None, {"time": 0, "nodes": 0, "error": "Unsolvable puzzle"}
        
        algorithms = {
            'bfs': self.bfs,
            'dfs': self.dfs,
            'ucs': self.ucs,
            'greedy': self.greedy,
            'iddfs': self.iddfs,
            'astar': self.astar,
            'ida_star': self.ida_star,
            'hill': self.hill_climbing_base,
            'shill': self.steepest_ascent_hill_climbing,
            'randomhill': self.random_hill_climbing,
            'sa': self.simulated_annealing,
            'bs': self.beam_search,
            'andor': self.and_or_search,
            'ga': self.genetic_algorithm,
            'noobs': self.search_no_observation,
            'belief_astar': self.belief_astar,
            'ac3': self.ac3,
            'backtracking': self.backtracking,
            'qlearning': self.q_learning,
            'reinforce': self.reinforce,
            'global_beam': self.global_beam_search,
            'bidirectional': self.bidirectional_search,
            'tabu': self.tabu_search
        }
        return algorithms[algorithm](start_state)

    def bfs(self, start_state: List[int]):
        queue = deque([PuzzleState(start_state, [])])
        return self._generic_search(queue, lambda q: q.popleft())

    def dfs(self, start_state: List[int], max_depth: int = 1000):
        stack = [PuzzleState(start_state, [])]
        return self._generic_search(stack, lambda q: q.pop(), max_depth)

    def ucs(self, start_state: List[int]):
        counter = itertools.count()
        queue = [(0, next(counter), PuzzleState(start_state, []))]
        return self._generic_search(queue, heapq.heappop, is_priority=True)

    def greedy(self, start_state: List[int]):
        counter = itertools.count()
        queue = [(self.manhattan_distance(start_state), next(counter), PuzzleState(start_state, []))]
        return self._generic_search(queue, heapq.heappop, is_priority=True, heuristic=lambda s: self.manhattan_distance(s.state))

    def astar(self, start_state: List[int]):
        counter = itertools.count()
        queue = [(self.manhattan_distance(start_state), next(counter), 0, PuzzleState(start_state, []))]
        return self._generic_search(queue, heapq.heappop, is_priority=True, 
                                  heuristic=lambda s: self.manhattan_distance(s.state) + s.cost)

    def hill_climbing_base(self, start_state: List[int], max_iterations: int = 1000):
        current = PuzzleState(start_state, [])
        start_time = time.time()
        nodes_expanded = 0
        
        for _ in range(max_iterations):
            nodes_expanded += 1
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            best_neighbor = None
            for neighbor in neighbors:
                if self.manhattan_distance(neighbor.state) < self.manhattan_distance(current.state):
                    best_neighbor = neighbor
                    break
            if not best_neighbor or self.manhattan_distance(best_neighbor.state) >= self.manhattan_distance(current.state):
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            current = best_neighbor
        return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def steepest_ascent_hill_climbing(self, start_state: List[int], max_iterations: int = 1000):
        current = PuzzleState(start_state, [])
        start_time = time.time()
        nodes_expanded = 0
        
        for _ in range(max_iterations):
            nodes_expanded += 1
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            if not neighbors:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            best_neighbor = min(neighbors, 
                              key=lambda x: self.manhattan_distance(x.state))
            if self.manhattan_distance(best_neighbor.state) >= self.manhattan_distance(current.state):
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            current = best_neighbor
        return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def random_hill_climbing(self, start_state: List[int], max_iterations: int = 1000):
        current = PuzzleState(start_state, [])
        start_time = time.time()
        nodes_expanded = 0
        
        for _ in range(max_iterations):
            nodes_expanded += 1
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            for neighbor in neighbors:
                if self.manhattan_distance(neighbor.state) < self.manhattan_distance(current.state):
                    current = neighbor
                    break
        return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def simulated_annealing(self, start_state: List[int], max_iterations: int = 1000):
        T = 100
        current = PuzzleState(start_state, [])
        best = current
        start_time = time.time()
        nodes_expanded = 0
        
        for iteration in range(max_iterations):
            nodes_expanded += 1
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            if not neighbors:
                continue
            
            next_state = random.choice(neighbors)
            delta = self.manhattan_distance(next_state.state) - self.manhattan_distance(current.state)
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = next_state
            
            if self.manhattan_distance(current.state) < self.manhattan_distance(best.state):
                best = current
            
            T = T / (1 + math.log1p(iteration + 1))
            if T < 0.1:
                break
        
        return best.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def beam_search(self, start_state: List[int], max_iterations: int = 1000):
        beam_width = 3
        current_beam = [PuzzleState(start_state, [])]
        visited = set([tuple(start_state)])
        start_time = time.time()
        nodes_expanded = 0
        
        for _ in range(max_iterations):
            nodes_expanded += 1
            for state in current_beam:
                if state.state == self.GOAL_STATE:
                    return state.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            all_neighbors = []
            for state in current_beam:
                zero_idx = state.state.index(0)
                neighbors = self.generate_new_states(state, zero_idx)
                for neighbor in neighbors:
                    if tuple(neighbor.state) not in visited:
                        visited.add(tuple(neighbor.state))
                        all_neighbors.append(neighbor)
            
            if not all_neighbors:
                break
            all_neighbors.sort(key=lambda x: self.manhattan_distance(x.state))
            current_beam = all_neighbors[:beam_width]
        
        best_state = min(current_beam, key=lambda x: self.manhattan_distance(x.state))
        return best_state.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def and_or_search(self, start_state: List[int], max_iterations: int = 1000):
        start_time = time.time()
        nodes_expanded = 0
        visited = set()
        counter = itertools.count()
        
        queue = [(self.manhattan_distance(start_state), next(counter), PuzzleState(start_state, []))]
        heapq.heapify(queue)
        
        while queue and nodes_expanded < max_iterations:
            nodes_expanded += 1
            _, _, current = heapq.heappop(queue)
            
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            state_tuple = tuple(current.state)
            if state_tuple in visited:
                continue
                
            visited.add(state_tuple)
            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            
            for neighbor in neighbors:
                if tuple(neighbor.state) not in visited:
                    f_cost = neighbor.cost + self.manhattan_distance(neighbor.state)
                    heapq.heappush(queue, (f_cost, next(counter), neighbor))
        
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def genetic_algorithm(self, start_state: List[int], population_size: int = 100, max_generations: int = 1000, mutation_rate: float = 0.1):
        start_time = time.time()
        nodes_expanded = 0
        
        def apply_moves(state: List[int], moves: List[int]) -> List[int]:
            current = state.copy()
            zero_idx = current.index(0)
            for move in moves:
                new_idx = zero_idx + move
                if new_idx in range(9) and move in self.get_valid_moves(zero_idx):
                    current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
                    zero_idx = new_idx
            return current
        
        def fitness(moves: List[int]) -> float:
            result_state = apply_moves(start_state, moves)
            dist = self.manhattan_distance(result_state)
            return -dist - (len(moves) * 0.1)
        
        def generate_random_individual(length: int) -> List[int]:
            return [random.choice(self.get_valid_moves(start_state.index(0))) for _ in range(length)]
        
        def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
            point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            return parent1[:point] + parent2[point:]
        
        def mutate(individual: List[int]) -> List[int]:
            if random.random() < mutation_rate:
                idx = random.randint(0, len(individual) - 1)
                zero_idx = apply_moves(start_state, individual[:idx]).index(0)
                individual[idx] = random.choice(self.get_valid_moves(zero_idx))
            return individual
        
        initial_dist = self.manhattan_distance(start_state)
        individual_length = max(10, int(initial_dist * 1.5))
        population = [generate_random_individual(individual_length) for _ in range(population_size)]
        nodes_expanded += population_size
        
        for generation in range(max_generations):
            population = sorted(population, key=fitness, reverse=True)
            
            best_individual = population[0]
            best_state = apply_moves(start_state, best_individual)
            if best_state == self.GOAL_STATE:
                current = start_state.copy()
                zero_idx = current.index(0)
                path = []
                for move in best_individual:
                    new_idx = zero_idx + move
                    if new_idx in range(9) and move in self.get_valid_moves(zero_idx):
                        path.append((zero_idx, new_idx))
                        current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
                        zero_idx = new_idx
                return path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            elite_size = population_size // 2
            new_population = population[:elite_size]
            
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population[:elite_size], 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
                nodes_expanded += 1
            
            population = new_population
        
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def search_no_observation(self, start_state: List[int], max_iterations: int = 1000, belief_size: int = 3):
        start_time = time.time()
        nodes_expanded = 0
        
        belief_states = set()
        while len(belief_states) < belief_size:
            random_state = self.generate_random_solvable_state_with_first_tile(first_tile=start_state[0])
            if tuple(random_state) not in {tuple(s.state) for s in belief_states}:
                belief_states.add(PuzzleState(random_state, []))
        initial_belief = BeliefState(belief_states, [])
        
        queue = deque([initial_belief])
        visited = set([frozenset(tuple(s.state) for s in initial_belief.states)])
        nodes_expanded += 1
        
        while queue and nodes_expanded < max_iterations:
            current_belief = queue.popleft()
            nodes_expanded += 1
            
            if all(s.state == self.GOAL_STATE for s in current_belief.states):
                return current_belief.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            possible_moves = set()
            for state in current_belief.states:
                zero_idx = state.state.index(0)
                possible_moves.update(self.get_valid_moves(zero_idx))
            
            for move in possible_moves:
                new_belief_states = set()
                for state in current_belief.states:
                    zero_idx = state.state.index(0)
                    if move in self.get_valid_moves(zero_idx):
                        new_idx = zero_idx + move
                        new_state = state.state.copy()
                        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                        new_path = state.path + [(zero_idx, new_idx)]
                        new_belief_states.add(PuzzleState(new_state, new_path, state.cost + 1))
                    else:
                        new_belief_states.add(state)
                
                new_belief = BeliefState(new_belief_states, current_belief.path + [(0, move)])
                belief_tuple = frozenset(tuple(s.state) for s in new_belief.states)
                
                if belief_tuple not in visited:
                    visited.add(belief_tuple)
                    queue.append(new_belief)
        
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def belief_astar(self, start_state: List[int], max_iterations: int = 1000, belief_size: int = 3):
        start_time = time.time()
        nodes_expanded = 0
        
        belief_states = set()
        while len(belief_states) < belief_size:
            random_state = self.generate_random_solvable_state_with_first_tile(first_tile=start_state[0])
            if tuple(random_state) not in {tuple(s.state) for s in belief_states}:
                belief_states.add(PuzzleState(random_state, []))
        initial_belief = BeliefState(belief_states, [])
        
        counter = itertools.count()
        queue = [(self._belief_heuristic(initial_belief), next(counter), initial_belief)]
        heapq.heapify(queue)
        visited = set([frozenset(tuple(s.state) for s in initial_belief.states)])
        nodes_expanded += 1
        
        while queue and nodes_expanded < max_iterations:
            _, _, current_belief = heapq.heappop(queue)
            nodes_expanded += 1
            
            if all(s.state == self.GOAL_STATE for s in current_belief.states):
                return current_belief.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            possible_moves = set()
            for state in current_belief.states:
                zero_idx = state.state.index(0)
                possible_moves.update(self.get_valid_moves(zero_idx))
            
            for move in possible_moves:
                new_belief_states = set()
                for state in current_belief.states:
                    zero_idx = state.state.index(0)
                    if move in self.get_valid_moves(zero_idx):
                        new_idx = zero_idx + move
                        new_state = state.state.copy()
                        new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                        new_path = state.path + [(zero_idx, new_idx)]
                        new_belief_states.add(PuzzleState(new_state, new_path, state.cost + 1))
                    else:
                        new_belief_states.add(state)
                
                new_belief = BeliefState(new_belief_states, current_belief.path + [(0, move)])
                belief_tuple = frozenset(tuple(s.state) for s in new_belief.states)
                
                if belief_tuple not in visited:
                    visited.add(belief_tuple)
                    f_score = self._belief_heuristic(new_belief) + new_belief.cost
                    heapq.heappush(queue, (f_score, next(counter), new_belief))
        
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def _belief_heuristic(self, belief: BeliefState) -> float:
        if not belief.states:
            return float('inf')
        return sum(self.manhattan_distance(s.state) for s in belief.states) / len(belief.states)

    def _generic_search(self, queue, extract: Callable, max_depth: int = float('inf'), 
                       is_priority: bool = False, heuristic: Callable = None):
        initial_state = queue[0][-1].state if is_priority else queue[0].state
        visited = set([tuple(initial_state)])
        start_time = time.time()
        nodes_expanded = 0
        counter = itertools.count()

        while queue:
            nodes_expanded += 1
            if is_priority:
                current = extract(queue)[-1]
            else:
                current = extract(queue)
            
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
            
            if len(current.path) >= max_depth:
                continue

            zero_idx = current.state.index(0)
            for new_state in self.generate_new_states(current, zero_idx):
                state_tuple = tuple(new_state.state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    if is_priority:
                        priority = heuristic(new_state) if heuristic else new_state.cost
                        heapq.heappush(queue, (priority, next(counter), new_state))
                    else:
                        queue.append(new_state)
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def iddfs(self, start_state: List[int]):
        start_time = time.time()
        nodes_expanded = 0
        
        for depth in range(5, 1000, 5):
            visited = set()
            result, new_nodes = self._dls(PuzzleState(start_state, []), depth, visited, nodes_expanded)
            nodes_expanded = new_nodes
            if result:
                return result, {"time": time.time() - start_time, "nodes": nodes_expanded}
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def _dls(self, current: PuzzleState, depth: int, visited: set, nodes_expanded: int):
        nodes_expanded += 1
        if current.state == self.GOAL_STATE:
            return current.path, nodes_expanded
        if len(current.path) >= depth:
            return None, nodes_expanded

        zero_idx = current.state.index(0)
        for new_state in self.generate_new_states(current, zero_idx):
            state_tuple = tuple(new_state.state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                result, nodes_expanded = self._dls(new_state, depth, visited, nodes_expanded)
                if result:
                    return result, nodes_expanded
        return None, nodes_expanded

    def ida_star(self, start_state: List[int]):
        start_time = time.time()
        nodes_expanded = 0
        
        def search(state: PuzzleState, bound: int, visited: set) -> Tuple[Optional[List], float]:
            nonlocal nodes_expanded
            nodes_expanded += 1
            f_score = state.cost + self.manhattan_distance(state.state)
            if f_score > bound:
                return None, f_score
            if state.state == self.GOAL_STATE:
                return state.path, f_score
                
            min_exceeded = float('inf')
            zero_idx = state.state.index(0)
            for new_state in self.generate_new_states(state, zero_idx):
                state_tuple = tuple(new_state.state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    result, new_f = search(new_state, bound, visited)
                    visited.remove(state_tuple)
                    if result:
                        return result, new_f
                    min_exceeded = min(min_exceeded, new_f)
            return None, min_exceeded

        bound = self.manhattan_distance(start_state)
        while True:
            visited = {tuple(start_state)}
            solution, new_bound = search(PuzzleState(start_state, []), bound, visited)
            if solution:
                return solution, {"time": time.time() - start_time, "nodes": nodes_expanded}
            if new_bound == float('inf'):
                return None, {"time": time.time() - start_time, "nodes": nodes_expanded}
            bound = new_bound



    def ac3(self, start_state: List[int]):
        """AC-3 algorithm for 8-puzzle as a CSP."""
        start_time = time.time()
        nodes_expanded = 0


        variables = list(range(9))  
        domains = {v: list(range(9)) for v in variables}  
        initial_state = {i: start_state[i] for i in range(9)}
        for pos, val in initial_state.items():
            domains[val] = [pos]


        constraints = []
        for i in range(9):
            for j in range(i + 1, 9):
                constraints.append((i, j))

        def revise(xi, xj):
            revised = False
            values_to_remove = []
            for x in domains[xi]:
                if all(x == y for y in domains[xj]):
                    values_to_remove.append(x)
                    revised = True
            for x in values_to_remove:
                domains[xi].remove(x)
            return revised


        queue = deque(constraints)
        while queue:
            nodes_expanded += 1
            (xi, xj) = queue.popleft()
            if revise(xi, xj):
                if len(domains[xi]) == 0:
                    return None, {"time": time.time() - start_time, "nodes": nodes_expanded, "error": "Inconsistent CSP"}
                for xk in variables:
                    if xk != xj and xk != xi:
                        queue.append((xk, xi))


        if all(len(domains[v]) == 1 for v in variables):
            final_state = [0] * 9
            for v in variables:
                final_state[domains[v][0]] = v
            if final_state == self.GOAL_STATE:
                return [], {"time": time.time() - start_time, "nodes": nodes_expanded}
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def backtracking(self, start_state: List[int]):
        """Backtracking search for 8-puzzle as a CSP."""
        start_time = time.time()
        nodes_expanded = 0


        variables = list(range(9))  # Positions 0-8
        domains = {v: list(range(9)) for v in variables}  
        assignment = {i: start_state[i] for i in range(9)}

        def is_consistent(var, value, assignment):
            for assigned_var, assigned_val in assignment.items():
                if assigned_var != var and assigned_val == value:
                    return False
            return True

        def backtrack(assignment, domains):
            nonlocal nodes_expanded
            nodes_expanded += 1
            if len(assignment) == len(variables):
                state = [0] * 9
                for pos, tile in assignment.items():
                    state[pos] = tile
                if state == self.GOAL_STATE:
                    return []
                return None

            var = next((v for v in variables if v not in assignment), None)
            if var is None:
                return None

            for value in domains[var]:
                if is_consistent(var, value, assignment):
                    assignment[var] = value
                    result = backtrack(assignment, domains)
                    if result is not None:
                        return result
                    del assignment[var]
            return None

        result = backtrack(assignment, domains)
        return result, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def q_learning(self, start_state: List[int], episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        """Q-Learning for 8-puzzle."""
        start_time = time.time()
        nodes_expanded = 0

        q_table = {}
        def get_q(state, action):
            return q_table.get((tuple(state), action), 0.0)

        def update_q(state, action, reward, next_state):
            current_q = get_q(state, action)
            max_next_q = max([get_q(next_state, a) for a in self.get_valid_moves(next_state.index(0))] + [0])
            q_table[(tuple(state), action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)

        def get_action(state):
            if random.random() < epsilon:
                return random.choice(self.get_valid_moves(state.index(0)))
            valid_moves = self.get_valid_moves(state.index(0))
            if not valid_moves:
                return None
            return max(valid_moves, key=lambda a: get_q(state, a), default=random.choice(valid_moves))

        path = []
        for _ in range(episodes):
            current_state = start_state.copy()
            nodes_expanded += 1
            steps = 0
            while current_state != self.GOAL_STATE and steps < 100:
                zero_idx = current_state.index(0)
                action = get_action(current_state)
                if action is None:
                    break
                new_idx = zero_idx + action
                next_state = current_state.copy()
                next_state[zero_idx], next_state[new_idx] = next_state[new_idx], next_state[zero_idx]
                reward = -self.manhattan_distance(next_state)
                update_q(current_state, action, reward, next_state)
                current_state = next_state
                steps += 1
                nodes_expanded += 1


        current_state = start_state.copy()
        path = []
        steps = 0
        visited = set()
        while current_state != self.GOAL_STATE and steps < 100:
            state_tuple = tuple(current_state)
            if state_tuple in visited:
                break
            visited.add(state_tuple)
            zero_idx = current_state.index(0)
            action = max(self.get_valid_moves(zero_idx), key=lambda a: get_q(current_state, a), default=None)
            if action is None:
                break
            new_idx = zero_idx + action
            path.append((zero_idx, new_idx))
            current_state[zero_idx], current_state[new_idx] = current_state[new_idx], current_state[zero_idx]
            steps += 1
            nodes_expanded += 1

        if current_state == self.GOAL_STATE:
            return path, {"time": time.time() - start_time, "nodes": nodes_expanded}
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def reinforce(self, start_state: List[int], episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.9):
        """REINFORCE algorithm for 8-puzzle."""
        start_time = time.time()
        nodes_expanded = 0

        policy = {}
        def get_policy(state):
            if tuple(state) not in policy:
                valid_moves = self.get_valid_moves(state.index(0))
                if not valid_moves:
                    return {}
                policy[tuple(state)] = {move: 1/len(valid_moves) for move in self.MOVES if move in valid_moves}
            return policy[tuple(state)]

        def choose_action(state):
            probs = get_policy(state)
            if not probs:  
                return None
            valid_moves = [move for move, prob in probs.items() if prob > 0]
            valid_probs = [probs[move] for move in valid_moves]
            if not valid_moves:
                return None
            return random.choices(valid_moves, weights=valid_probs, k=1)[0]

        for _ in range(episodes):
            states, actions, rewards = [], [], []
            current_state = start_state.copy()
            steps = 0
            while current_state != self.GOAL_STATE and steps < 100:
                nodes_expanded += 1
                zero_idx = current_state.index(0)
                action = choose_action(current_state)
                if action is None:  
                    break
                new_idx = zero_idx + action
                next_state = current_state.copy()
                next_state[zero_idx], next_state[new_idx] = next_state[new_idx], next_state[zero_idx]
                reward = -self.manhattan_distance(next_state)
                states.append(current_state[:])
                actions.append(action)
                rewards.append(reward)
                current_state = next_state
                steps += 1

            G = 0
            for t in range(len(states) - 1, -1, -1):
                G = rewards[t] + gamma * G
                state = states[t]
                action = actions[t]
                probs = get_policy(state)
                if action in probs and probs[action] > 0:
                    probs[action] += alpha * G * (1 - probs[action])


        current_state = start_state.copy()
        path = []
        steps = 0
        visited = set()
        while current_state != self.GOAL_STATE and steps < 100:
            state_tuple = tuple(current_state)
            if state_tuple in visited:
                break
            visited.add(state_tuple)
            zero_idx = current_state.index(0)
            action = choose_action(current_state)
            if action is None:
                break
            new_idx = zero_idx + action
            path.append((zero_idx, new_idx))
            current_state[zero_idx], current_state[new_idx] = current_state[new_idx], current_state[zero_idx]
            steps += 1
            nodes_expanded += 1

        if current_state == self.GOAL_STATE:
            return path, {"time": time.time() - start_time, "nodes": nodes_expanded}
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def global_beam_search(self, start_state: List[int], beam_width: int = 3, max_iterations: int = 1000):
        """Global Beam Search for 8-puzzle."""
        start_time = time.time()
        nodes_expanded = 0
        queue = [(self.manhattan_distance(start_state), next(itertools.count()), PuzzleState(start_state, []))]
        visited = set([tuple(start_state)])

        for _ in range(max_iterations):
            nodes_expanded += 1
            if not queue:
                break

            new_queue = []
            for _ in range(min(beam_width, len(queue))):
                if not queue:
                    break
                _, _, current = heapq.heappop(queue)
                if current.state == self.GOAL_STATE:
                    return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

                zero_idx = current.state.index(0)
                for new_state in self.generate_new_states(current, zero_idx):
                    state_tuple = tuple(new_state.state)
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        f_score = self.manhattan_distance(new_state.state)
                        heapq.heappush(new_queue, (f_score, next(itertools.count()), new_state))

            queue = new_queue[:beam_width]

        if queue:
            best_state = min(queue, key=lambda x: x[0])[2]
            return best_state.path, {"time": time.time() - start_time, "nodes": nodes_expanded}
        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def bidirectional_search(self, start_state: List[int]):
        """Bidirectional Search for 8-puzzle."""
        start_time = time.time()
        nodes_expanded = 0

        forward_queue = deque([PuzzleState(start_state, [])])
        backward_queue = deque([PuzzleState(self.GOAL_STATE, [])])
        forward_visited = {tuple(start_state): None}
        backward_visited = {tuple(self.GOAL_STATE): None}

        while forward_queue and backward_queue:
            nodes_expanded += 1


            current_forward = forward_queue.popleft()
            forward_zero_idx = current_forward.state.index(0)
            for new_state in self.generate_new_states(current_forward, forward_zero_idx):
                state_tuple = tuple(new_state.state)
                if state_tuple in backward_visited:

                    forward_path = new_state.path
                    backward_path = backward_visited[state_tuple]
                    if backward_path is not None:
                        backward_path = [(b, a) for (a, b) in backward_path[::-1]]
                        return forward_path + backward_path, {"time": time.time() - start_time, "nodes": nodes_expanded}
                if state_tuple not in forward_visited:
                    forward_visited[state_tuple] = new_state.path
                    forward_queue.append(new_state)


            current_backward = backward_queue.popleft()
            backward_zero_idx = current_backward.state.index(0)
            for new_state in self.generate_new_states(current_backward, backward_zero_idx):
                state_tuple = tuple(new_state.state)
                if state_tuple in forward_visited:
                    backward_path = new_state.path
                    forward_path = forward_visited[state_tuple]
                    if forward_path is not None:
                        backward_path = [(b, a) for (a, b) in backward_path[::-1]]
                        return forward_path + backward_path, {"time": time.time() - start_time, "nodes": nodes_expanded}
                if state_tuple not in backward_visited:
                    backward_visited[state_tuple] = new_state.path
                    backward_queue.append(new_state)

        return None, {"time": time.time() - start_time, "nodes": nodes_expanded}

    def tabu_search(self, start_state: List[int], max_iterations: int = 1000, tabu_tenure: int = 10):
        """Tabu Search for 8-puzzle."""
        start_time = time.time()
        nodes_expanded = 0
        current = PuzzleState(start_state, [])
        best = current
        tabu_list = deque(maxlen=tabu_tenure)

        for _ in range(max_iterations):
            nodes_expanded += 1
            if current.state == self.GOAL_STATE:
                return current.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

            zero_idx = current.state.index(0)
            neighbors = self.generate_new_states(current, zero_idx)
            if not neighbors:
                continue

            best_neighbor = None
            best_score = float('inf')
            for neighbor in neighbors:
                if tuple(neighbor.state) not in tabu_list:
                    score = self.manhattan_distance(neighbor.state)
                    if score < best_score:
                        best_score = score
                        best_neighbor = neighbor

            if best_neighbor is None:
                best_neighbor = min(neighbors, key=lambda x: self.manhattan_distance(x.state))

            current = best_neighbor
            tabu_list.append(tuple(current.state))
            if self.manhattan_distance(current.state) < self.manhattan_distance(best.state):
                best = current

        return best.path, {"time": time.time() - start_time, "nodes": nodes_expanded}

class PuzzleUI:
    def __init__(self):
        self.solver = PuzzleSolver()
        self.initial_state = [2, 6, 5, 0, 8, 7, 4, 3, 1]
        self.current_state = self.initial_state.copy()
        self.buttons = ["BFS", "DFS", "UCS", "Greedy", "IDDFS", "A*", "IDA*", 
                        "Hill", "Shill", "Randomhill", "SA", "BS", "AndOr", "GA", 
                        "NoObs", "BeliefAStar", "AC3", "Backtracking", "QLearning", 
                        "Reinforce", "GlobalBeam", "Bidirectional", "Reset", "Random"]
    
    def draw_board(self):
        pygame.draw.rect(WINDOW, COLORS['GREY'], 
                        (PADDING - 5, PADDING - 5, TILE_SIZE * 3 + 10, TILE_SIZE * 3 + 10), 
                        border_radius=10)
        pygame.draw.rect(WINDOW, COLORS['DARK_GREY'], 
                        (PADDING - 5, PADDING - 5, TILE_SIZE * 3 + 10, TILE_SIZE * 3 + 10), 
                        2, border_radius=10)
        
        for i, num in enumerate(self.current_state):
            x, y = (i % 3) * TILE_SIZE + PADDING, (i // 3) * TILE_SIZE + PADDING
            if num != 0:
                pygame.draw.rect(WINDOW, COLORS['DARK_BLUE'], 
                               (x + 3, y + 3, TILE_SIZE - 2, TILE_SIZE - 2), border_radius=5)
                pygame.draw.rect(WINDOW, COLORS['BLUE'], 
                               (x, y, TILE_SIZE - 2, TILE_SIZE - 2), border_radius=5)
                pygame.draw.rect(WINDOW, COLORS['DARK_GREY'], 
                               (x, y, TILE_SIZE - 2, TILE_SIZE - 2), 2, border_radius=5)
                text = FONTS['tile'].render(str(num), True, COLORS['WHITE'])
                WINDOW.blit(text, text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2)))

    def draw_buttons(self, hovered: Optional[str] = None):
        btn_width, btn_height = 180, 40
        spacing = 10
        mid_idx = len(self.buttons) // 2
        left_buttons = self.buttons[:mid_idx]
        right_buttons = self.buttons[mid_idx:]
        
        start_x_left = WINDOW_SIZE[0] - 2 * btn_width - 40
        start_y_left = (WINDOW_SIZE[1] - (len(left_buttons) * (btn_height + spacing))) // 2
        for i, text in enumerate(left_buttons):
            x = start_x_left
            y = start_y_left + i * (btn_height + spacing)
            color = COLORS['BUTTON_HOVER'] if hovered == text else COLORS['LIGHT_BLUE']
            pygame.draw.rect(WINDOW, COLORS['DARK_BLUE'], (x + 3, y + 3, btn_width, btn_height), border_radius=8)
            pygame.draw.rect(WINDOW, color, (x, y, btn_width, btn_height), border_radius=8)
            pygame.draw.rect(WINDOW, COLORS['DARK_GREY'], (x, y, btn_width, btn_height), 2, border_radius=8)
            btn_text = FONTS['button'].render(text, True, COLORS['WHITE'])
            WINDOW.blit(btn_text, btn_text.get_rect(center=(x + btn_width // 2, y + btn_height // 2)))

        start_x_right = WINDOW_SIZE[0] - btn_width - 30
        start_y_right = (WINDOW_SIZE[1] - (len(right_buttons) * (btn_height + spacing))) // 2
        for i, text in enumerate(right_buttons):
            x = start_x_right
            y = start_y_right + i * (btn_height + spacing)
            color = COLORS['BUTTON_HOVER'] if hovered == text else COLORS['LIGHT_BLUE']
            pygame.draw.rect(WINDOW, COLORS['DARK_BLUE'], (x + 3, y + 3, btn_width, btn_height), border_radius=8)
            pygame.draw.rect(WINDOW, color, (x, y, btn_width, btn_height), border_radius=8)
            pygame.draw.rect(WINDOW, COLORS['DARK_GREY'], (x, y, btn_width, btn_height), 2, border_radius=8)
            btn_text = FONTS['button'].render(text, True, COLORS['WHITE'])
            WINDOW.blit(btn_text, btn_text.get_rect(center=(x + btn_width // 2, y + btn_height // 2)))

    def get_button_at_pos(self, pos: Tuple[int, int]) -> Tuple[Optional[Callable], Optional[str]]:
        btn_width, btn_height = 140, 40
        spacing = 10
        mid_idx = len(self.buttons) // 2
        left_buttons = self.buttons[:mid_idx]
        right_buttons = self.buttons[mid_idx:]
        
        algo_map = {
            "bfs": "bfs",
            "dfs": "dfs",
            "ucs": "ucs",
            "greedy": "greedy",
            "iddfs": "iddfs",
            "a*": "astar",
            "ida*": "ida_star",
            "hill": "hill",
            "shill": "shill",
            "randomhill": "randomhill",
            "sa": "sa",
            "bs": "bs",
            "andor": "andor",
            "ga": "ga",
            "noobs": "noobs",
            "beliefastar": "belief_astar",
            "ac3": "ac3",
            "backtracking": "backtracking",
            "qlearning": "qlearning",
            "reinforce": "reinforce",
            "globalbeam": "global_beam",
            "bidirectional": "bidirectional",
            "tabu": "tabu",
            "reset": "reset",
            "random": "random"
        }
        
        start_x_left = WINDOW_SIZE[0] - 2 * btn_width - 40
        start_y_left = (WINDOW_SIZE[1] - (len(left_buttons) * (btn_height + spacing))) // 2
        for i, btn_text in enumerate(left_buttons):
            rect = pygame.Rect(start_x_left, start_y_left + i * (btn_height + spacing), btn_width, btn_height)
            if rect.collidepoint(pos):
                if btn_text.lower() == "reset":
                    return "reset", btn_text
                if btn_text.lower() == "random":
                    return "random", btn_text
                method_name = algo_map[btn_text.lower()]
                return (lambda x: self.solver.solve(x, method_name)), btn_text

        start_x_right = WINDOW_SIZE[0] - btn_width - 30
        start_y_right = (WINDOW_SIZE[1] - (len(right_buttons) * (btn_height + spacing))) // 2
        for i, btn_text in enumerate(right_buttons):
            rect = pygame.Rect(start_x_right, start_y_right + i * (btn_height + spacing), btn_width, btn_height)
            if rect.collidepoint(pos):
                if btn_text.lower() == "reset":
                    return "reset", btn_text
                if btn_text.lower() == "random":
                    return "random", btn_text
                method_name = algo_map[btn_text.lower()]
                return (lambda x: self.solver.solve(x, method_name)), btn_text
        
        return None, None

    def generate_random_solvable_state(self) -> List[int]:
        while True:
            state = list(range(9))
            random.shuffle(state)
            if self.solver.is_solvable(state):
                return state

def main():
    ui = PuzzleUI()
    running = True
    solving = False
    solution = None
    metrics = None
    step = 0
    step_count = 0
    selected_algorithm = None

    while running:
        WINDOW.fill(COLORS['WHITE'])
        mouse_pos = pygame.mouse.get_pos()
        _, hovered = ui.get_button_at_pos(mouse_pos)

        ui.draw_board()
        ui.draw_buttons(hovered)
        status = "Solving..." if solving else "Waiting ..."
        WINDOW.blit(FONTS['status'].render(status, True, COLORS['BLACK']), (PADDING + 480, WINDOW_SIZE[1] - 30))
        WINDOW.blit(FONTS['status'].render(f"Steps: {step_count}", True, COLORS['BLACK']), (PADDING, WINDOW_SIZE[1] - 30))
        
        if metrics and not solving:
            if metrics.get("error"):
                WINDOW.blit(FONTS['metrics'].render(metrics["error"], True, COLORS['RED']), 
                           (PADDING + 150, WINDOW_SIZE[1] - 60))
            else:
                WINDOW.blit(FONTS['metrics'].render(f"Time: {metrics['time']:.2f}s", True, COLORS['BLACK']), 
                           (PADDING + 150, WINDOW_SIZE[1] - 30))
                WINDOW.blit(FONTS['metrics'].render(f"Nodes: {metrics['nodes']}", True, COLORS['BLACK']), 
                           (PADDING + 300, WINDOW_SIZE[1] - 30))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not solving:
                    algo, btn_text = ui.get_button_at_pos(event.pos)
                    if algo:
                        if algo == "reset":
                            ui.current_state = ui.initial_state.copy()
                            solving = False
                            solution = metrics = None
                            step = step_count = 0
                        elif algo == "random":
                            ui.current_state = ui.generate_random_solvable_state()
                            ui.initial_state = ui.current_state.copy()
                            solving = False
                            solution = metrics = None
                            step = step_count = 0
                        else:
                            solving = True
                            selected_algorithm = btn_text
                            solution, metrics = algo(ui.current_state)
                            step = step_count = 0
                            if solution is None:
                                solving = False

        if solving and solution:
            if step < len(solution):
                zero_idx, move_idx = solution[step]
                ui.current_state[zero_idx], ui.current_state[move_idx] = \
                    ui.current_state[move_idx], ui.current_state[zero_idx]
                step += 1
                step_count += 1
                pygame.time.delay(100)
            else:
                solving = False

        clock.tick(120)

    pygame.quit()

if __name__ == "__main__":
    main()
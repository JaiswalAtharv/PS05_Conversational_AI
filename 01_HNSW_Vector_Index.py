"""PS05: Educational NumPy HNSW-style approximate nearest-neighbour index.

Required mission benchmark: 25,000 vectors with 768 dimensions, compared with
exact brute-force cosine search. This is an educational implementation, not a
replacement for a production library such as hnswlib or FAISS.
"""
from __future__ import annotations

import argparse
import heapq
import math
import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np


def _normalize(vector: np.ndarray, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Each vector must be one-dimensional.")
    if dim is not None and arr.shape[0] != dim:
        raise ValueError(f"Vector dimension {arr.shape[0]} does not match index dimension {dim}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vectors must contain only finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Zero vectors cannot be cosine-normalized.")
    return arr / norm


def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional.")
    if vectors.ndim != 2 or vectors.shape[1] != query.shape[0]:
        raise ValueError("vectors must be a 2D matrix with the same feature dimension as query.")
    q_norm = float(np.linalg.norm(query))
    v_norms = np.linalg.norm(vectors, axis=1)
    if q_norm <= 1e-12:
        raise ValueError("A zero query vector has undefined cosine similarity.")
    if np.any(v_norms <= 1e-12):
        raise ValueError("vectors contains a zero vector.")
    return (vectors @ query) / (v_norms * q_norm)


class HNSWIndex:
    """Simplified HNSW-style graph using cosine distance = 1 - cosine similarity.

    M is the upper-layer degree cap; layer 0 permits 2*M neighbours. Neighbour
    selection uses nearest-distance pruning rather than the full diversity heuristic
    found in mature HNSW implementations.
    """
    def __init__(self, dim: int = 768, M: int = 16, ef_construction: int = 64,
                 ef_search: int = 64, seed: int | None = 42):
        for name, value, minimum in (("dim", dim, 1), ("M", M, 2),
                                     ("ef_construction", ef_construction, 1),
                                     ("ef_search", ef_search, 1)):
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.dim = dim
        self.M = M
        self.M_max0 = 2 * M
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self.vectors: Dict[int, np.ndarray] = {}
        self.graph: Dict[int, Dict[int, Set[int]]] = {}
        self.node_levels: Dict[int, int] = {}
        self.entry_point: int | None = None
        self.max_level = -1

    def _get_random_level(self) -> int:
        u = max(self._rng.random(), 1e-12)
        return int(-math.log(u) * self.mL)

    def _dist(self, node_id: int, vector: np.ndarray) -> float:
        return 1.0 - max(-1.0, min(1.0, float(np.dot(self.vectors[node_id], vector))))

    def _dist_many(self, node_ids, vector: np.ndarray) -> np.ndarray:
        """Vectorized cosine distances for a small batch of unit-normalized nodes."""
        ids = list(node_ids)
        if not ids:
            return np.empty(0, dtype=np.float32)
        matrix = np.stack([self.vectors[node_id] for node_id in ids])
        similarities = matrix @ vector
        return 1.0 - np.clip(similarities, -1.0, 1.0)

    def _greedy_search(self, entry_id: int, query: np.ndarray, layer: int) -> int:
        current = entry_id
        current_dist = self._dist(current, query)
        while True:
            neighbors = list(self.graph.get(current, {}).get(layer, set()))
            if not neighbors:
                return current
            distances = self._dist_many(neighbors, query)
            best_index = int(np.argmin(distances))
            if float(distances[best_index]) < current_dist:
                current = neighbors[best_index]
                current_dist = float(distances[best_index])
            else:
                return current

    def search_layer(self, entry_points: List[int], query: np.ndarray,
                     ef: int, layer: int) -> List[Tuple[float, int]]:
        if ef < 1:
            raise ValueError("ef must be >= 1.")
        entries = [e for e in entry_points if e in self.vectors and layer in self.graph.get(e, {})]
        if not entries:
            return []
        visited = set(entries)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []  # max-heap encoded as negative distance
        for node_id in entries:
            distance = self._dist(node_id, query)
            heapq.heappush(candidates, (distance, node_id))
            heapq.heappush(results, (-distance, node_id))
        while candidates:
            candidate_dist, candidate_id = heapq.heappop(candidates)
            worst_dist = -results[0][0]
            if len(results) >= ef and candidate_dist > worst_dist:
                break
            neighbors = [neighbor for neighbor in self.graph.get(candidate_id, {}).get(layer, set())
                         if neighbor not in visited]
            if not neighbors:
                continue
            distances = self._dist_many(neighbors, query)
            for neighbor, distance_value in zip(neighbors, distances):
                visited.add(neighbor)
                distance = float(distance_value)
                worst_dist = -results[0][0]
                if len(results) < ef or distance < worst_dist:
                    heapq.heappush(candidates, (distance, neighbor))
                    heapq.heappush(results, (-distance, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-negative_distance, node_id) for negative_distance, node_id in results)

    def _prune_neighbors(self, node_id: int, layer: int) -> None:
        neighbors = self.graph[node_id].get(layer)
        if neighbors is None:
            return
        limit = self.M_max0 if layer == 0 else self.M
        if len(neighbors) <= limit:
            return
        ranked = list(neighbors)
        distances = self._dist_many(ranked, self.vectors[node_id])
        ranked = [ranked[i] for i in np.argsort(distances)]
        keep = set(ranked[:limit])
        for removed in neighbors - keep:
            self.graph.get(removed, {}).get(layer, set()).discard(node_id)
        self.graph[node_id][layer] = keep

    def insert(self, node_id: int, vector: np.ndarray) -> None:
        if node_id in self.vectors:
            raise ValueError(f"Duplicate node_id: {node_id}")
        normalized = _normalize(vector, self.dim)
        target_level = self._get_random_level()
        self.vectors[node_id] = normalized
        self.node_levels[node_id] = target_level
        self.graph[node_id] = {level: set() for level in range(target_level + 1)}
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = target_level
            return
        entry = self.entry_point
        for layer in range(self.max_level, target_level, -1):
            entry = self._greedy_search(entry, normalized, layer)
        for layer in range(min(target_level, self.max_level), -1, -1):
            candidates = self.search_layer([entry], normalized, self.ef_construction, layer)
            limit = self.M_max0 if layer == 0 else self.M
            selected = [other for _, other in candidates[:limit]]
            for neighbor in selected:
                self.graph[node_id][layer].add(neighbor)
                self.graph[neighbor][layer].add(node_id)
                self._prune_neighbors(neighbor, layer)
            if candidates:
                entry = candidates[0][1]
        if target_level > self.max_level:
            self.entry_point = node_id
            self.max_level = target_level

    def query(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if self.entry_point is None:
            return []
        query = _normalize(query_vector, self.dim)
        current = self.entry_point
        for layer in range(self.max_level, 0, -1):
            current = self._greedy_search(current, query, layer)
        candidates = self.search_layer([current], query, max(self.ef_search, top_k), 0)
        return [(node_id, 1.0 - distance) for distance, node_id in candidates[:top_k]]

    def validate_graph(self) -> None:
        """Validate layer membership, degree limits, and neighbour references."""
        for node_id, layers in self.graph.items():
            assert node_id in self.vectors
            assert set(layers) == set(range(self.node_levels[node_id] + 1))
            for layer, neighbors in layers.items():
                limit = self.M_max0 if layer == 0 else self.M
                assert len(neighbors) <= limit, f"Node {node_id}, layer {layer}: degree {len(neighbors)} > {limit}"
                for neighbor in neighbors:
                    assert neighbor != node_id, "Self-links are not allowed."
                    assert neighbor in self.vectors
                    assert layer in self.graph[neighbor], "Neighbour does not exist on this layer."


def exact_top_k(data_unit: np.ndarray, query: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Exact cosine top-k IDs for already unit-normalized data."""
    query_unit = _normalize(query, data_unit.shape[1])
    scores = data_unit @ query_unit
    return np.argsort(scores)[::-1][:top_k]


def run_latency_benchmark(N: int = 25_000, dim: int = 768,
                          num_queries: int = 10, seed: int = 42,
                          M: int = 16, ef_construction: int = 64,
                          ef_search: int = 64) -> Dict[str, float]:
    """Benchmark HNSW against exact search using the mission's required dimensions."""
    if N < 2 or dim < 1 or num_queries < 1:
        raise ValueError("N must be >= 2; dim and num_queries must be positive.")
    print(f"=== HNSW vs exact flat search (N={N:,}, dim={dim}, queries={num_queries}) ===", flush=True)
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N, dim), dtype=np.float32)
    data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)
    index = HNSWIndex(dim=dim, M=M, ef_construction=ef_construction,
                      ef_search=ef_search, seed=seed)
    start = time.perf_counter()
    for node_id, vector in enumerate(data_norm):
        index.insert(node_id, vector)
        if (node_id + 1) % 2500 == 0:
            print(f"Inserted {node_id + 1:,}/{N:,} vectors", flush=True)
    build_seconds = time.perf_counter() - start
    index.validate_graph()
    hnsw_ms, flat_ms, recalls = [], [], []
    for query_index in range(num_queries):
        query = rng.standard_normal(dim, dtype=np.float32)
        start = time.perf_counter()
        approx = index.query(query, top_k=10)
        hnsw_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        exact_ids = exact_top_k(data_norm, query, top_k=10)
        flat_ms.append((time.perf_counter() - start) * 1000)
        approx_ids = {node_id for node_id, _ in approx}
        recalls.append(len(approx_ids.intersection(set(map(int, exact_ids)))) / 10.0)
        print(f"Query {query_index + 1}/{num_queries}: recall@10={recalls[-1]:.3f}", flush=True)
    mean_hnsw, mean_flat = float(np.mean(hnsw_ms)), float(np.mean(flat_ms))
    result = {
        "N_indexed": int(N), "dim": int(dim), "num_queries": int(num_queries),
        "build_seconds": float(build_seconds), "hnsw_mean_latency_ms": mean_hnsw,
        "flat_mean_latency_ms": mean_flat,
        "speedup": mean_flat / max(mean_hnsw, 1e-12),
        "recall_at_10": float(np.mean(recalls)),
        "M": int(M), "ef_construction": int(ef_construction), "ef_search": int(ef_search),
    }
    print("\n=== Benchmark summary ===")
    print(f"Indexed vectors:             {N:,}")
    print(f"Dimensions:                  {dim}")
    print(f"Build time:                  {build_seconds:.2f} s")
    print(f"Mean HNSW query latency:     {mean_hnsw:.3f} ms")
    print(f"Mean exact flat latency:     {mean_flat:.3f} ms")
    print(f"Measured speedup:            {result['speedup']:.3f}x")
    print(f"Mean recall@10:              {result['recall_at_10']:.3f}")
    print("Measurements are specific to this machine and configuration.")
    return result


def run_smoke_tests() -> None:
    index = HNSWIndex(dim=3, M=4, ef_construction=12, ef_search=10, seed=7)
    vectors = [np.array([1., 0., 0.]), np.array([.9, .1, 0.]),
               np.array([0., 1., 0.]), np.array([0., 0., 1.]),
               np.array([-1., 0., 0.])]
    for i, vector in enumerate(vectors):
        index.insert(i, vector)
    index.validate_graph()
    results = index.query(np.array([1., 0., 0.]), top_k=3)
    assert results[0][0] in (0, 1)
    assert len(results) == 3
    # Check returned similarities are sorted descending and bounded.
    assert all(results[i][1] >= results[i + 1][1] for i in range(len(results) - 1))
    assert all(-1.00001 <= similarity <= 1.00001 for _, similarity in results)
    try:
        index.insert(0, vectors[0])
        raise AssertionError("Duplicate IDs should be rejected.")
    except ValueError:
        pass
    for invalid in (np.zeros(3), np.array([np.nan, 0., 1.]), np.ones(4)):
        try:
            index.query(invalid)
            raise AssertionError("Invalid query should be rejected.")
        except ValueError:
            pass
    # Verify exact-search helper independently on a tiny deterministic dataset.
    unit = np.vstack([_normalize(v) for v in vectors])
    exact = exact_top_k(unit, np.array([1., 0., 0.]), top_k=2)
    assert int(exact[0]) == 0
    print("Smoke tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run full benchmark")
    parser.add_argument("--n", type=int, default=25_000, help="number of vectors (default: 25000)")
    parser.add_argument("--dim", type=int, default=768, help="vector dimension (default: 768)")
    parser.add_argument("--queries", type=int, default=10, help="number of benchmark queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=64)
    args = parser.parse_args()
    run_smoke_tests()
    if args.benchmark:
        run_latency_benchmark(args.n, args.dim, args.queries, args.seed,
                              args.m, args.ef_construction, args.ef_search)
    else:
        print("Full benchmark not run. Use --benchmark to run the configured 25,000 x 768 test.")

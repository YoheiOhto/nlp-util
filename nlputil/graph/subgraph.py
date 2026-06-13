"""
統一BFS・サブグラフ構築。

exp32 (nhop_subgraph_stats) / exp37 (nhop_stats_overall) のBFSロジックを統合。
"""

from itertools import combinations

import pandas as pd


def nhop_bfs(
    seed:          str,
    adj:           dict[str, dict],
    n_hops:        int,
    max_neighbors: int | None = None,
    min_weight:    float | None = None,
) -> set[str]:
    """
    seed から n-hop BFS で到達できるノード集合を返す。

    Args:
        seed          : 起点ノード
        adj           : 隣接辞書 {node: {neighbor: weight}}
        n_hops        : BFS の深さ
        max_neighbors : 各ノードから展開する近傍の上限 (weight 降順, None=制限なし)
        min_weight    : weight がこの値未満の近傍はスキップ (None=制限なし)

    Returns:
        到達ノード集合 (seed を含む)
    """
    if seed not in adj:
        return set()

    def _neighbors(node: str) -> list[str]:
        nbrs = adj.get(node, {})
        if min_weight is not None:
            nbrs = {k: v for k, v in nbrs.items() if v >= min_weight}
        if max_neighbors is not None:
            nbrs = dict(sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[:max_neighbors])
        return list(nbrs.keys())

    visited:  set[str] = {seed}
    frontier: set[str] = {seed}
    for _ in range(n_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for nbr in _neighbors(node):
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited  |= next_frontier
        frontier  = next_frontier
        if not frontier:
            break
    return visited


def build_subgraph(
    seed:          str,
    adj:           dict[str, dict],
    n_hops:        int = 1,
    max_neighbors: int | None = None,
    min_weight:    float | None = None,
    weight_col:    str = "weight",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    seed を起点とした n-hop サブグラフを構築する。

    Args:
        seed          : 起点ノード
        adj           : 隣接辞書 {node: {neighbor: weight}}
        n_hops        : BFS の深さ
        max_neighbors : 各ノードから展開する近傍の上限
        min_weight    : weight 下限フィルタ
        weight_col    : edges_df / nodes_df に出力する weight カラム名。
                        ネットワーク固有の名前 (combined_score / n_interactions 等) を指定可。

    Returns:
        nodes_df : DataFrame(node, is_seed, degree_in_subgraph, <weight_col>_max)
                   <weight_col>_max = そのノードの最大エッジ重みへのリンク強度
        edges_df : DataFrame(node_a, node_b, <weight_col>)
                   node_a < node_b (アルファベット順)
    """
    nodes = nhop_bfs(seed, adj, n_hops, max_neighbors, min_weight)
    if not nodes:
        return (
            pd.DataFrame(columns=["node", "is_seed", "degree_in_subgraph", f"{weight_col}_max"]),
            pd.DataFrame(columns=["node_a", "node_b", weight_col]),
        )

    edge_rows: list[dict] = []
    for a, b in combinations(sorted(nodes), 2):
        w = adj.get(a, {}).get(b) or adj.get(b, {}).get(a)
        if w is not None:
            edge_rows.append({"node_a": a, "node_b": b, weight_col: float(w)})

    edges_df = (
        pd.DataFrame(edge_rows)
        if edge_rows
        else pd.DataFrame(columns=["node_a", "node_b", weight_col])
    )

    degree:     dict[str, int]   = {n: 0   for n in nodes}
    max_weight: dict[str, float] = {n: 0.0 for n in nodes}
    for row in edge_rows:
        w = row[weight_col]
        degree[row["node_a"]] += 1
        degree[row["node_b"]] += 1
        max_weight[row["node_a"]] = max(max_weight[row["node_a"]], w)
        max_weight[row["node_b"]] = max(max_weight[row["node_b"]], w)

    nodes_df = pd.DataFrame([
        {
            "node":                n,
            "is_seed":             n == seed,
            "degree_in_subgraph":  degree[n],
            f"{weight_col}_max":   max_weight[n],
        }
        for n in sorted(nodes)
    ])
    return nodes_df, edges_df


def nhop_scale_stats(
    seed:          str,
    adj:           dict[str, dict],
    max_hops:      int = 5,
    max_neighbors: int | None = None,
    min_weight:    float | None = None,
) -> list[dict]:
    """
    hop 別サブグラフ規模 (n_nodes, n_gold_edges, n_pairs) を返す。

    exp32 の nhop_subgraph_stats / exp37 の nhop_stats_overall を統合。

    Args:
        seed          : 起点ノード
        adj           : 隣接辞書
        max_hops      : 最大 hop 数
        max_neighbors : 各ノードから展開する近傍の上限
        min_weight    : weight 下限フィルタ

    Returns:
        list[{"hop": int, "n_nodes": int, "n_gold_edges": int, "n_pairs": int}]
    """
    if seed not in adj:
        return [{"hop": h, "n_nodes": 0, "n_gold_edges": 0, "n_pairs": 0}
                for h in range(1, max_hops + 1)]

    def _neighbors(node: str) -> list[str]:
        nbrs = adj.get(node, {})
        if min_weight is not None:
            nbrs = {k: v for k, v in nbrs.items() if v >= min_weight}
        if max_neighbors is not None:
            nbrs = dict(sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[:max_neighbors])
        return list(nbrs.keys())

    visited:  set[str] = {seed}
    frontier: set[str] = {seed}
    results: list[dict] = []

    for hop in range(1, max_hops + 1):
        next_frontier: set[str] = set()
        for node in frontier:
            for nbr in _neighbors(node):
                if nbr not in visited:
                    next_frontier.add(nbr)
        visited  |= next_frontier
        frontier  = next_frontier

        n_nodes      = len(visited)
        n_pairs      = n_nodes * (n_nodes - 1) // 2
        n_gold_edges = sum(
            1 for a in visited
            for b in adj.get(a, {})
            if b in visited and a < b
        )
        results.append({"hop": hop, "n_nodes": n_nodes,
                        "n_gold_edges": n_gold_edges, "n_pairs": n_pairs})

        if not frontier:
            for h in range(hop + 1, max_hops + 1):
                results.append({"hop": h, **{k: v for k, v in results[-1].items() if k != "hop"}})
            break

    return results

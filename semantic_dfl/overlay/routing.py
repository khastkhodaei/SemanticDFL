"""
Multi-Hop Underlay Packet Routing, Forwarding Auditing, and Path Trace Ledger.

Implements realistic multi-hop packet transmission when peer communication traverses
underlay physical graph constraints G = (V, E):
- Computes shortest network paths via Dijkstra / BFS.
- Simulates and logs hop-by-hop relay forwarding.
- Records structured path traces and transaction audit ledgers.
"""

from __future__ import annotations

import collections
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx


class MultiHopRouter:
    """
    Simulates multi-hop packet routing and forwarding audits over physical network topologies.
    """

    def __init__(
        self,
        adjacency_graph: Optional[Dict[int, List[int]]] = None,
        default_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.adjacency_graph = adjacency_graph or {}
        self.default_logger = default_logger
        self.graph = nx.Graph()

        if adjacency_graph:
            for u, neighbors in adjacency_graph.items():
                self.graph.add_node(u)
                for v in neighbors:
                    self.graph.add_edge(u, v)

        self._shortest_paths: Dict[Tuple[int, int], List[int]] = {}
        self._precompute_paths()

    def set_topology(self, adjacency_graph: Dict[int, List[int]]) -> None:
        """Updates the underlying physical topology and recomputes paths."""
        self.adjacency_graph = adjacency_graph
        self.graph = nx.Graph()
        for u, neighbors in adjacency_graph.items():
            self.graph.add_node(u)
            for v in neighbors:
                self.graph.add_edge(u, v)
        self._shortest_paths.clear()
        self._precompute_paths()

    def _precompute_paths(self) -> None:
        """Precomputes all-pairs shortest paths using NetworkX BFS/Dijkstra."""
        if not self.graph.nodes():
            return
        
        # Handle connected components
        for source in self.graph.nodes():
            try:
                paths = nx.single_source_shortest_path(self.graph, source)
                for target, path in paths.items():
                    self._shortest_paths[(source, target)] = path
            except Exception:
                pass

    def get_shortest_path(self, src: int, dst: int) -> List[int]:
        """
        Returns the sequence of node IDs along the shortest path from src to dst.
        If src == dst, returns [src].
        If no path exists, returns fallback direct hop [src, dst].
        """
        if src == dst:
            return [src]

        if (src, dst) in self._shortest_paths:
            return self._shortest_paths[(src, dst)]

        # Fallback if uncomputed or disconnected
        if self.graph.has_node(src) and self.graph.has_node(dst):
            try:
                path = nx.shortest_path(self.graph, src, dst)
                self._shortest_paths[(src, dst)] = path
                return path
            except nx.NetworkXNoPath:
                return [src, dst]
        return [src, dst]

    def get_hop_count(self, src: int, dst: int) -> int:
        """Returns the number of network hops (edges) from src to dst."""
        path = self.get_shortest_path(src, dst)
        return max(0, len(path) - 1)

    def route_packet(
        self,
        src: int,
        dst: int,
        packet_type: str = "MODEL_PULL",
        payload_bytes: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """
        Simulates forwarding a packet along its shortest network path and logs each hop transit.
        
        Args:
            src: Originating peer ID.
            dst: Target destination peer ID.
            packet_type: 'MODEL_PULL', 'SEARCH_QUERY', 'SIGNATURE_EXCHANGE', 'TREE_SPLIT'.
            payload_bytes: Size of packet in bytes.
            logger: Optional custom logger sink.
            
        Returns:
            Dictionary containing full route trace record.
        """
        active_logger = logger or self.default_logger
        path = self.get_shortest_path(src, dst)
        total_hops = max(1, len(path) - 1)

        hop_records = []
        if len(path) > 1:
            for step in range(len(path) - 1):
                curr_node = path[step]
                next_node = path[step + 1]
                hop_num = step + 1

                log_msg = (
                    f"[Node {curr_node}] Forwarded {packet_type}(src={src}, dest={dst}) "
                    f"-> next_hop=[Node {next_node}] (Hop {hop_num}/{total_hops})"
                )
                if active_logger:
                    active_logger.info(log_msg)

                hop_records.append({
                    "hop": hop_num,
                    "from_node": curr_node,
                    "to_node": next_node,
                })

        return {
            "src": src,
            "dst": dst,
            "packet_type": packet_type,
            "payload_bytes": payload_bytes,
            "path": path,
            "hop_count": total_hops if src != dst else 0,
            "hops": hop_records,
            "timestamp": datetime.now().isoformat(),
        }

    def export_routing_audit(
        self,
        round_num: int,
        transactions: List[Dict[str, Any]],
        output_dir: str = "results",
    ) -> str:
        """
        Exports structured routing audit matrix for a communication round.
        
        File: {output_dir}/routing_audit_round_{round_num}.json
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"routing_audit_round_{round_num}.json")

        total_hops = sum(t.get("hop_count", 0) for t in transactions)
        avg_hops = total_hops / float(max(1, len(transactions)))
        max_hops = max((t.get("hop_count", 0) for t in transactions), default=0)

        serializable = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_transactions": len(transactions),
                "total_hops": total_hops,
                "avg_hops_per_tx": float(avg_hops),
                "max_hops_per_tx": int(max_hops),
            },
            "transactions": transactions,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        return filepath

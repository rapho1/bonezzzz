"""
DAG executor with per-node caching and a heavy-node gate.

Cache key for a node = hash(type + params + parent keys), so a node is
recomputed only when its own params or anything upstream changes.

Heavy nodes (pose_estimation) are only executed when allow_heavy=True or a
disk cache already exists; otherwise the node reports status "needs_run" so the
frontend can prompt the user to press Run.
"""
import hashlib
import json
import os
from collections import defaultdict

from engine.nodes import run_node, node_info, HEAVY_TYPES, CACHE_DIR


def _node_key(node: dict, parent_keys: list[str]) -> str:
    payload = json.dumps(
        {"t": node["type"], "p": node.get("params", {}), "parents": parent_keys},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class Engine:
    def __init__(self):
        self._cache: dict[str, dict] = {}   # key -> result

    def execute(self, graph: dict, target_id: str, allow_heavy: bool = False) -> dict:
        nodes = {n["id"]: n for n in graph["nodes"]}
        parents = defaultdict(list)
        for e in graph.get("edges", []):
            parents[e["target"]].append(e["source"])

        statuses: dict[str, dict] = {}
        keys: dict[str, str] = {}

        def compute(nid: str):
            if nid not in nodes:
                raise ValueError(f"Unknown node id: {nid}")
            node = nodes[nid]

            parent_results = []
            parent_keys = []
            for pid in parents[nid]:
                res = compute(pid)
                if res is None:
                    statuses.setdefault(nid, {"status": "blocked",
                                              "info": "waiting on upstream"})
                    return None
                parent_results.append(res)
                parent_keys.append(keys[pid])

            key = _node_key(node, parent_keys)
            keys[nid] = key

            if key in self._cache:
                res = self._cache[key]
                statuses[nid] = {"status": "cached", "kind": res["kind"],
                                 "info": node_info(res)}
                return res

            ntype = node["type"]
            heavy = ntype in HEAVY_TYPES
            disk_cached = os.path.exists(
                os.path.join(CACHE_DIR, f"pose_{key}.json"))
            if heavy and not disk_cached and not allow_heavy:
                statuses[nid] = {"status": "needs_run", "info": "press Run"}
                return None

            statuses[nid] = {"status": "running", "info": "..."}
            res = run_node(ntype, node.get("params", {}), parent_results, key)
            self._cache[key] = res
            statuses[nid] = {"status": "done", "kind": res["kind"],
                             "info": node_info(res)}
            return res

        target_result = compute(target_id)

        return {
            "target": target_id,
            "statuses": statuses,
            "result": self._summarize(target_result),
        }

    def clear_cache(self) -> None:
        """Drop the in-memory result cache (per-process; the heavy-node disk
        cache under CACHE_DIR is cleared separately by the caller)."""
        self._cache.clear()

    def get_result(self, graph: dict, target_id: str):
        """Re-resolve the cached result object for a node (no recompute)."""
        nodes = {n["id"]: n for n in graph["nodes"]}
        parents = defaultdict(list)
        for e in graph.get("edges", []):
            parents[e["target"]].append(e["source"])
        keys: dict[str, str] = {}

        def resolve(nid):
            pkeys = []
            for pid in parents[nid]:
                if resolve(pid) is None:
                    return None
                pkeys.append(keys[pid])
            key = _node_key(nodes[nid], pkeys)
            keys[nid] = key
            return self._cache.get(key)

        return resolve(target_id)

    @staticmethod
    def _summarize(result):
        if result is None:
            return None
        out = {"kind": result["kind"], "fps": result.get("fps")}
        if result["kind"] == "bvh":
            out["bvh_len"] = len(result["bvh"])
        return out

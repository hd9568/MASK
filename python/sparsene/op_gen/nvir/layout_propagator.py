from python.sparsene.op_gen.nvir.codegen import (
    NvOp,
)
from typing import Set, Dict
from sparsene.op_gen.nvir.dtc_ops import (
    dtc_pipeline,
)
import datetime


def traverse_nvop(nvop: NvOp) -> Dict:
    ops = {}
    visited = set()

    def _traverse(nvop: NvOp, ops: Dict, visited: Set[NvOp]) -> None:
        if nvop in visited:
            return
        visited.add(nvop)
        ops[nvop.name] = {}
        for inp in nvop.inputs:
            if isinstance(inp.source, tuple):
                if inp.source[0] not in visited:
                    new_ops = {}
                    _traverse(inp.source[0], new_ops, visited)
                    if new_ops:
                        ops[nvop.name].setdefault("inputs", []).append(new_ops)
        for out in nvop.outputs:
            if out.owning:
                for consumer in out.consumers:
                    new_ops = {}
                    _traverse(consumer, new_ops, visited)
                    if new_ops:
                        ops[nvop.name].setdefault("outputs", []).append(new_ops)
            else:  # non-owning
                if isinstance(out.source, tuple):
                    new_ops = {}
                    if out.source[0] not in visited:
                        _traverse(out.source[0], new_ops, visited)
                    if new_ops:
                        ops[nvop.name].setdefault("outputs", []).append(new_ops)

    _traverse(nvop, ops, visited)
    return ops


if __name__ == "__main__":
    ops = traverse_nvop(dtc_pipeline.find_op_by_name("R2gCValStoreOp"))

    def custom_serializer(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif hasattr(obj, "to_json"):
            return obj.to_json()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    import json

    print(json.dumps(ops, indent=2, default=custom_serializer))

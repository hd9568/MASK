"""Export saved agent Format DSL JSON files as importable Python modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pprint

from sparsene.formats.custom.loader import module_stem
from sparsene.formats.dsl import validate_format_spec


MODULE_TEMPLATE = '''"""Generated from {source}; do not edit the embedded agent proposal."""

from sparsene.formats.dsl import build_format_from_spec

FORMAT_SPEC = {spec}
FORMAT = build_format_from_spec(FORMAT_SPEC)
PARAMS = FORMAT_SPEC["params"]
PROVENANCE = FORMAT_SPEC["provenance"]
'''


def export_run(run_dir: Path, output_dir: Path) -> list[Path]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_id = run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index: dict[str, str] = {}
    for dataset in manifest["datasets"]:
        source = run_dir / "formats" / f"{dataset}.format.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        core = {
            "dataset": dataset,
            "format_name": payload["format_name"],
            "levels": payload["levels"],
            "atomic": payload["atomic"],
            "params": payload["params"],
            "rationale": payload.get("rationale_cn") or payload.get("rationale", ""),
            "provenance": {
                "run_id": run_id,
                "source": str(source.resolve()),
                "schema_version": payload.get("schema_version"),
                "request_id": payload.get("request_id"),
            },
        }
        validation = validate_format_spec(core)
        if validation["status"] != "valid":
            raise ValueError(f"{dataset}: {validation['errors']}")
        stem = module_stem(dataset)
        if stem in index.values():
            raise ValueError(f"Module-name collision for dataset {dataset!r}: {stem}")
        index[dataset] = stem
        target = output_dir / f"{stem}.py"
        target.write_text(
            MODULE_TEMPLATE.format(
                source=source.resolve(),
                spec=pprint.pformat(core, width=100, sort_dicts=False),
            ),
            encoding="utf-8",
        )
        written.append(target)
    (output_dir / "index.json").write_text(
        json.dumps(
            {"run_id": run_id, "datasets": index},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("python/sparsene/formats/custom"),
    )
    args = parser.parse_args()
    paths = export_run(args.run_dir.resolve(), args.output_dir.resolve())
    print(f"exported {len(paths)} formats to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

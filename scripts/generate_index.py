#!/usr/bin/env python3
"""Generate an index.html comparing all training runs in the logs directory.

Scans logs/*/config.yaml and logs/*/metrics.jsonl, extracts key metrics,
and produces a sortable HTML table with links to individual run reports.

Usage:
    python scripts/generate_index.py
    python scripts/generate_index.py --log-dir ./logs
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def extract_run_info(run_dir: Path) -> dict | None:
    config_file = run_dir / "config.yaml"
    metrics_file = run_dir / "metrics.jsonl"
    report_file = run_dir / "report.html"

    if not config_file.exists():
        return None

    with open(config_file) as f:
        config = yaml.safe_load(f)

    info = {
        "dir": run_dir.name,
        "circuit": config.get("env_name", "?"),
        "total_timesteps": config.get("total_timesteps", "?"),
        "seed": config.get("seed", "?"),
        "num_envs": config.get("num_envs", 1),
        "run_name": config.get("run_name") or "-",
        "report": report_file.name if report_file.exists() else None,
        "final_hv": None,
        "final_eum": None,
        "final_card": None,
    }

    if metrics_file.exists():
        try:
            last_eval = None
            with open(metrics_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("event") == "evaluation":
                        last_eval = ev
            if last_eval:
                info["final_hv"] = last_eval.get("hypervolume")
                info["final_eum"] = last_eval.get("eum")
                info["final_card"] = last_eval.get("cardinality")
        except Exception:
            pass

    return info


SORT_JS = """
<script>
function sortTable(n) {
    const table = document.getElementById("runs-table");
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const asc = table.getAttribute("data-sort-col") != String(n);
    table.setAttribute("data-sort-col", asc ? String(n) : "");
    rows.sort((a, b) => {
        let va = a.cells[n].textContent.trim();
        let vb = b.cells[n].textContent.trim();
        let na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) { va = na; vb = nb; }
        if (va < vb) return asc ? -1 : 1;
        if (va > vb) return asc ? 1 : -1;
        return 0;
    });
    rows.forEach(r => tbody.appendChild(r));
}
</script>
"""

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }
h1 { border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
th { background: #1f77b4; color: white; padding: 10px 12px; text-align: left;
     cursor: pointer; user-select: none; font-size: 0.85em; text-transform: uppercase; }
th:hover { background: #1565c0; }
td { padding: 8px 12px; border-bottom: 1px solid #e9ecef; font-size: 0.9em; }
tr:hover { background: #f1f3f5; }
td a { color: #1f77b4; text-decoration: none; font-weight: 600; }
td a:hover { text-decoration: underline; }
.footer { color: #868e96; font-size: 0.8em; margin-top: 16px; }
"""


def render_index(infos: list[dict]) -> str:
    rows = ""
    for info in infos:
        hv = f'{info["final_hv"]:.4f}' if info["final_hv"] is not None else "-"
        eum = f'{info["final_eum"]:.4f}' if info["final_eum"] is not None else "-"
        card = str(info["final_card"]) if info["final_card"] is not None else "-"

        dir_cell = info["dir"]
        if info["report"]:
            dir_cell = f'<a href="{info["dir"]}/{info["report"]}">{info["dir"]}</a>'

        rows += f"""<tr>
            <td>{dir_cell}</td>
            <td>{info["circuit"]}</td>
            <td>{info["total_timesteps"]}</td>
            <td>{info["num_envs"]}</td>
            <td>{info["seed"]}</td>
            <td>{hv}</td>
            <td>{eum}</td>
            <td>{card}</td>
            <td>{info["run_name"]}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Runs — morl-ckt-sizing</title>
<style>{CSS}</style>
{SORT_JS}
</head>
<body>
<h1>Training Runs</h1>
<p>{len(infos)} run(s) in logs/</p>
<table id="runs-table">
<thead>
<tr>
    <th onclick="sortTable(0)">Run</th>
    <th onclick="sortTable(1)">Circuit</th>
    <th onclick="sortTable(2)">Steps</th>
    <th onclick="sortTable(3)">Envs</th>
    <th onclick="sortTable(4)">Seed</th>
    <th onclick="sortTable(5)">Final HV</th>
    <th onclick="sortTable(6)">Final EUM</th>
    <th onclick="sortTable(7)">Pareto</th>
    <th onclick="sortTable(8)">Name</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
<p class="footer">Generated by generate_index.py</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate experiment index page")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"),
                        help="Base log directory (default: ./logs)")
    args = parser.parse_args()

    log_dir = args.log_dir
    if not log_dir.is_dir():
        print(f"ERROR: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    infos = []
    for run_dir in sorted(log_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        info = extract_run_info(run_dir)
        if info:
            infos.append(info)

    if not infos:
        print(f"No runs found in {log_dir}")
        sys.exit(0)

    html = render_index(infos)
    output_path = log_dir / "index.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Index written to {output_path} ({len(infos)} runs)")


if __name__ == "__main__":
    main()

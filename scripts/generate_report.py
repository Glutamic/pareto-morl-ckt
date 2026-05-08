#!/usr/bin/env python3
"""Generate a self-contained HTML report for a single training run.

Reads config.yaml and metrics.jsonl from a run directory, produces
report.html with matplotlib charts embedded as base64 images.

Usage:
    python scripts/generate_report.py logs/20260503_170017_COMP
    python scripts/generate_report.py logs/20260503_170017_COMP -o /tmp/report.html
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def load_metrics(run_dir: Path) -> list[dict]:
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    events = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def load_config(run_dir: Path) -> dict:
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        return {}
    with open(config_file) as f:
        return yaml.safe_load(f)


def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def filter_events(events: list[dict], event_type: str) -> list[dict]:
    return [e for e in events if e.get("event") == event_type]


def make_overview_section(config: dict, events: list[dict]) -> str:
    eval_events = filter_events(events, "evaluation")
    circuit = config.get("env_name", "?")
    total_steps = config.get("total_timesteps", "?")
    seed = config.get("seed", "?")

    final_hv = "N/A"
    final_eum = "N/A"
    final_card = "N/A"
    if eval_events:
        last = eval_events[-1]
        final_hv = f"{last.get('hypervolume', 0):.4f}"
        final_eum = f"{last.get('eum', 0):.4f}"
        final_card = str(last.get("cardinality", 0))

    config_rows = ""
    for k, v in config.items():
        config_rows += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"

    return f"""
    <div class="section">
        <h2>Run Overview</h2>
        <div class="overview-grid">
            <div class="stat"><span class="label">Circuit</span><span class="value">{circuit}</span></div>
            <div class="stat"><span class="label">Total Steps</span><span class="value">{total_steps}</span></div>
            <div class="stat"><span class="label">Seed</span><span class="value">{seed}</span></div>
            <div class="stat"><span class="label">Final HV</span><span class="value">{final_hv}</span></div>
            <div class="stat"><span class="label">Final EUM</span><span class="value">{final_eum}</span></div>
            <div class="stat"><span class="label">Pareto Size</span><span class="value">{final_card}</span></div>
        </div>
        <details>
            <summary>Full Configuration</summary>
            <table class="config-table">{config_rows}</table>
        </details>
    </div>"""


def make_training_curves(events: list[dict]) -> str:
    tm = filter_events(events, "training_metrics")
    if not tm:
        return '<div class="section"><h2>Training Curves</h2><p>No training metrics recorded.</p></div>'

    steps = [e["global_step"] for e in tm]
    critic = [e["critic_loss"] for e in tm]
    policy = [e["policy_loss"] for e in tm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(steps, critic, color="#1f77b4", linewidth=1.0)
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Critic Loss")
    ax1.set_title("Critic Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, policy, color="#ff7f0e", linewidth=1.0)
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Policy Loss")
    ax2.set_title("Policy Loss")
    ax2.grid(True, alpha=0.3)

    img = fig_to_b64(fig)
    return f'<div class="section"><h2>Training Curves</h2><img src="{img}" alt="Training Curves"></div>'


def make_episode_returns(events: list[dict]) -> str:
    ep = filter_events(events, "episode")
    if not ep:
        return '<div class="section"><h2>Episode Returns</h2><p>No episode data recorded.</p></div>'

    has_step = all("global_step" in e for e in ep)
    if has_step:
        x = [e["global_step"] for e in ep]
        xlabel = "Global Step"
    else:
        x = list(range(len(ep)))
        xlabel = "Episode Index"

    returns = [e["return"] for e in ep]
    n_obj = len(returns[0]) if returns else 1

    n_plots = n_obj + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    axes = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i in range(n_obj):
        vals = [r[i] for r in returns]
        axes[i].plot(x, vals, color=colors[i % len(colors)], linewidth=0.8, alpha=0.8)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(f"Return [{i}]")
        axes[i].set_title(f"Objective {i}")
        axes[i].grid(True, alpha=0.3)

    scal = [e.get("scalarized_return", 0) for e in ep]
    axes[n_obj].plot(x, scal, color="green", linewidth=0.8, alpha=0.8)
    axes[n_obj].set_xlabel(xlabel)
    axes[n_obj].set_ylabel("Scalarized Return")
    axes[n_obj].set_title("Scalarized")
    axes[n_obj].grid(True, alpha=0.3)

    img = fig_to_b64(fig)
    return f'<div class="section"><h2>Episode Returns</h2><img src="{img}" alt="Episode Returns"></div>'


def make_mo_metrics(events: list[dict]) -> str:
    ev = filter_events(events, "evaluation")
    if not ev:
        return '<div class="section"><h2>Multi-Objective Metrics</h2><p>No evaluations recorded.</p></div>'

    steps = [e["global_step"] for e in ev]
    hv = [e.get("hypervolume", 0) for e in ev]
    eum = [e.get("eum", 0) for e in ev]
    card = [e.get("cardinality", 0) for e in ev]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(steps, hv, color="#1f77b4", marker="o", markersize=4)
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Hypervolume")
    ax1.set_title("Hypervolume")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, eum, color="#ff7f0e", marker="o", markersize=4)
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("EUM")
    ax2.set_title("Expected Utility Metric")
    ax2.grid(True, alpha=0.3)

    ax3.plot(steps, card, color="#2ca02c", marker="o", markersize=4)
    ax3.set_xlabel("Global Step")
    ax3.set_ylabel("Solutions")
    ax3.set_title("Pareto Cardinality")
    ax3.grid(True, alpha=0.3)

    img = fig_to_b64(fig)
    return f'<div class="section"><h2>Multi-Objective Metrics</h2><img src="{img}" alt="MO Metrics"></div>'


def make_pareto_front(events: list[dict]) -> str:
    sols = filter_events(events, "add_solution")
    if not sols:
        return '<div class="section"><h2>Pareto Front</h2><p>No solutions recorded.</p></div>'

    values = [s["value"] for s in sols]
    n_obj = len(values[0]) if values else 1

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        colors = range(len(xs))
        scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", alpha=0.8, s=30)
        ax.set_xlabel("Objective 0")
        ax.set_ylabel("Objective 1")
        ax.set_title("Pareto Front Solutions")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Iteration")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(n_obj):
            vals = [v[i] for v in values]
            ax.plot(vals, label=f"Obj {i}", alpha=0.8, linewidth=0.8)
        ax.set_xlabel("Solution Index")
        ax.set_ylabel("Value")
        ax.set_title("Solutions by Objective")
        ax.legend()
        ax.grid(True, alpha=0.3)

    img = fig_to_b64(fig)
    return f'<div class="section"><h2>Pareto Front</h2><img src="{img}" alt="Pareto Front"></div>'


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }
h1 { border-bottom: 2px solid #1f77b4; padding-bottom: 8px; }
h2 { color: #1f77b4; margin-top: 24px; }
.section { background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.section img { max-width: 100%; height: auto; }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.stat { background: #f1f3f5; padding: 12px; border-radius: 6px; }
.stat .label { display: block; font-size: 0.8em; color: #868e96; text-transform: uppercase; }
.stat .value { display: block; font-size: 1.3em; font-weight: 600; margin-top: 4px; }
.config-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.config-table td { padding: 4px 8px; border-bottom: 1px solid #e9ecef; }
.config-table td:first-child { font-weight: 600; white-space: nowrap; }
details { margin-top: 12px; }
summary { cursor: pointer; color: #1f77b4; font-weight: 600; }
"""


def render_html(run_dir: Path, config: dict, events: list[dict]) -> str:
    circuit = config.get("env_name", "?")
    title = f"{circuit} — {run_dir.name}"

    parts = [
        make_overview_section(config, events),
        make_training_curves(events),
        make_episode_returns(events),
        make_mo_metrics(events),
        make_pareto_front(events),
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p>Run directory: <code>{run_dir}</code></p>
{"".join(parts)}
<p style="color:#868e96;font-size:0.8em;margin-top:32px;">
Generated by generate_report.py</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for a training run")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("-o", "--output", type=Path, help="Output path (default: run_dir/report.html)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    config = load_config(run_dir)
    events = load_metrics(run_dir)

    if not events:
        print(f"WARNING: No metrics.jsonl found or empty in {run_dir}", file=sys.stderr)

    html = render_html(run_dir, config, events)

    output_path = args.output or (run_dir / "report.html")
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()

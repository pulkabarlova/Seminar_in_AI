"""Comparison tables for the planner benchmarks.

A table is a dict {"index": row names, "columns": column names, "values": rows},
as returned by `benchmark_batches`, `benchmark_first_map_fixed_points_reuse` and
`build_path_length_tables` in `utils.path_planning_utils`. NaN means the planner
failed on that case (no path found or an error).
"""

import math

ALGORITHMS = ("VG", "RRT_star", "PRM")


def format_benchmark_table(title, table, digits=6):
    header = ["map"] + table["columns"]
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row_name, row_values in zip(table["index"], table["values"]):
        formatted_row = [row_name]
        for value in row_values:
            if value != value:
                formatted_row.append("NaN")
            else:
                formatted_row.append(f"{value:.{digits}f}")
        lines.append("| " + " | ".join(formatted_row) + " |")

    return "\n".join(lines)


def summarize_benchmark_table(table):
    batch_summaries = []

    for col_idx, batch_name in enumerate(table["columns"]):
        values = [row[col_idx] for row in table["values"]]
        finite_values = [value for value in values if not math.isnan(value)]
        success_rate = len(finite_values) / len(values) if values else float("nan")

        batch_summaries.append({
            "batch": batch_name,
            "success_rate": success_rate,
            "mean": _safe_mean(finite_values),
            "median": _safe_median(finite_values),
            "min": min(finite_values) if finite_values else float("nan"),
            "max": max(finite_values) if finite_values else float("nan"),
            "count": len(finite_values),
            "total": len(values),
        })

    all_values = [
        value
        for row in table["values"]
        for value in row
        if not math.isnan(value)
    ]
    total_count = sum(len(row) for row in table["values"])

    overall = {
        "success_rate": len(all_values) / total_count if total_count else float("nan"),
        "mean": _safe_mean(all_values),
        "median": _safe_median(all_values),
        "min": min(all_values) if all_values else float("nan"),
        "max": max(all_values) if all_values else float("nan"),
        "count": len(all_values),
        "total": total_count,
    }

    return {
        "batches": batch_summaries,
        "overall": overall,
    }


def format_benchmark_summary(title, table, digits=6):
    summary = summarize_benchmark_table(table)
    lines = [f"### {title} summary", ""]
    lines.append("| batch | success_rate | median | mean | min | max | valid |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for row in summary["batches"]:
        lines.append(
            "| "
            + " | ".join([
                row["batch"],
                _format_ratio(row["success_rate"]),
                _format_number(row["median"], digits),
                _format_number(row["mean"], digits),
                _format_number(row["min"], digits),
                _format_number(row["max"], digits),
                f'{row["count"]}/{row["total"]}',
            ])
            + " |"
        )

    overall = summary["overall"]
    lines.extend([
        "",
        "Overall:",
        f'- success_rate: {_format_ratio(overall["success_rate"])}',
        f'- median: {_format_number(overall["median"], digits)}',
        f'- mean: {_format_number(overall["mean"], digits)}',
        f'- min: {_format_number(overall["min"], digits)}',
        f'- max: {_format_number(overall["max"], digits)}',
        f'- valid: {overall["count"]}/{overall["total"]}',
    ])
    return "\n".join(lines)


def build_path_length_comparison_table(path_length_tables):
    columns = ["VG", "RRT_star", "PRM", "RRT_star_vs_VG", "PRM_vs_VG"]
    index = path_length_tables["VG"]["columns"]
    values = []

    for col_idx, _batch_name in enumerate(index):
        vg_values = _finite_column_values(path_length_tables["VG"], col_idx)
        rrt_values = _finite_column_values(path_length_tables["RRT_star"], col_idx)
        prm_values = _finite_column_values(path_length_tables["PRM"], col_idx)

        vg_mean = _safe_mean(vg_values)
        rrt_mean = _safe_mean(rrt_values)
        prm_mean = _safe_mean(prm_values)

        values.append([
            vg_mean,
            rrt_mean,
            prm_mean,
            _safe_ratio(rrt_mean, vg_mean),
            _safe_ratio(prm_mean, vg_mean),
        ])

    return {
        "index": index,
        "columns": columns,
        "values": values,
    }


def algorithm_batch_means(tables, algorithms=ALGORITHMS):
    columns = tables[algorithms[0]]["columns"]
    means = {algorithm: [] for algorithm in algorithms}

    for algorithm in algorithms:
        table = tables[algorithm]
        for col_idx in range(len(columns)):
            means[algorithm].append(_safe_mean(_finite_column_values(table, col_idx)))

    return columns, means


def relabel_table_columns(table, labels):
    """Return a copy of `table` with new column names (e.g. obs_50, obs_80, ...)."""
    return {
        "index": table["index"],
        "columns": list(labels),
        "values": table["values"],
    }


def relabel_tables(tables, labels):
    """Apply `relabel_table_columns` to every algorithm's table."""
    return {name: relabel_table_columns(table, labels) for name, table in tables.items()}


def show_benchmark_tables(title, tables, algorithms=ALGORITHMS, summary=True, digits=6):
    """Display the full table (and optionally the summary) for each algorithm."""
    from IPython.display import Markdown, display

    for algorithm in algorithms:
        display(Markdown(format_benchmark_table(f"{title}: {algorithm}", tables[algorithm], digits)))
        if summary:
            display(Markdown(format_benchmark_summary(f"{title}: {algorithm}", tables[algorithm], digits)))


def _safe_mean(values):
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _safe_median(values):
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def _finite_column_values(table, col_idx):
    return [
        row[col_idx]
        for row in table["values"]
        if not math.isnan(row[col_idx])
    ]


def _safe_ratio(numerator, denominator):
    if math.isnan(numerator) or math.isnan(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return numerator / denominator


def _format_number(value, digits):
    if math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _format_ratio(value):
    if math.isnan(value):
        return "NaN"
    return f"{100 * value:.1f}%"

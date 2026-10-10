"""
report_template.py
Builds the printable batch report served by the dashboard's download button.

Kept separate from the application so that the wording can be reviewed by the hatchery manager
without touching application code, and so the usability study can evaluate the report as an
artefact in its own right.

No third-party dependency. Markdown tables are assembled directly rather than through
pandas.to_markdown, which requires `tabulate` and fails at the point of download if it is not
installed -- the worst possible moment, since the user has already done the work.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

__all__ = ["build_markdown_report", "markdown_table"]

_COLUMNS = [
    ("file", "Egg"),
    ("status", "Status"),
    ("pattern", "Morphology"),
    ("crack_probability", "Crack prob."),
    ("confidence", "Confidence"),
    ("physiological_pct", "Physiological %"),
    ("mechanical_pct", "Mechanical %"),
    ("needs_review", "Review"),
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.3f}" if v < 1 else f"{v:.0f}"
    return str(v)


def markdown_table(df: pd.DataFrame, columns=None) -> str:
    """Assemble a GitHub-flavoured Markdown table without external packages."""
    cols = columns or [(c, c) for c in df.columns]
    cols = [(k, h) for k, h in cols if k in df.columns]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r[k]) for k, _ in cols) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, rule, *body])


def build_markdown_report(df: pd.DataFrame, flock: dict, age_class: str,
                          operator: str = "", flock_id: str = "",
                          notes: str = "") -> str:
    ts = datetime.now().strftime("%d %B %Y, %H:%M")
    n = len(df)
    cracked = int((df["status"] == "Micro_cracked").sum())
    rate = cracked / n * 100 if n else 0.0
    flagged = int(df["needs_review"].sum()) if "needs_review" in df else 0
    pat = (df.loc[df["status"] == "Micro_cracked", "pattern"].value_counts()
           if cracked else pd.Series(dtype=int))

    L = [
        "# Eggshell Micro-Crack Inspection Report",
        "",
        f"**Date** {ts}  ",
        f"**Flock / batch** {flock_id or 'not recorded'}  ",
        f"**Breeder age class** {age_class}  ",
        f"**Inspected by** {operator or 'not recorded'}  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Eggs inspected | {n} |",
        f"| Micro-cracks detected | {cracked} ({rate:.1f}% of batch) |",
        f"| Flagged for manual review | {flagged} |",
        "",
    ]

    if cracked:
        L += ["### Crack morphology", "", "| Pattern | Eggs | Share of cracked |", "|---|---|---|"]
        for k, v in pat.items():
            L.append(f"| {k} | {v} | {v / cracked * 100:.0f}% |")
        L += [
            "",
            "## Probable root cause across the batch",
            "",
            f"**{flock['phys']:.0f}% physiological / nutritional** against "
            f"**{flock['mech']:.0f}% mechanical / environmental**",
            "",
            f"{flock['verdict']}",
            "",
            "### Recommended actions",
            "",
        ]
        for a in flock["actions"]:
            L.append(f"**{a['pattern']}** ({a['count']} egg{'s' if a['count'] > 1 else ''})  ")
            L.append(f"{a['text']}")
            L.append("")
    else:
        L += ["No micro-cracks were detected in this batch. No action required.", ""]

    if notes:
        L += ["## Operator notes", "", notes, ""]

    L += [
        "## Per-egg results",
        "",
        markdown_table(df, _COLUMNS),
        "",
        "---",
        "",
        "## How to read this report",
        "",
        "**Detection errs towards over-flagging.** The threshold was chosen so that no cracked "
        "egg was missed on the validation data. A flagged egg costs one manual check; a missed "
        "crack enters the setter, fails to hatch, and can contaminate neighbouring eggs.",
        "",
        "**Eggs marked for review** were classified with confidence below 60%, a band in which "
        "measured accuracy falls under 50%. Inspect these by eye before acting on them.",
        "",
        "**The root-cause split is structured expert guidance, not measurement.** It is a "
        "deterministic lookup on crack morphology and flock lay stage, assembled from Ross 308 "
        "specifications and the crack-morphology literature with a hatchery manager. It has not "
        "been validated against hatch outcomes.",
        "",
        "**A batch is weak evidence on its own.** Read this alongside your flock records and "
        "recent changes to feed, equipment or collection routine.",
        "",
        "*This report supports inspection by a qualified hatchery technician. It does not "
        "replace it.*",
    ]
    return "\n".join(L)

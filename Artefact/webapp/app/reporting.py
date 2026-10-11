"""
reporting.py
CSV and printable report generation for a completed batch.

Markdown tables are assembled directly rather than through pandas.to_markdown, which requires the
optional `tabulate` package and fails at the moment of download — after the user has already done
the work. The same mistake was made once in the Streamlit application; it is not repeated here.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime

CSV_COLUMNS = [
    ("file", "file"), ("status", "status"), ("pattern", "pattern"),
    ("crack_probability", "crack_probability"), ("confidence", "confidence"),
    ("physiological_pct", "physiological_pct"), ("mechanical_pct", "mechanical_pct"),
    ("needs_review", "needs_review"),
]

MD_COLUMNS = [
    ("file", "Egg"), ("status", "Status"), ("pattern", "Morphology"),
    ("crack_probability", "Crack prob."), ("confidence", "Confidence"),
    ("physiological_pct", "Physiological %"), ("mechanical_pct", "Mechanical %"),
    ("needs_review", "Review"),
]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.3f}" if v < 1 else f"{v:.0f}"
    return str(v)


def records_to_csv(records: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in CSV_COLUMNS])
    for r in records:
        w.writerow([_fmt(r.get(k)) if not isinstance(r.get(k), (int, float)) or
                    r.get(k) is None else r.get(k) for k, _ in CSV_COLUMNS])
    return buf.getvalue()


def markdown_table(records: list[dict], columns=MD_COLUMNS) -> str:
    head = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |" for r in records]
    return "\n".join([head, rule, *body])


def build_report(records: list[dict], summary: dict, flock: dict, age_class: str,
                 operator: str = "", flock_id: str = "", notes: str = "") -> str:
    ts = datetime.now().strftime("%d %B %Y, %H:%M")
    readable = [r for r in records if r.get("status") != "unreadable"]
    cracked = summary["n_cracked"]

    L = [
        "# Eggshell Micro-Crack Inspection Report", "",
        f"**Date** {ts}  ",
        f"**Flock / batch** {flock_id or 'not recorded'}  ",
        f"**Breeder age class** {age_class}  ",
        f"**Inspected by** {operator or 'not recorded'}  ",
        "", "---", "", "## Summary", "",
        "| Measure | Value |", "|---|---|",
        f"| Eggs inspected | {summary['n_readable']} |",
        f"| Micro-cracks detected | {cracked} ({summary['crack_rate']:.1f}% of batch) |",
        f"| Flagged for manual review | {summary['n_review']} |",
    ]
    if summary["n_unreadable"]:
        L.append(f"| Unreadable images | {summary['n_unreadable']} |")
    L.append("")

    if cracked:
        counts = flock.get("counts", {})
        L += ["### Crack morphology", "", "| Pattern | Eggs | Share of cracked |", "|---|---|---|"]
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v} | {v / cracked * 100:.0f}% |")
        L += ["", "## Probable root cause across the batch", "",
              f"**{flock['phys']:.0f}% physiological / nutritional** against "
              f"**{flock['mech']:.0f}% mechanical / environmental**", "",
              flock["verdict"], "", "### Recommended actions", ""]
        for a in flock["actions"]:
            n = a["count"]
            L.append(f"**{a['pattern']}** ({n} egg{'s' if n != 1 else ''})  ")
            L.append(a["text"])
            L.append("")
    else:
        L += ["No micro-cracks were detected in this batch. No action required.", ""]

    if notes:
        L += ["## Operator notes", "", notes, ""]

    L += ["## Per-egg results", "", markdown_table(readable), "", "---", "",
          "## How to read this report", "",
          "**Detection errs towards over-flagging.** The threshold was chosen so that no cracked "
          "egg was missed on the validation data. A flagged egg costs one manual check; a missed "
          "crack enters the setter, fails to hatch, and can contaminate neighbouring eggs.", "",
          "**Eggs marked for review** were classified with confidence below 60%, a band in which "
          "measured accuracy falls under 50%. Inspect these by eye before acting on them.", "",
          "**The root-cause split is structured expert guidance, not measurement.** It is a "
          "deterministic lookup on crack morphology and flock lay stage, assembled from Ross 308 "
          "specifications and the crack-morphology literature with a hatchery manager. It has "
          "not been validated against hatch outcomes.", "",
          "**A batch is weak evidence on its own.** Read this alongside your flock records and "
          "recent changes to feed, equipment or collection routine.", "",
          "*This report supports inspection by a qualified hatchery technician. "
          "It does not replace it.*"]
    return "\n".join(L)

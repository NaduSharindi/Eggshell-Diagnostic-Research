"""
Adds five columns to the Label Studio CSV export, keeping every existing column:
  age_class, physiological_nutritional_pct, mechanical_environmental_pct,
  interpretation, suggested_actions

Age class options (choose one):
  --age-file age_class.csv     columns: egg_id, age_class (for eggs from several flocks)
  --all-age "Late lay"         when every egg came from the same flock age

Allowed age classes: Early lay, Peak production, Late lay

Usage:
  python add_root_cause.py final_labels_v2.csv root_cause_rules.csv --all-age "Late lay"
  python add_root_cause.py final_labels_v2.csv root_cause_rules.csv --age-file age_class.csv
Output: final_labels_v3.csv
"""
import argparse, re
import pandas as pd

AGES = ["Early lay", "Peak production", "Late lay"]

ap = argparse.ArgumentParser()
ap.add_argument("labels"); ap.add_argument("rules")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--age-file"); g.add_argument("--all-age", choices=AGES)
ap.add_argument("--out", default="final_labels_v3.csv")
a = ap.parse_args()

df = pd.read_csv(a.labels, dtype=str, keep_default_na=False)
rules = pd.read_csv(a.rules, dtype=str, keep_default_na=False).set_index(["crack_pattern", "age_class"])
egg_ids = df["image"].str.extract(r"(E\d{3,})")[0]

if a.all_age:
    df["age_class"] = a.all_age
else:
    ages = pd.read_csv(a.age_file, dtype=str).set_index("egg_id")["age_class"].str.strip()
    df["age_class"] = egg_ids.map(ages).fillna("")

new = {"physiological_nutritional_pct": [], "mechanical_environmental_pct": [],
       "interpretation": [], "suggested_actions": []}
problems = []
for egg, status, pattern, age in zip(egg_ids, df["egg_status"], df["crack_pattern"], df["age_class"]):
    if age not in AGES:
        problems.append(f"{egg}: age class missing or misspelt ('{age}')")
    if status == "Micro_cracked" and (pattern, age) in rules.index:
        r = rules.loc[(pattern, age)]
        vals = [r.physiological_nutritional_pct, r.mechanical_environmental_pct,
                r.interpretation, r.suggested_actions]
    elif status == "Micro_cracked":
        problems.append(f"{egg}: no rule for pattern '{pattern}' with age '{age}'")
        vals = ["", "", "", ""]
    else:
        vals = ["", "", "No micro-crack detected; shell integrity is acceptable for setting.",
                "Set the egg for incubation; no corrective action needed."]
    for k, v in zip(new, vals):
        new[k].append(v)
for k, v in new.items():
    df[k] = v

df.to_csv(a.out, index=False, encoding="utf-8-sig")
print(f"Wrote {a.out}: {len(df)} rows, {len(df.columns)} columns (original columns kept).")
print("Problems:" if problems else "No problems found.")
for p in problems[:20]:
    print("  -", p)
if len(problems) > 20:
    print(f"  ... and {len(problems) - 20} more")

"""
rules.py
The root-cause rule base and the flock-level aggregation.

The attribution is a deterministic lookup on (crack morphology, breeder age class), assembled from
Ross 308 specifications and the crack-morphology literature with a hatchery manager. It is not
learned from data and has not been validated against hatch outcomes. The application says so
wherever it shows an attribution, and this module carries the same caveat so it is not lost if the
rules are ever reused elsewhere.

Because age class is supplied by the user rather than predicted, every attribution error traces to
a morphology error. On the held-out test set the attribution was exact for 82.4% of cracked eggs,
with a mean error of 4.8 percentage points, and the dominant cause reversed on 13.7%.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class _Settings:
    """Attribute view over a Flask config mapping.

    `app.config` is a dict subclass, so `cfg.TRAIN_SIDE` raises AttributeError while
    `cfg["TRAIN_SIDE"]` works. Wrapping it here keeps the call sites readable and means a missing
    key fails with a message naming the key instead of a bare AttributeError. The wrapper also
    accepts a plain object, so tests can pass a SimpleNamespace.
    """

    __slots__ = ("_m",)

    def __init__(self, mapping):
        object.__setattr__(self, "_m", mapping)

    def __getattr__(self, name):
        m = object.__getattribute__(self, "_m")
        try:
            return m[name] if hasattr(m, "__getitem__") else getattr(m, name)
        except (KeyError, TypeError, AttributeError):
            raise AttributeError(
                f"configuration key '{name}' is not set. Check config.py and any "
                f"environment overrides.") from None

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default


class RuleBase:
    def __init__(self, cfg):
        self.cfg = _Settings(cfg)
        self._rules: dict[tuple[str, str], dict] = {}
        self._error: Optional[str] = None
        self.patterns: list[str] = []
        self.age_classes: list[str] = []

    def load(self) -> None:
        try:
            from common import project_config as CFG
            self.patterns = list(CFG.PATTERNS)
            self.age_classes = list(CFG.AGE_ORDER)
            candidates = []
            if self.cfg.get("RULES_CSV"):
                candidates.append(Path(self.cfg.RULES_CSV))
            candidates += [CFG.DATA_DIR / "processed" / "root_cause_rules_recovered.csv",
                           CFG.ROOT_CAUSE_RULES_CSV]
            for c in candidates:
                if Path(c).is_file():
                    df = pd.read_csv(c)
                    self._rules = {
                        (str(r["crack_pattern"]).strip(), str(r["age_class"]).strip()): {
                            "phys": float(r["phys_pct"]), "mech": float(r["mech_pct"]),
                            "interpretation": str(r.get("interpretation", "")).strip(),
                            "actions": str(r.get("suggested_actions", "")).strip(),
                        } for _, r in df.iterrows()}
                    log.info("loaded %d rules from %s", len(self._rules), c)
                    break
            if not self._rules:
                self._error = ("Root-cause rules not found. Run notebook 01 to produce "
                               "data/processed/root_cause_rules_recovered.csv")
                log.error(self._error)
                return
            missing = [(p, a) for p in self.patterns for a in self.age_classes
                       if (p, a) not in self._rules]
            if missing:
                self._error = f"Rule base is incomplete; missing {missing}"
                log.error(self._error)
        except Exception as e:  # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            log.exception("rule load failed")

    @property
    def ready(self) -> bool:
        return bool(self._rules) and self._error is None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def get(self, pattern: str, age_class: str) -> Optional[dict]:
        return self._rules.get((str(pattern).strip(), str(age_class).strip()))

    def matrix(self) -> list[dict]:
        """The 3 x 3 table, for the model card."""
        out = []
        for p in self.patterns:
            row = {"pattern": p}
            for a in self.age_classes:
                r = self.get(p, a)
                row[a] = r["phys"] if r else None
            out.append(row)
        return out

    # ------------------------------------------------------------ aggregation
    def flock_assessment(self, records: list[dict], age_class: str) -> dict:
        """Aggregate per-egg attributions into one flock-level recommendation.

        Actions are grouped by crack pattern and reproduced in full. An earlier version split the
        action text on full stops to deduplicate it, which cut sentences at decimal points and
        inside parentheses, producing fragments such as "2%), and check house temperature".
        Advice shown to a farm manager must appear exactly as the domain expert wrote it.
        """
        cracked = [r for r in records if r.get("status") == "Micro_cracked" and r.get("pattern")]
        if not cracked:
            return {"phys": 0.0, "mech": 0.0, "n": 0, "actions": [],
                    "verdict": "No micro-cracks detected in this batch.",
                    "verdict_class": "ok"}

        phys_values = [r["physiological_pct"] for r in cracked
                       if r.get("physiological_pct") is not None]
        phys = float(sum(phys_values) / len(phys_values)) if phys_values else 0.0

        if phys >= 60:
            verdict = ("Predominantly shell weakness. The crack mix points at nutrition and bird "
                       "physiology rather than handling.")
            vclass = "phys"
        elif phys <= 40:
            verdict = ("Predominantly mechanical damage. The crack mix points at handling, "
                       "equipment and collection practice rather than shell quality.")
            vclass = "mech"
        else:
            verdict = ("Mixed causes. Shell quality and handling are contributing in comparable "
                       "measure; address both.")
            vclass = "mixed"

        counts: dict[str, int] = {}
        for r in cracked:
            counts[r["pattern"]] = counts.get(r["pattern"], 0) + 1
        actions = []
        for p, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            rule = self.get(p, age_class)
            if rule and rule["actions"]:
                actions.append({"pattern": p, "count": n, "phys": rule["phys"],
                                "text": rule["actions"]})
        return {"phys": phys, "mech": 100.0 - phys, "n": len(cracked),
                "verdict": verdict, "verdict_class": vclass, "actions": actions,
                "counts": counts}

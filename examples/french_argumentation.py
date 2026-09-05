# -*- coding: utf-8 -*-
"""Adapter for the French student argumentative corpus (Task1/2/3).

Everything here is specific to the corpus CSV format and stays out of the
library core. Usage:

    python french_argumentation.py --csv data.csv --out results.csv
        [--chance] [--weighted] [--tau 0.5] [--alpha 0.5] [--eta 0.5]
        [--nan-policy orphan_all] [--admissibility hard] [--null-model mathet]

Corpus quirks handled here rather than in the library: composite Task2
categories (relevance + quality), the Explicitation_argument alias, and the
unaccented "Synthese" spelling of the CSV. The weighted relation matrix from
an early version had a nonzero diagonal and is not shipped; relations use the
binary distance.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List, Optional

import pandas as pd

from opt_gamma import (
    CategoryDistance,
    GammaConfig,
    Relation,
    Span,
    Unit,
    gamma_composite,
    gamma_relations,
    gamma_segments,
)

# --------------------------------------------------------------- matrices
TASK1_LABELS = ["OnSujet", "HorsSujet"]

TASK2_PERT_WEIGHTED = {
    ("Thèse", "Non_thèse"): 0.5,
    ("Thèse", "Antithèse"): 1.0,
    ("Thèse", "Non_argumentatif"): 0.8,
    ("Non_thèse", "Antithèse"): 0.8,
    ("Non_thèse", "Non_argumentatif"): 0.5,
    ("Antithèse", "Non_argumentatif"): 1.0,
}
QUALITY_WEIGHTED = {("Qualitatif", "Vide"): 1.0}

TASK3_WEIGHTED = {
    ("Argument", "Explicitation"): 0.5,
    ("Argument", "Exemple"): 0.6,
    ("Argument", "Synthese"): 1.0,
    ("Argument", "Autre"): 1.0,
    ("Argument", "Annonce_these"): 0.8,
    ("Explicitation", "Exemple"): 0.7,
    ("Explicitation", "Synthese"): 1.0,
    ("Explicitation", "Autre"): 1.0,
    ("Explicitation", "Annonce_these"): 0.9,
    ("Exemple", "Synthese"): 1.0,
    ("Exemple", "Autre"): 1.0,
    ("Exemple", "Annonce_these"): 1.0,
    ("Synthese", "Autre"): 1.0,
    ("Synthese", "Annonce_these"): 1.0,
    ("Autre", "Annonce_these"): 1.0,
}

RELATION_LABELS = ["Argument", "Explicitation", "Exemple", "Repetition", "Illustration"]
# NOTE: the notebook's weighted relation matrix is intentionally NOT ported
# (invalid semantics: nonzero diagonal, off-diagonal < diagonal).
RELATION_BINARY = CategoryDistance(
    {(a, b): (0.0 if a == b else 1.0) for a in RELATION_LABELS for b in RELATION_LABELS}
)


def task2_distance(weighted: bool) -> CategoryDistance:
    pert = CategoryDistance(TASK2_PERT_WEIGHTED) if weighted else CategoryDistance.binarized(
        {**TASK2_PERT_WEIGHTED}
    )
    qual = CategoryDistance(QUALITY_WEIGHTED) if weighted else CategoryDistance.binarized(
        {**QUALITY_WEIGHTED}
    )

    def fn(a: str, b: str) -> float:
        if a == "Non_classifiable" or b == "Non_classifiable":
            return 0.0 if a == b else 1.0
        (ba, qa), (bb, qb) = _split(a), _split(b)
        p, q = pert(ba, bb), qual(qa, qb)
        if "NonFixé" in (qa, qb) and "NonFixé" not in (ba, bb):
            return p
        return 0.8 * p + 0.2 * q

    return CategoryDistance(fn=fn)


def _split(cat: str):
    parts = cat.split("_", 1)
    return parts[0], (parts[1] if len(parts) == 2 else "NonFixé")


def make_distances(weighted: bool):
    if weighted:
        return {
            "Task1": CategoryDistance(),  # binary anyway (2 labels)
            "Task2": task2_distance(True),
            "Task3": CategoryDistance(TASK3_WEIGHTED),
            "R": RELATION_BINARY,
        }
    return {
        "Task1": CategoryDistance(),
        "Task2": task2_distance(False),
        "Task3": CategoryDistance.binarized(TASK3_WEIGHTED),
        "R": RELATION_BINARY,
    }


# ------------------------------------------------------------------ parsing
def _parse_span(value) -> Optional[Span]:
    """CSV spans are INCLUSIVE [start, end] -> convert to half-open."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            parts = value.split(",")
            if len(parts) == 2:
                try:
                    value = [int(parts[0]), int(parts[1])]
                except ValueError:
                    return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            s, e = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
        if e >= s:
            return Span.from_inclusive(s, e)
    return None


def rebuild_task2_category(group: pd.DataFrame) -> str:
    if "Non_classifiable" in group["tag"].unique():
        return "Non_classifiable"
    pert, qual = None, None
    for tag in group["tag"].unique():
        if tag in ("These", "Antithese", "Non_these", "Non_argumentatif"):
            pert = (tag.replace("These", "Thèse")
                       .replace("Antithese", "Antithèse")
                       .replace("Non_these", "Non_thèse"))
        elif tag in ("Qualitatif", "Vide"):
            qual = tag
    return f"{pert or 'NonFixé'}_{qual or 'NonFixé'}"


def extract_units(df_tsk: pd.DataFrame, task: str) -> dict[str, List[Unit]]:
    df = df_tsk[df_tsk["label"].astype(str).str.startswith("T")].copy()
    if task == "Task1":
        df["category"] = df["tag"].map({"OK": "OnSujet", "HS": "HorsSujet"}).fillna("HorsSujet")
    elif task == "Task2":
        comp = df.groupby("label", group_keys=False)[["tag"]].apply(rebuild_task2_category).rename("category")
        df = df.merge(comp, left_on="label", right_index=True)
    else:
        df["category"] = df["tag"].replace({"Explicitation_argument": "Explicitation"})

    out: dict[str, List[Unit]] = {}
    for _, row in df.iterrows():
        span = _parse_span(row["char_span"])
        if span is None:
            continue
        out.setdefault(str(row["annotator"]), []).append(Unit(span, str(row["category"])))
    return out


def extract_relations(df_tsk: pd.DataFrame, df_full: pd.DataFrame) -> dict[str, List[Relation]]:
    t3 = df_full[df_full["task"] == "Task3"]
    out: dict[str, List[Relation]] = {}
    for _, row in df_tsk[df_tsk["label"].astype(str).str.startswith("R")].iterrows():
        try:
            parsed = ast.literal_eval(row["char_span"])
        except Exception:
            continue
        if not isinstance(parsed, list) or len(parsed) != 2:
            continue
        try:
            t1 = next(p.split(":")[1] for p in parsed if str(p).startswith("Arg1"))
            t2 = next(p.split(":")[1] for p in parsed if str(p).startswith("Arg2"))
        except StopIteration:
            continue

        def span_of(tid):
            ref = t3[(t3["label"] == tid)
                     & (t3["annotator"] == row["annotator"])
                     & (t3["text_ids"] == row["text_ids"])]
            return _parse_span(ref.iloc[0]["char_span"]) if len(ref) else None

        s1, s2 = span_of(t1), span_of(t2)
        if s1 and s2:
            out.setdefault(str(row["annotator"]), []).append(Relation(s1, s2, str(row["tag"])))
    return out


# --------------------------------------------------------------------- main
def run(csv_path: Path, out_path: Path, cfg: GammaConfig, weighted: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ("text_id", "text_ids"):
        if c in df.columns:
            df = df.rename(columns={c: "text_ids"})
            break
    else:
        raise ValueError("No text_id/text_ids column in CSV")
    df = df[df["task"].isin(["Task1", "Task2", "Task3"])].copy()

    dists = make_distances(weighted)
    rows = []
    for text_id in df["text_ids"].unique():
        for task in ("Task1", "Task2", "Task3"):
            df_tsk = df[(df["task"] == task) & (df["text_ids"] == text_id)]
            rec = {"text_id": text_id, "task": task,
                   "gamma_typ": float("nan"), "reason_typ": "",
                   "gamma_rel": float("nan"), "reason_rel": "",
                   "final_score": float("nan")}
            if df_tsk.empty:
                rec["reason_typ"] = "no annotations for task"
                rows.append(rec)
                continue

            units = extract_units(df_tsk, task)
            if len(units) != 2:
                rec["reason_typ"] = f"{len(units)} annotators (expected 2)"
            else:
                (u1, u2) = units.values()
                r = gamma_segments(u1, u2, cat_dist=dists[task], config=cfg)
                rec["gamma_typ"], rec["reason_typ"] = r.gamma, r.reason

            rel_res = None
            if task == "Task3":
                rels = extract_relations(df_tsk, df)
                if len(rels) != 2:
                    rec["reason_rel"] = f"{len(rels)} annotators with relations"
                else:
                    (r1, r2) = rels.values()
                    rel_res = gamma_relations(r1, r2, rel_dist=dists["R"], config=cfg)
                    rec["gamma_rel"], rec["reason_rel"] = rel_res.gamma, rel_res.reason

            if task == "Task3" and rel_res is not None:
                from opt_gamma.gamma import GammaResult
                typ_stub = GammaResult(rec["gamma_typ"], 0, None, 0, None)
                rec["final_score"] = gamma_composite(typ_stub, rel_res, cfg.lambda_typ)
            else:
                rec["final_score"] = rec["gamma_typ"]
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("gamma_results.csv"))
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--lambda-typ", type=float, default=0.7)
    ap.add_argument("--chance", action="store_true")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--nan-policy", choices=["nan", "orphan_all", "raise"], default="orphan_all",
                    help="'orphan_all' (recommended); 'nan' reproduces the original notebook")
    ap.add_argument("--admissibility", choices=["hard", "soft"], default="hard",
                    help="'hard' = threshold tau; 'soft' = Mathet-2017 confidence weighting")
    ap.add_argument("--null-model", choices=["mathet", "li"], default="mathet",
                    help="null model for --chance: circular shift (Mathet 2015) or Li 2024")
    args = ap.parse_args()

    from opt_gamma import CircularShiftNull, LiNonOverlapNull
    null = CircularShiftNull() if args.null_model == "mathet" else LiNonOverlapNull()
    cfg = GammaConfig(
        alpha=args.alpha, tau=args.tau, eta=args.eta, lambda_typ=args.lambda_typ,
        chance_correction=args.chance, on_no_admissible=args.nan_policy,
        admissibility=args.admissibility, null_model=null, n_iter="auto",
    )
    res = run(args.csv, args.out, cfg, args.weighted)
    print(res.head(30).to_string(index=False))
    print(f"\nExported: {args.out}")

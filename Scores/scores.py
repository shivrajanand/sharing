"""
How to run:
! python3 scores.py /home/shivraj-pg/DEPNECT/Scores/test.csv gold_col model-output-col
"""

import json
import pandas as pd
import sys
from datetime import datetime
# -------------------------------
# Helper: Parse compound spans
# -------------------------------
def parse_spans(analysis_json):
    """
    Extract spans as list of tuples (start, end, label) from JSON.
    Handles string or dict input.
    """
    if pd.isna(analysis_json):
        return []
    if isinstance(analysis_json, str):
        try:
            analysis = json.loads(analysis_json)
        except Exception:
            # fallback for single quotes / malformed json
            try:
                analysis = eval(analysis_json)
            except Exception:
                return []
    else:
        analysis = analysis_json

    spans = []
    for c in analysis.get("compounds", []):
        if not isinstance(c, dict):
            continue
        span = c.get("span", [])
        label = str(c.get("label", "")).strip()
        if len(span) < 2:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
            spans.append((start, end, label))
        except Exception:
            continue
    return spans


# -------------------------------
# Helper: Compute metrics for one sentence
# -------------------------------
def span_metrics(pred_spans, gold_spans):
    """
    Compute USS, LSS, EM for one sample.
    """
    gold_unlabeled = {(s, e) for s, e, _ in gold_spans}
    pred_unlabeled = {(s, e) for s, e, _ in pred_spans}

    gold_labeled = set(gold_spans)
    pred_labeled = set(pred_spans)

    # overlaps
    unlabeled_overlap = len(gold_unlabeled & pred_unlabeled)
    labeled_overlap = len(gold_labeled & pred_labeled)

    USS = 0.0 if (len(pred_unlabeled) + len(gold_unlabeled)) == 0 else (
        2 * unlabeled_overlap / (len(pred_unlabeled) + len(gold_unlabeled))
    )
    LSS = 0.0 if (len(pred_labeled) + len(gold_labeled)) == 0 else (
        2 * labeled_overlap / (len(pred_labeled) + len(gold_labeled))
    )
    EM = 1.0 if gold_labeled == pred_labeled else 0.0

    return USS, LSS, EM


# -------------------------------
# Main evaluation function
# -------------------------------
def evaluate_inplace(df, gold_col, pred_col, out_prefix="eval"):
    """
    Compare gold and predicted JSONs row-wise inside a DataFrame.

    Args:
        df: pandas DataFrame with both columns.
        gold_col: column name containing gold JSONs.
        pred_col: column name containing predicted JSONs.
        out_prefix: prefix for new columns.

    Returns:
        summary dict of mean USS, LSS, EM
    """
    if gold_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Columns '{gold_col}' or '{pred_col}' not found in DataFrame")

    uss_list, lss_list, em_list = [], [], []

    for _, row in df.iterrows():
        gold_spans = parse_spans(row[gold_col])
        pred_spans = parse_spans(row[pred_col])
        USS, LSS, EM = span_metrics(pred_spans, gold_spans)
        uss_list.append(USS)
        lss_list.append(LSS)
        em_list.append(EM)

    df[f"{out_prefix}_USS"] = uss_list
    df[f"{out_prefix}_LSS"] = lss_list
    df[f"{out_prefix}_EM"] = em_list

    n = len(df)
    summary = {
        "USS": round(sum(uss_list) / n, 4) if n > 0 else 0.0,
        "LSS": round(sum(lss_list) / n, 4) if n > 0 else 0.0,
        "EM": round(sum(em_list) / n, 4) if n > 0 else 0.0
    }

    return summary


if __name__=='__main__':

    df_path, gold_col, model_out = sys.argv[1], sys.argv[2], sys.argv[3]
    
    df = pd.read_csv(df_path)
    
    summary = evaluate_inplace(df, gold_col=gold_col, pred_col=model_out, out_prefix="eval")
    
    with open("summary-eval.txt", "a", encoding="utf-8") as f:
        f.write(str(datetime.now()))
        f.write(f"Corpus-level summary on: \nFilepath: {df_path}\nGold-Col: {gold_col}\nModel-Output-Column: {model_out}\n")
        f.write(str(summary))
        f.write("\n#########################################################\n\n")
        
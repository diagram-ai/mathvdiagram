"""
HTML report for the dataset_helper classification results.

Generates a standalone HTML file showing each classified image with its
category, reliability score, provider votes, and agreement level.
"""

from __future__ import annotations

import html
import json
import os

import pandas as pd

from .. import config
from ..data_loader import load_mathvision, get_image_base64


# Colour palette for category badges
_CATEGORY_COLORS: dict[str, str] = {
    "geometric_construction": "#2563eb",
    "coordinate_plot": "#059669",
    "statistical_chart": "#d97706",
    "schematic_diagram": "#7c3aed",
    "3d_figure": "#db2777",
    "non_diagram": "#dc2626",
}


def _esc(text) -> str:
    if pd.isna(text):
        return "<em>N/A</em>"
    return html.escape(str(text)).replace("\n", "<br>")


def _img_tag(image_id) -> str:
    b64, media_type = get_image_base64(image_id)
    if b64 is None:
        return '<div class="no-image">Image not available</div>'
    return f'<img src="data:{media_type};base64,{b64}" alt="Image {image_id}">'


def _provider_vote_html(row: pd.Series, providers: list[str]) -> str:
    """Build small coloured chips showing each provider's vote."""
    chips = []
    for prov in providers:
        cat = row.get(f"cat_{prov}")
        conf = row.get(f"conf_{prov}", "")
        if pd.isna(cat) or cat == "unknown":
            continue
        color = _CATEGORY_COLORS.get(cat, "#6b7280")
        label = cat.replace("_", " ").title()
        conf_str = f" ({conf})" if pd.notna(conf) else ""
        chips.append(
            f'<span class="provider-chip" style="border-color:{color};color:{color}">'
            f"<strong>{prov.title()}</strong>: {label}{conf_str}</span>"
        )
    if not chips:
        return ""
    return '<div class="provider-votes">' + " ".join(chips) + "</div>"


def generate_classification_report(
    classification_csv: str | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate an HTML report from full_classification.csv.

    Shows every image with its category badge, reliability score,
    agreement level, per-provider votes, and the original question.
    """
    classification_csv = classification_csv or os.path.join(
        config.OUTPUT_DIR, "full_classification.csv"
    )
    output_path = output_path or os.path.join(
        config.OUTPUT_DIR, "classification_report.html"
    )

    if not os.path.exists(classification_csv):
        print(f"Classification CSV not found: {classification_csv}")
        print("Run --classify first.")
        return ""

    load_mathvision()
    df = pd.read_csv(classification_csv)

    # Detect provider columns
    providers = [c.replace("cat_", "") for c in df.columns if c.startswith("cat_")]

    cat_col = "final_category" if "final_category" in df.columns else "majority_category"

    # Summary counts
    total = len(df)
    n_diagram = int(df["is_diagram"].sum()) if "is_diagram" in df.columns else 0
    n_non = total - n_diagram
    cat_counts = df[cat_col].value_counts().to_dict() if cat_col in df.columns else {}

    # Build category filter buttons
    all_cats = sorted(cat_counts.keys())
    filter_buttons = ['<button class="filter-btn active" data-cat="all">All</button>']
    for cat in all_cats:
        color = _CATEGORY_COLORS.get(cat, "#6b7280")
        label = cat.replace("_", " ").title()
        filter_buttons.append(
            f'<button class="filter-btn" data-cat="{cat}" '
            f'style="--btn-color:{color}">{label} ({cat_counts[cat]})</button>'
        )

    # Sort: diagrams first, then by reliability descending
    sort_cols = []
    if "is_diagram" in df.columns:
        sort_cols.append("is_diagram")
    if "reliability_score" in df.columns:
        sort_cols.append("reliability_score")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    # Build cards
    cards_html = []
    for _, row in df.iterrows():
        img_id = row.get("image_id", row.get("id", ""))
        question = row.get("question", "")
        category = row.get(cat_col, "unknown")
        is_diagram = bool(row.get("is_diagram", False))
        reliability = row.get("reliability_score", None)
        agreement = row.get("agreement_level", "")
        subject = row.get("subject", "")
        source = row.get("classification_source", "")

        color = _CATEGORY_COLORS.get(category, "#6b7280")
        cat_label = category.replace("_", " ").title()

        # Reliability bar width
        rel_pct = f"{reliability * 100:.0f}" if pd.notna(reliability) else "0"
        rel_text = f"{reliability:.0%}" if pd.notna(reliability) else "N/A"

        # Agreement badge
        agree_cls = {
            "unanimous": "agree-unanimous",
            "majority": "agree-majority",
            "no_agreement": "agree-none",
            "insufficient_data": "agree-none",
            "prefilter": "agree-prefilter",
        }.get(agreement, "")
        agree_label = agreement.replace("_", " ").title() if agreement else ""

        border_class = "diagram-card" if is_diagram else "non-diagram-card"
        provider_html = _provider_vote_html(row, providers) if providers else ""

        card = f"""
        <div class="card {border_class}" data-cat="{category}">
            <div class="card-header">
                <div class="card-header-left">
                    <span class="image-id">ID: {_esc(str(img_id))}</span>
                    <span class="subject-tag">{_esc(subject)}</span>
                </div>
                <div class="card-header-right">
                    <span class="badge" style="background:{color}20;color:{color};border:1px solid {color}">{cat_label}</span>
                    {'<span class="agree-badge ' + agree_cls + '">' + agree_label + '</span>' if agree_label else ''}
                </div>
            </div>
            <div class="card-body">
                <div class="card-top-row">
                    <div class="image-container">{_img_tag(img_id)}</div>
                    <div class="card-info">
                        <div class="question"><strong>Question:</strong> {_esc(question)}</div>
                        <div class="reliability-row">
                            <span class="rel-label">Reliability:</span>
                            <div class="rel-bar-container">
                                <div class="rel-bar" style="width:{rel_pct}%;background:{color}"></div>
                            </div>
                            <span class="rel-value">{rel_text}</span>
                        </div>
                        {f'<div class="source-tag">Source: {_esc(source)}</div>' if source else ''}
                        {provider_html}
                    </div>
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    # Agreement stats summary
    agree_stats = ""
    if "agreement_level" in df.columns:
        una = int((df["agreement_level"] == "unanimous").sum())
        maj = int((df["agreement_level"] == "majority").sum())
        no_ag = int((df["agreement_level"] == "no_agreement").sum())
        insuf = int((df["agreement_level"] == "insufficient_data").sum())
        prefil = int((df["agreement_level"] == "prefilter").sum())
        agree_stats = f"""
        <div class="agree-summary">
            <span>Unanimous: {una}</span>
            <span>Majority: {maj}</span>
            <span>No agreement: {no_ag}</span>
            {'<span>Insufficient: ' + str(insuf) + '</span>' if insuf else ''}
            {'<span>Pre-filter: ' + str(prefil) + '</span>' if prefil else ''}
        </div>"""

    mean_rel = ""
    if "reliability_score" in df.columns:
        mean_rel = f'<div class="stat"><div class="num">{df["reliability_score"].mean():.0%}</div>Mean Reliability</div>'

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classification Report — MathVDiagram</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
    .header {{ text-align: center; padding: 30px 20px; margin-bottom: 20px; background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; border-radius: 12px; }}
    .header h1 {{ font-size: 2em; margin-bottom: 6px; }}
    .header p {{ opacity: 0.85; }}
    .stats {{ display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-top: 15px; }}
    .stat {{ background: rgba(255,255,255,0.15); padding: 10px 20px; border-radius: 8px; }}
    .stat .num {{ font-size: 1.5em; font-weight: bold; }}
    .filters {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 20px; }}
    .filter-btn {{ padding: 6px 14px; border-radius: 20px; border: 1px solid #ddd; background: white; cursor: pointer; font-size: 0.85em; font-weight: 500; transition: all 0.2s; }}
    .filter-btn:hover {{ border-color: var(--btn-color, #333); color: var(--btn-color, #333); }}
    .filter-btn.active {{ background: var(--btn-color, #1a1a2e); color: white; border-color: var(--btn-color, #1a1a2e); }}
    .agree-summary {{ display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; font-size: 0.9em; color: #555; }}
    .card {{ background: white; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
    .card.hidden {{ display: none; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid #eee; flex-wrap: wrap; gap: 8px; }}
    .card-header-left, .card-header-right {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .image-id {{ font-size: 1.1em; font-weight: 600; }}
    .subject-tag {{ font-size: 0.8em; color: #888; background: #f0f2f5; padding: 2px 8px; border-radius: 4px; }}
    .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }}
    .agree-badge {{ padding: 4px 10px; border-radius: 20px; font-size: 0.75em; font-weight: 500; }}
    .agree-unanimous {{ background: #d1fae5; color: #065f46; }}
    .agree-majority {{ background: #dbeafe; color: #1e40af; }}
    .agree-none {{ background: #fee2e2; color: #991b1b; }}
    .agree-prefilter {{ background: #f3e8ff; color: #6b21a8; }}
    .card-body {{ padding: 20px; }}
    .card-top-row {{ display: grid; grid-template-columns: 300px 1fr; gap: 20px; align-items: start; }}
    .image-container {{ text-align: center; }}
    .image-container img {{ max-width: 100%; max-height: 300px; border: 1px solid #ddd; border-radius: 8px; }}
    .no-image {{ padding: 40px; background: #f8f9fa; color: #999; border-radius: 8px; }}
    .question {{ padding: 10px 14px; background: #f8f9fa; border-radius: 8px; margin-bottom: 12px; line-height: 1.5; font-size: 0.9em; }}
    .reliability-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
    .rel-label {{ font-size: 0.85em; font-weight: 500; color: #555; white-space: nowrap; }}
    .rel-bar-container {{ flex: 1; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }}
    .rel-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
    .rel-value {{ font-size: 0.85em; font-weight: 600; min-width: 36px; text-align: right; }}
    .source-tag {{ font-size: 0.8em; color: #888; margin-bottom: 8px; }}
    .provider-votes {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
    .provider-chip {{ font-size: 0.78em; padding: 3px 10px; border: 1px solid; border-radius: 16px; background: white; }}
    .diagram-card {{ border-left: 4px solid #28a745; }}
    .non-diagram-card {{ border-left: 4px solid #dc3545; }}
    @media (max-width: 768px) {{ .card-top-row {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
    <h1>Classification Report</h1>
    <p>MathVDiagram &mdash; Taxonomy-based multi-model classification</p>
    <div class="stats">
        <div class="stat"><div class="num">{total}</div>Total</div>
        <div class="stat"><div class="num">{n_diagram}</div>Diagrams</div>
        <div class="stat"><div class="num">{n_non}</div>Non-Diagrams</div>
        {mean_rel}
    </div>
</div>
<div class="filters">
    {''.join(filter_buttons)}
</div>
{agree_stats}
{''.join(cards_html)}
<script>
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const cat = btn.dataset.cat;
        document.querySelectorAll('.card').forEach(card => {{
            card.classList.toggle('hidden', cat !== 'all' && card.dataset.cat !== cat);
        }});
    }});
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"Classification report saved to {output_path}")
    return output_path

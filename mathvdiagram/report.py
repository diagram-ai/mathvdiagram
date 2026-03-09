"""
Generate an HTML report showing classification results, images,
model descriptions, and consensus prompts.
"""

import html
import os

import pandas as pd

from . import config
from .data_loader import load_mathvision, get_image_base64


def _img_tag(image_id) -> str:
    """Return an <img> tag with base64 embedded image, or a placeholder."""
    b64, media_type = get_image_base64(image_id)
    if b64 is None:
        return '<div class="no-image">Image not available</div>'
    return f'<img src="data:{media_type};base64,{b64}" alt="Image {image_id}">'


def _esc(text) -> str:
    """HTML-escape text, handle NaN."""
    if pd.isna(text):
        return '<em>N/A</em>'
    return html.escape(str(text)).replace("\n", "<br>")


def generate_report(output_path: str | None = None):
    """
    Generate an HTML report from the pipeline output CSVs.

    Args:
        output_path: Path for the HTML file. Defaults to output/report.html.
    """
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "report.html")

    # Load the HF dataset so images are available
    load_mathvision()

    # Load all CSVs
    classification_df = pd.read_csv(config.CLASSIFICATION_CSV) if os.path.exists(config.CLASSIFICATION_CSV) else pd.DataFrame()
    skipped_df = pd.read_csv(config.SKIPPED_CSV) if os.path.exists(config.SKIPPED_CSV) else pd.DataFrame()
    descriptions_df = pd.read_csv(config.DESCRIPTIONS_CSV) if os.path.exists(config.DESCRIPTIONS_CSV) else pd.DataFrame()
    consensus_df = pd.read_csv(config.CONSENSUS_CSV) if os.path.exists(config.CONSENSUS_CSV) else pd.DataFrame()

    # Merge descriptions and consensus into a lookup by image_id
    desc_map = {}
    if not descriptions_df.empty:
        for _, row in descriptions_df.iterrows():
            desc_map[row["image_id"]] = row

    cons_map = {}
    if not consensus_df.empty:
        for _, row in consensus_df.iterrows():
            cons_map[row["image_id"]] = row

    # Combine all image IDs (math + skipped)
    all_ids = []
    math_ids = set()
    if not classification_df.empty:
        for _, row in classification_df.iterrows():
            entry = {"image_id": row["image_id"], "question": row["question"], "is_math": True}
            if "final_category" in row.index and pd.notna(row.get("final_category")):
                entry["category"] = row["final_category"]
            elif "majority_category" in row.index and pd.notna(row.get("majority_category")):
                entry["category"] = row["majority_category"]
            if "reliability_score" in row.index and pd.notna(row.get("reliability_score")):
                entry["reliability"] = row["reliability_score"]
            all_ids.append(entry)
            math_ids.add(row["image_id"])
    if not skipped_df.empty:
        for _, row in skipped_df.iterrows():
            all_ids.append({"image_id": row["image_id"], "question": row["question"], "is_math": False})

    # Sort by image_id
    all_ids.sort(key=lambda x: int(x["image_id"]) if str(x["image_id"]).isdigit() else x["image_id"])

    num_math = len(math_ids)
    num_skipped = len(all_ids) - num_math

    # Build HTML
    cards_html = []
    for item in all_ids:
        img_id = item["image_id"]
        is_math = item["is_math"]
        question = item["question"]

        category = item.get("category")
        reliability = item.get("reliability")
        if is_math and category:
            cat_label = category.replace("_", " ").title()
            rel_text = f" ({reliability:.0%})" if reliability is not None else ""
            badge = f'<span class="badge math">{_esc(cat_label)}{rel_text}</span>'
        elif is_math:
            badge = '<span class="badge math">MATH</span>'
        else:
            badge = '<span class="badge non-math">NON-MATH</span>'
        img_tag = _img_tag(img_id)

        # Descriptions section (only for math images)
        desc_section = ""
        if is_math and img_id in desc_map:
            d = desc_map[img_id]
            desc_section = f"""
            <div class="descriptions">
                <h3>Model Descriptions</h3>
                <div class="desc-grid">
                    <div class="desc-box gemini">
                        <h4>Gemini</h4>
                        <div class="desc-content">{_esc(d.get("gemini_prompt", ""))}</div>
                    </div>
                    <div class="desc-box openai">
                        <h4>OpenAI GPT-4o</h4>
                        <div class="desc-content">{_esc(d.get("openai_prompt", ""))}</div>
                    </div>
                </div>
            </div>
            """

        # Consensus section (only for math images)
        cons_section = ""
        if is_math and img_id in cons_map:
            c = cons_map[img_id]
            cons_section = f"""
            <div class="consensus">
                <h3>Claude Consensus</h3>
                <div class="desc-grid">
                    <div class="desc-box detailed">
                        <h4>Detailed Prompt</h4>
                        <div class="desc-content">{_esc(c.get("detailed_prompt", ""))}</div>
                    </div>
                    <div class="desc-box concise">
                        <h4>Concise Prompt</h4>
                        <div class="desc-content">{_esc(c.get("concise_prompt", ""))}</div>
                    </div>
                </div>
            </div>
            """

        card = f"""
        <div class="card {'math-card' if is_math else 'skip-card'}">
            <div class="card-header">
                <span class="image-id">ID: {img_id}</span>
                {badge}
            </div>
            <div class="card-body">
                <div class="image-container">{img_tag}</div>
                <div class="question"><strong>Question:</strong> {_esc(question)}</div>
                {desc_section}
                {cons_section}
            </div>
        </div>
        """
        cards_html.append(card)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MathVDiagram Benchmarking Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
    .header {{ text-align: center; padding: 30px 20px; margin-bottom: 30px; background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; border-radius: 12px; }}
    .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
    .stats {{ display: flex; gap: 20px; justify-content: center; margin-top: 15px; }}
    .stat {{ background: rgba(255,255,255,0.15); padding: 10px 20px; border-radius: 8px; }}
    .stat .num {{ font-size: 1.5em; font-weight: bold; }}
    .card {{ background: white; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid #eee; }}
    .image-id {{ font-size: 1.1em; font-weight: 600; }}
    .badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; text-transform: uppercase; }}
    .badge.math {{ background: #d4edda; color: #155724; }}
    .badge.non-math {{ background: #f8d7da; color: #721c24; }}
    .card-body {{ padding: 20px; }}
    .image-container {{ text-align: center; margin-bottom: 16px; }}
    .image-container img {{ max-width: 100%; max-height: 400px; border: 1px solid #ddd; border-radius: 8px; }}
    .no-image {{ padding: 40px; background: #f8f9fa; color: #999; border-radius: 8px; }}
    .question {{ padding: 12px 16px; background: #f8f9fa; border-radius: 8px; margin-bottom: 16px; line-height: 1.5; }}
    .descriptions, .consensus {{ margin-top: 16px; }}
    .descriptions h3, .consensus h3 {{ font-size: 1.1em; margin-bottom: 12px; padding-bottom: 6px; border-bottom: 2px solid #eee; }}
    .desc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .desc-box {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 14px; }}
    .desc-box h4 {{ margin-bottom: 8px; font-size: 0.95em; }}
    .desc-box.gemini h4 {{ color: #4285f4; }}
    .desc-box.openai h4 {{ color: #10a37f; }}
    .desc-box.detailed h4 {{ color: #d97706; }}
    .desc-box.concise h4 {{ color: #7c3aed; }}
    .desc-content {{ font-size: 0.85em; line-height: 1.6; max-height: 300px; overflow-y: auto; color: #444; }}
    .math-card {{ border-left: 4px solid #28a745; }}
    .skip-card {{ border-left: 4px solid #dc3545; }}
    @media (max-width: 768px) {{ .desc-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="header">
    <h1>MathVDiagram Benchmarking Report</h1>
    <p>Classification, Descriptions &amp; Consensus Results</p>
    <div class="stats">
        <div class="stat"><div class="num">{len(all_ids)}</div>Total Images</div>
        <div class="stat"><div class="num">{num_math}</div>Math Diagrams</div>
        <div class="stat"><div class="num">{num_skipped}</div>Non-Math (Skipped)</div>
    </div>
</div>
{''.join(cards_html)}
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"Report saved to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_report()

"""
generate.py -- Generate diagram images from concise prompts using LLM APIs.

For code-generating LLMs: prompt -> code response -> detect format -> compile -> PNG
For image-generating models: prompt -> image directly -> PNG

Run with::

    python -m mathvdiagram.benchmark.generate --model deepseek-v3
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

from PIL import Image
from tqdm import tqdm

from .models import MODELS, get_client, generate_code, generate_image

# ---------------------------------------------------------------------------
# Code extraction and format detection
# ---------------------------------------------------------------------------

def extract_code_block(response_text: str) -> tuple[str, str]:
    """Extract code from markdown code blocks and detect the format.

    Returns (code, format) where format is one of:
    'tikz', 'svg', 'python', 'unknown'
    """
    if not response_text:
        return "", "unknown"

    # Strip <think>...</think> blocks from reasoning models.
    text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if not text:
        text = response_text  # fallback if everything was in <think>

    # Find all fenced code blocks.
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Use the LAST code block (reasoning models put the answer last).
        lang_hint, code = matches[-1]
        lang_hint = lang_hint.lower().strip()
    else:
        # No code block found -- treat entire response as code.
        code = text.strip()
        lang_hint = ""

    fmt = detect_format(code, lang_hint)
    return code.strip(), fmt


def detect_format(code: str, lang_hint: str = "") -> str:
    """Auto-detect whether code is TikZ, SVG, Python/matplotlib, or unknown."""
    if lang_hint in ("latex", "tex", "tikz"):
        return "tikz"
    if lang_hint in ("svg", "xml", "html"):
        return "svg"
    if lang_hint in ("python", "py", "python3"):
        return "python"

    code_lower = code.lower()

    if "\\begin{tikzpicture}" in code or "\\tikz" in code_lower:
        return "tikz"
    if code.strip().startswith("<svg") or "<svg " in code_lower:
        return "svg"
    if "import matplotlib" in code or "plt.savefig" in code_lower or "plt.show" in code_lower:
        return "python"
    if "\\documentclass" in code or "\\begin{document}" in code:
        return "tikz"  # LaTeX document with likely TikZ
    if "import " in code and ("draw" in code_lower or "figure" in code_lower):
        return "python"

    return "unknown"


# ---------------------------------------------------------------------------
# Compilers: code -> PNG image
# ---------------------------------------------------------------------------

def compile_tikz(code: str, output_path: str) -> bool:
    """Compile TikZ/LaTeX code to PNG (requires pdflatex + pdftoppm)."""
    if "\\documentclass" not in code:
        code = (
            "\\documentclass[border=2pt]{standalone}\n"
            "\\usepackage{tikz}\n"
            "\\usepackage{tikz-3dplot}\n"
            "\\usepackage{pgfplots}\n"
            "\\pgfplotsset{compat=1.18}\n"
            "\\usepackage{amsmath,amssymb,amsfonts}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{lmodern}\n"
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{circuitikz}\n"
            "\\usepackage{tikz-cd}\n"
            "\\usetikzlibrary{\n"
            "  calc,positioning,shapes.geometric,shapes.misc,shapes.symbols,\n"
            "  shapes.arrows,shapes.multipart,\n"
            "  decorations.pathmorphing,decorations.markings,decorations.pathreplacing,\n"
            "  decorations.text,\n"
            "  arrows.meta,arrows,\n"
            "  angles,quotes,\n"
            "  intersections,through,\n"
            "  patterns,shadings,\n"
            "  backgrounds,fit,\n"
            "  matrix,chains,trees,\n"
            "  3d,perspective,\n"
            "  plotmarks,\n"
            "  automata,petri,\n"
            "  mindmap,shadows,\n"
            "  spy,turtle,\n"
            "  folding,\n"
            "  babel,\n"
            "}\n"
            "\\begin{document}\n"
            f"{code}\n"
            "\\end{document}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "diagram.tex")
        with open(tex_path, "w") as f:
            f.write(code)

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "diagram.tex"],
            cwd=tmpdir,
            capture_output=True,
            timeout=30,
        )

        pdf_path = os.path.join(tmpdir, "diagram.pdf")
        if not os.path.exists(pdf_path):
            return False

        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", pdf_path,
             os.path.join(tmpdir, "output")],
            capture_output=True,
            timeout=15,
        )

        png_path = os.path.join(tmpdir, "output.png")
        if os.path.exists(png_path):
            Image.open(png_path).save(output_path)
            return True

    return False


def compile_svg(code: str, output_path: str) -> bool:
    """Compile SVG code to PNG (requires cairosvg)."""
    import cairosvg

    if not code.strip().startswith("<svg"):
        code = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">\n{code}\n</svg>'

    try:
        cairosvg.svg2png(bytestring=code.encode("utf-8"), write_to=output_path,
                         output_width=800, output_height=800)
        return True
    except Exception:
        return False


def compile_python(code: str, output_path: str) -> bool:
    """Execute Python/matplotlib code and capture the saved figure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, "figure.png")

        modified_code = code.replace("plt.show()", f"plt.savefig('{fig_path}', dpi=300, bbox_inches='tight')")

        # Redirect ALL savefig calls to our known path (handles hardcoded filenames).
        modified_code = re.sub(
            r"plt\.savefig\s*\([^)]*\)",
            f"plt.savefig('{fig_path}', dpi=300, bbox_inches='tight')",
            modified_code,
        )
        modified_code = re.sub(
            r"fig\.savefig\s*\([^)]*\)",
            f"fig.savefig('{fig_path}', dpi=300, bbox_inches='tight')",
            modified_code,
        )

        if "savefig" not in modified_code:
            modified_code += f"\nimport matplotlib.pyplot as plt\nplt.savefig('{fig_path}', dpi=300, bbox_inches='tight')\n"

        # Redirect PIL Image.save() calls to our path.
        modified_code = re.sub(
            r"(\w+)\.save\s*\(\s*['\"][^'\"]+['\"]\s*\)",
            rf"\1.save('{fig_path}')",
            modified_code,
        )

        # Use non-interactive backend to avoid display issues.
        modified_code = "import matplotlib\nmatplotlib.use('Agg')\n" + modified_code

        script_path = os.path.join(tmpdir, "render.py")
        with open(script_path, "w") as f:
            f.write(modified_code)

        subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            timeout=60,
            cwd=tmpdir,
        )

        if os.path.exists(fig_path):
            Image.open(fig_path).save(output_path)
            return True

    return False


def compile_code(code: str, fmt: str, output_path: str) -> bool:
    """Route to the appropriate compiler, falling back to trying all."""
    compilers = {
        "tikz": compile_tikz,
        "svg": compile_svg,
        "python": compile_python,
    }

    compiler = compilers.get(fmt)
    if compiler is None:
        for _name, comp in compilers.items():
            try:
                if comp(code, output_path):
                    return True
            except Exception:
                continue
        return False

    try:
        return compiler(code, output_path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-prompt processing
# ---------------------------------------------------------------------------

def process_prompt(client, model_cfg, image_id, prompt, output_dir):
    """Process a single prompt: generate -> (compile) -> save PNG.

    Returns (log_entry, status, format) -- thread-safe, no shared mutable state.
    """
    img_path = os.path.join(output_dir, f"{image_id}.png")

    # Skip if already generated and valid.
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            img.verify()
            return None, "cached", None
        except Exception:
            os.remove(img_path)  # corrupt PNG, regenerate

    # Image generation pathway: API -> binary image -> save PNG directly.
    if model_cfg.get("type") == "image_gen":
        for attempt in range(5):
            try:
                image_bytes = generate_image(client, model_cfg, prompt)
                img = Image.open(BytesIO(image_bytes))
                img.save(img_path, "PNG")
                log_entry = {
                    "image_id": image_id,
                    "format_detected": "image_gen",
                    "status": "success",
                }
                return log_entry, "success", "image_gen"
            except Exception as e:
                err = str(e)
                retryable = "429" in err or "500" in err or "502" in err or "503" in err or "timeout" in err.lower()
                credit_error = "402" in err or "Insufficient credits" in err
                if attempt < 4 and (retryable or credit_error):
                    wait = 30 if credit_error else 2 ** attempt
                    time.sleep(wait)
                    continue
                return {"image_id": image_id, "error": err}, "api_error", None
        return {"image_id": image_id, "error": "All retries failed"}, "api_error", None

    # Code generation pathway: API -> code -> extract -> compile -> PNG.
    raw_response = None
    for attempt in range(5):
        try:
            raw_response = generate_code(client, model_cfg, prompt)
            break
        except Exception as e:
            err = str(e)
            retryable = "429" in err or "500" in err or "502" in err or "503" in err or "timeout" in err.lower()
            credit_error = "402" in err or "Insufficient credits" in err
            if attempt < 4 and (retryable or credit_error):
                wait = 30 if credit_error else 2 ** attempt
                time.sleep(wait)
                continue
            return {"image_id": image_id, "error": err}, "api_error", None

    if raw_response is None:
        return {"image_id": image_id, "error": "All retries failed"}, "api_error", None

    code, fmt = extract_code_block(raw_response)

    log_entry = {
        "image_id": image_id,
        "format_detected": fmt,
        "code": code,
        "raw_response": raw_response,
    }

    success = compile_code(code, fmt, img_path)
    log_entry["status"] = "success" if success else "compile_error"

    return log_entry, log_entry["status"], fmt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate diagram images from prompts")
    parser.add_argument("--model", default="deepseek-v3", choices=list(MODELS.keys()))
    parser.add_argument("--csv", default="data/concise_prompts.csv",
                        help="Curated prompts CSV (from the curation pipeline)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/<model>)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N prompts")
    parser.add_argument("--offset", type=int, default=0, help="Start from prompt N")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    model_cfg = MODELS[args.model]
    output_dir = args.output_dir or os.path.join("outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    if args.offset:
        prompts = prompts[args.offset:]
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Model: {args.model} ({model_cfg['model_id']})")
    print(f"Prompts: {len(prompts)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")

    client = get_client(model_cfg)

    logs = []
    stats = {"success": 0, "compile_error": 0, "api_error": 0, "cached": 0}
    format_counts = {}

    if args.workers <= 1:
        for row in tqdm(prompts, desc=f"Generating ({args.model})"):
            log_entry, status, fmt = process_prompt(
                client, model_cfg,
                row["image_id"], row["concise_prompt"],
                output_dir,
            )
            stats[status] = stats.get(status, 0) + 1
            if log_entry:
                logs.append(log_entry)
            if fmt:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
            time.sleep(0.5)
    else:
        pbar = tqdm(total=len(prompts), desc=f"Generating ({args.model}, {args.workers}w)")

        def _worker(row):
            return process_prompt(
                client, model_cfg,
                row["image_id"], row["concise_prompt"],
                output_dir,
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, row): row for row in prompts}
            for future in as_completed(futures):
                try:
                    log_entry, status, fmt = future.result()
                except Exception:
                    status = "api_error"
                    log_entry, fmt = None, None

                stats[status] = stats.get(status, 0) + 1
                if log_entry:
                    logs.append(log_entry)
                if fmt:
                    format_counts[fmt] = format_counts.get(fmt, 0) + 1
                pbar.update(1)

        pbar.close()

    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, "w") as f:
        json.dump({"stats": stats, "format_counts": format_counts, "entries": logs}, f, indent=2)

    print("\n--- Results ---")
    print(f"Success:       {stats['success']}")
    print(f"Compile error: {stats['compile_error']}")
    print(f"API error:     {stats['api_error']}")
    print(f"Cached:        {stats['cached']}")
    print(f"Format distribution: {format_counts}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()

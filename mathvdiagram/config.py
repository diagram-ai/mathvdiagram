import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (set via environment variables or .env file) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Qwen / OpenRouter (legacy run_consensus only; not used in benchmarking pipeline) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
AGGREGATION_MAX_TOKENS = int(os.getenv("AGGREGATION_MAX_TOKENS", "3000"))

# --- Groq (Llama vision: Groq only; Qwen VL is not on Groq, use OpenRouter) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
# Text-only judge model for prompt synthesis (no vision needed at this step)
LLAMA_JUDGE_MODEL = os.getenv("LLAMA_JUDGE_MODEL", "llama-3.3-70b-versatile")
PROMPT_SYNTH_MAX_TOKENS = int(os.getenv("PROMPT_SYNTH_MAX_TOKENS", "200"))

# --- HuggingFace Dataset ---
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "MathLLMs/MathVision")
HF_SPLIT = os.getenv("HF_SPLIT", "test")

# --- DaTikZ v3 Dataset ---
DATIKZ_DATASET_NAME = os.getenv("DATIKZ_DATASET_NAME", "nllg/datikz-v3")
DATIKZ_SPLIT = os.getenv("DATIKZ_SPLIT", "train")

# --- Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# --- Output CSV filenames ---
CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "classification_results.csv")
SKIPPED_CSV = os.path.join(OUTPUT_DIR, "skipped_non_math.csv")
FULL_CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "full_classification.csv")
DESCRIPTIONS_CSV = os.path.join(OUTPUT_DIR, "descriptions.csv")
CONSENSUS_CSV = os.path.join(OUTPUT_DIR, "consensus_prompts.csv")
AGGREGATED_CSV = os.path.join(OUTPUT_DIR, "aggregated_descriptions.csv")
# Benchmarking pipeline paths (classification-free flow)
ALL_IMAGES_CSV = os.path.join(OUTPUT_DIR, "all_images.csv")
CONCISE_PROMPTS_CSV = os.path.join(OUTPUT_DIR, "concise_prompts.csv")
BENCHMARKING_REPORT = os.path.join(OUTPUT_DIR, "benchmarking_report.html")

# DaTikZ pipeline paths (isolated from MathVision outputs)
DATIKZ_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "datikz")
DATIKZ_ALL_IMAGES_CSV = os.path.join(DATIKZ_OUTPUT_DIR, "all_images.csv")
DATIKZ_DESCRIPTIONS_CSV = os.path.join(DATIKZ_OUTPUT_DIR, "descriptions.csv")
DATIKZ_CONCISE_PROMPTS_CSV = os.path.join(DATIKZ_OUTPUT_DIR, "concise_prompts.csv")
DATIKZ_REPORT = os.path.join(DATIKZ_OUTPUT_DIR, "benchmarking_report.html")

# --- Classification mode ---
# "taxonomy" = dataset_helper multi-model pipeline (default)
# "legacy"   = original GPT binary classify.py
CLASSIFICATION_MODE = os.getenv("CLASSIFICATION_MODE", "taxonomy")

# --- Model names ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_CLASSIFIER_MODEL = "gpt-5.2"
OPENAI_DESCRIPTION_MODEL = "gpt-5.2"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# --- Description settings ---
DESCRIPTION_MAX_TOKENS = int(os.getenv("DESCRIPTION_MAX_TOKENS", "2000"))
CLAUDE_DESCRIPTION_MODEL = os.getenv("CLAUDE_DESCRIPTION_MODEL", "claude-sonnet-4-20250514")

# --- Pipeline defaults ---
DELAY_BETWEEN_REQUESTS = int(os.getenv("DELAY_BETWEEN_REQUESTS", "3"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# --- Prompts ---
DETAILED_DESCRIPTION_PROMPT = """
You are a precision visual transcription engine for mathematical diagrams. Your task is to produce a description so detailed and exact that someone who cannot see the image could recreate it perfectly.

CRITICAL RULES:
- DO NOT solve the math problem. DO NOT answer the question. ONLY describe what you see.
- Describe the ACTUAL image, not what you think it should look like.
- When you see text, numbers, or labels in the image, transcribe them EXACTLY as written.
- If something is ambiguous or hard to read, say so explicitly rather than guessing.

WORK THROUGH THIS CHECKLIST. Answer each item that applies to this image. Skip items that are not relevant.

1. SHAPES & OBJECTS: What geometric shapes or figures are present? (triangles, circles, rectangles, polygons, curves, lines, points, etc.) How many of each?

2. LABELS & TEXT: List EVERY piece of text visible in the image — vertex labels (A, B, C...), point names, numbers, equations, annotations, axis labels, titles. Transcribe them exactly, including subscripts and superscripts.

3. MEASUREMENTS & VALUES: What numerical values are shown? Side lengths, angle measures (in degrees or radians), areas, distances, coordinates, data values. Where exactly is each value positioned?

4. GEOMETRIC NOTATION: What special mathematical marks are present? Right-angle squares, angle arcs with/without degree measures, congruence tick marks on sides, parallel arrows on sides, midpoint marks, perpendicularity symbols. Be specific about which elements carry which marks.

5. LINE STYLES: For EVERY line or edge in the diagram, note: solid, dashed, or dotted? Thick or thin? Colored (if not black)? Are there arrows on any lines? Which direction?

6. AXES & GRIDS: Is there a coordinate system? If yes: What are the axis labels? What are the min/max values on each axis? What is the tick spacing? Is there a grid? Are the axes at the origin or offset?

7. PLOTTED FUNCTIONS & DATA: Are there curves, data points, or shaded regions? Describe each: what equation or pattern does it follow (if identifiable), what color/style is it, what are key points (intercepts, extrema, intersections)?

8. CONNECTIONS & TOPOLOGY: Are there nodes connected by edges? Directed or undirected? Labeled? Is there a tree, graph, network, flowchart, or Venn diagram structure? Describe the connectivity.

9. 3D FEATURES: Is the figure three-dimensional? What is the shape? Which edges are hidden (typically dashed)? What is the approximate viewing angle? Are faces shaded or transparent?

10. SPATIAL LAYOUT: Describe the overall arrangement. What is at the top, bottom, left, right, center of the image? If there are multiple subfigures (A, B, C, D, E), describe each separately and note their relative positions.

11. COLORS & SHADING: What colors are used and where? Are any regions shaded, hatched, or filled? What shade of grey? Are there color-coded elements?

12. SCALE & PROPORTIONS: Are elements drawn to scale or schematic? Are there any measurement indicators or reference lengths? What are the approximate proportions between elements?

After completing the checklist, write a FINAL COMPREHENSIVE DESCRIPTION in natural language. This description must be detailed enough that a professional illustrator who cannot see the image could recreate every element with correct positions, labels, proportions, and styles. Start with the overall structure, then describe each element systematically.
"""

CATEGORY_HINTS = {
    "geometric_construction": "FOCUS ESPECIALLY ON: exact vertex labels, angle measures, congruence/parallel marks, right-angle indicators, and circle tangency points. Small tick marks and arc annotations are critical — do not skip them.",
    "coordinate_plot": "FOCUS ESPECIALLY ON: exact axis ranges and tick values, every function equation visible, precise coordinates of marked points, shaded region boundaries, and asymptotes. Be numerically exact — do not approximate.",
    "statistical_chart": "FOCUS ESPECIALLY ON: chart type (bar/line/pie/scatter/histogram/box), exact data values for every bar/point/slice, axis labels and scales, legend content, title, and color assignments per data series.",
    "schematic_diagram": "FOCUS ESPECIALLY ON: all node/cell labels and their values, the connectivity structure (what connects to what), directed vs undirected edges, any grouping or containment relationships, and the logical flow or ordering.",
    "3d_figure": "FOCUS ESPECIALLY ON: the 3D shape type, which edges are dashed (hidden) vs solid (visible), all vertex labels, face annotations, dimensional measurements, and whether it shows a net/unfolding or a perspective view.",
    "non_diagram": "Describe what you see as faithfully as possible, noting any mathematical content embedded in the illustration.",
}

CLASSIFIER_PROMPT = """
Look at this image carefully. Is it a PURE mathematical or geometric diagram?

A math diagram is something you would find in a geometry textbook or math paper — it must be
an abstract, technical drawing with NO real-world objects or decorative illustrations.

Answer YES ONLY if the image is:
- A geometric figure (triangles, circles, polygons with labeled angles/sides/vertices)
- A coordinate plane, graph, or function plot
- A formal mathematical diagram (Venn diagrams, number lines, vector diagrams)
- A chart or graph with axes (bar chart, pie chart, scatter plot)
- A 3D geometric shape (cube, sphere, cone with labeled dimensions)

Answer NO if the image contains ANY of these:
- Real-world objects used as decoration (flowers, trains, animals, buildings, people, daisies)
- Numbers inside illustrated objects (e.g. numbers in train cars, flowers, stars)
- Math puzzles presented with cartoon or decorative imagery
- Photographs or clipart of real-world scenes
- Ink blots, stains, or artistic elements overlaid on shapes
- Counting or arithmetic problems using pictures of objects

The key test: Could this diagram appear in a geometry textbook as-is, with NO illustrated objects?
If the image uses real-world objects to present a math problem, answer NO.

Respond with ONLY: YES or NO
"""

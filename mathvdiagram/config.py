import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (set via environment variables or .env file) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- HuggingFace Dataset ---
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "MathLLMs/MathVision")
HF_SPLIT = os.getenv("HF_SPLIT", "test")

# --- Paths ---
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# --- Output CSV filenames ---
CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "classification_results.csv")
SKIPPED_CSV = os.path.join(OUTPUT_DIR, "skipped_non_math.csv")
FULL_CLASSIFICATION_CSV = os.path.join(OUTPUT_DIR, "full_classification.csv")
DESCRIPTIONS_CSV = os.path.join(OUTPUT_DIR, "descriptions.csv")
CONSENSUS_CSV = os.path.join(OUTPUT_DIR, "consensus_prompts.csv")

# --- Classification mode ---
# "taxonomy" = dataset_helper multi-model pipeline (default)
# "legacy"   = original GPT binary classify.py
CLASSIFICATION_MODE = os.getenv("CLASSIFICATION_MODE", "taxonomy")

# --- Model names ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_CLASSIFIER_MODEL = "gpt-5.2"
OPENAI_DESCRIPTION_MODEL = "gpt-5.2"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# --- Pipeline defaults ---
DELAY_BETWEEN_REQUESTS = int(os.getenv("DELAY_BETWEEN_REQUESTS", "3"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# --- Prompts ---
DETAILED_DESCRIPTION_PROMPT = """
You are a mathematical typesetting engine. Analyze the provided image with extreme precision.
Deconstruct every visual element: coordinate systems (Cartesian/Polar/3D), functions plotted (with exact equations),
grid styles, axis labels with units, all text/numbers/symbols (including subscripts, superscripts, Greek letters),
line styles (solid/dashed/dotted), colors, thicknesses, arrow directions, shading patterns, proportions, spacing,
and positioning. Include all mathematical notation, tables, charts, and measurement indicators.
Output comprehensive instructions that a blind illustrator could use to perfectly recreate every detail of this diagram.
"""

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

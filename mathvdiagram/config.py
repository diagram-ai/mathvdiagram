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
DESCRIPTIONS_CSV = os.path.join(OUTPUT_DIR, "descriptions.csv")
CONSENSUS_CSV = os.path.join(OUTPUT_DIR, "consensus_prompts.csv")

# --- Model names ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_CLASSIFIER_MODEL = "gpt-4o-mini"
OPENAI_DESCRIPTION_MODEL = "gpt-4o"
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
Look at this image carefully. Does it show mathematical or geometric content?

Answer YES if the image contains:
- Geometric shapes (triangles, circles, polygons, 3D objects)
- Coordinate systems, graphs, or plots
- Mathematical equations, formulas, or symbols
- Charts, tables with numbers, or measurements
- Diagrams with angles, lines, or mathematical annotations

Answer NO if the image contains:
- Real-world objects (bicycles, trains, animals, people)
- Photographs of everyday scenes
- Artistic drawings or illustrations (unless they are geometric diagrams)
- Text-only content without mathematical symbols

Respond with ONLY: YES or NO
"""

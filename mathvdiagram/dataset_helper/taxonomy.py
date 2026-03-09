"""
Diagram taxonomy definitions and subject-to-priority mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaxonomyCategory:
    id: str
    name: str
    definition: str
    includes: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


DIAGRAM_TAXONOMY: dict[str, TaxonomyCategory] = {
    "geometric_construction": TaxonomyCategory(
        id="geometric_construction",
        name="Geometric Construction",
        definition=(
            "Abstract 2D shapes with formal math annotations. Must contain labeled "
            "vertices (A, B, C), marked angles, length annotations, or formal geometric "
            "notation (parallel marks, right-angle squares, congruence ticks). Shapes must "
            "be line-drawn figures, not filled illustrations."
        ),
        includes=[
            "Labeled triangles with vertices A, B, C and angle marks",
            "Circle theorems with tangent lines and inscribed angles",
            "Polygon constructions with side-length labels",
            "Compass-and-straightedge figures",
            "Quadrilaterals with parallel marks and congruence ticks",
        ],
        excludes=[
            "3D renders of real objects",
            "Tangram puzzles with colored fills",
            "Decorative shapes without math annotations",
        ],
        keywords=[
            "triangle", "circle", "polygon", "angle", "vertex", "perpendicular",
            "bisector", "tangent", "chord", "arc", "congruent", "parallel",
        ],
    ),
    "coordinate_plot": TaxonomyCategory(
        id="coordinate_plot",
        name="Coordinate Plot",
        definition=(
            "Images with explicit coordinate systems and axes. Must have labeled axes "
            "(x/y or named variables) plus plotted functions, data points, or defined "
            "regions. The coordinate system is the primary structural element."
        ),
        includes=[
            "Cartesian coordinate planes with plotted functions",
            "Polar coordinate plots",
            "Parametric curves on labeled axes",
            "Shaded regions between functions with axis labels",
            "Vector fields on coordinate grids",
        ],
        excludes=[
            "Grid puzzles without mathematical axes",
            "Pixel art or game boards",
            "Tables of numbers without plotted data",
        ],
        keywords=[
            "x-axis", "y-axis", "origin", "coordinate", "graph", "plot",
            "function", "curve", "parabola", "slope", "intercept",
        ],
    ),
    "statistical_chart": TaxonomyCategory(
        id="statistical_chart",
        name="Statistical Chart",
        definition=(
            "Charts and graphs presenting quantitative data. Must have labeled axes or "
            "categories with numerical values. The purpose is data presentation rather "
            "than geometric construction."
        ),
        includes=[
            "Bar charts with labeled categories and values",
            "Line graphs with time-series data",
            "Scatter plots with data points",
            "Pie charts with labeled sectors and percentages",
            "Histograms and box plots",
        ],
        excludes=[
            "Infographics with decorative elements",
            "Pictographs using real objects as units",
            "Tables without any graphical chart",
        ],
        keywords=[
            "bar chart", "pie chart", "histogram", "scatter", "data",
            "frequency", "percentage", "mean", "median", "distribution",
        ],
    ),
    "schematic_diagram": TaxonomyCategory(
        id="schematic_diagram",
        name="Schematic Diagram",
        definition=(
            "Formally structured diagrams showing mathematical relationships without "
            "coordinate axes. Uses standard mathematical visual conventions such as set "
            "notation, directed edges, or logical structure."
        ),
        includes=[
            "Venn diagrams with set labels",
            "Number lines with marked points and intervals",
            "Tree diagrams for probability or combinatorics",
            "Labeled directed/undirected graphs with vertices and edges",
            "Matrix representations and truth tables",
        ],
        excludes=[
            "Maze puzzles",
            "Road maps or transit maps",
            "Real-world process flowcharts",
        ],
        keywords=[
            "Venn", "number line", "tree", "graph", "vertex", "edge",
            "node", "set", "matrix", "truth table", "directed",
        ],
    ),
    "3d_figure": TaxonomyCategory(
        id="3d_figure",
        name="3D Figure",
        definition=(
            "Wireframe or technical drawings of 3D mathematical objects with formal "
            "annotations. Must show dimensional information, labeled vertices/edges, or "
            "cross-sections. The drawing style is technical, not photorealistic."
        ),
        includes=[
            "Labeled cubes, prisms, and pyramids with dimensions",
            "Cross-sections of 3D solids",
            "Nets of solids with fold lines",
            "Spheres with great circles and labeled radii",
            "Cylinders and cones with height/radius labels",
        ],
        excludes=[
            "Photos of real 3D objects",
            "Artistic renders without math labels",
            "Building-block illustrations without math labels",
        ],
        keywords=[
            "cube", "prism", "pyramid", "sphere", "cylinder", "cone",
            "volume", "surface area", "cross-section", "net", "face", "edge",
        ],
    ),
    "non_diagram": TaxonomyCategory(
        id="non_diagram",
        name="Non-Diagram",
        definition=(
            "Any image relying on real-world objects, illustrations, photographs, or "
            "decorative elements to present a math problem. Cannot be meaningfully "
            "recreated as a mathematical diagram by a text-to-image model. The math "
            "content is embedded in illustrated context rather than abstract notation."
        ),
        includes=[
            "Animal or object counting images",
            "Illustrated puzzles with cartoon imagery",
            "Photographs of real-world scenes",
            "Decorated number problems (numbers in flowers, trains, stars)",
            "Clock faces, coin images, map-based problems",
        ],
        excludes=[
            "Abstract geometric figures (even simple ones)",
            "Any image with formal axis labels or vertex notation",
        ],
        keywords=[
            "photo", "illustration", "cartoon", "animal", "flower",
            "train", "clock", "coin", "real-world", "counting",
        ],
    ),
}

VALID_CATEGORIES: list[str] = list(DIAGRAM_TAXONOMY.keys())

# Backwards compatibility alias
DiagramCategory = TaxonomyCategory

SUBJECT_PRIORITY: dict[str, str] = {
    # TIER 1: INCLUDE VIA METADATA — these are overwhelmingly formal diagrams
    # No API calls needed. ~95% diagram rate or higher.
    "analytic geometry": "high",
    "solid geometry": "high",
    "transformation geometry": "high",
    "descriptive geometry": "high",
    "topology": "high",
    "statistics": "high",
    "combinatorial geometry": "high",
    "metric geometry - length": "high",
    "metric geometry - area": "high",
    "metric geometry - angle": "high",

    # TIER 2: NEEDS SINGLE LLM INSPECTION — mixed content
    # These subjects contain both formal diagrams and illustrated content.
    # A single structured LLM call with the boolean checklist is sufficient.
    "algebra": "mixed",
    "graph theory": "mixed",
    "number theory": "mixed",
    "combinatorics": "mixed",
    "counting": "low",
    "arithmetic": "low",
    "logic": "low",
}


SUBJECT_CATEGORY_MAP: dict[str, str] = {
    # Existing
    "analytic geometry": "coordinate_plot",
    "solid geometry": "3d_figure",
    "metric geometry - length": "geometric_construction",
    "metric geometry - area": "geometric_construction",
    "transformation geometry": "geometric_construction",
    "descriptive geometry": "3d_figure",
    "topology": "schematic_diagram",
    # Newly promoted to high tier
    "statistics": "statistical_chart",
    "combinatorial geometry": "geometric_construction",
    "metric geometry - angle": "geometric_construction",
}


def get_subject_to_category_mapping() -> dict[str, str]:
    """Return the default mapping from MathVision subjects to taxonomy categories.

    Used for pre-filter assignment of high-priority subjects.
    """
    return dict(SUBJECT_CATEGORY_MAP)


def build_taxonomy_text() -> str:
    """Build a plain-text block describing all taxonomy categories (for LLM prompts)."""
    lines = []
    for cat in DIAGRAM_TAXONOMY.values():
        lines.append(f"Category: {cat.id}")
        lines.append(f"  Name: {cat.name}")
        lines.append(f"  Definition: {cat.definition}")
        lines.append(f"  Includes: {'; '.join(cat.includes)}")
        lines.append(f"  Excludes: {'; '.join(cat.excludes)}")
        lines.append("")
    return "\n".join(lines)

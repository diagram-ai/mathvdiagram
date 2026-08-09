# data/

Inputs to the benchmark stage.

| File | Produced by | Tracked? |
|------|-------------|----------|
| `concise_prompts.csv` | Stage 1 curation (`python -m mathvdiagram.pipeline`), then reviewed with `curate` | yes (small) |
| `excluded_ids.txt` | `python -m mathvdiagram.benchmark.curate` | yes (small) |
| `ground_truth/<id>.png` | Downloaded from the [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) dataset | no (gitignored) |

## Prompt CSV schema

`concise_prompts.csv` has one row per curated image:

| Column | Description |
|--------|-------------|
| `image_id` | MathVision image identifier |
| `question` | Original MathVision exam question |
| `category` | Mathematical category |
| `concise_prompt` | Natural-language description used as the model prompt |
| `n_descriptions_used` | How many VLM descriptions were aggregated into the prompt |

## Ground truth

`ground_truth/` holds the reference PNGs (`<image_id>.png`) that generated
images are scored against. These come from the MathVision source images and are
not committed here; render or export them locally before running `evaluate`.

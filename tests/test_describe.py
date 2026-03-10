import unittest
from unittest.mock import MagicMock, patch


class _DummyChatCompletionResponse:
    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class _DummyOpenAIClient:
    def __init__(self):
        self.last_kwargs = None

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.last_kwargs = kwargs
                return _DummyChatCompletionResponse("ok")

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


class TestDescribeProviders(unittest.TestCase):
    def test_build_description_prompt_includes_category_hint(self):
        from mathvdiagram.describe import _build_description_prompt

        with patch(
            "mathvdiagram.describe.config.DETAILED_DESCRIPTION_PROMPT", "BASE"
        ), patch(
            "mathvdiagram.describe.config.CATEGORY_HINTS",
            {"geometric_construction": "HINT"},
        ):
            out = _build_description_prompt("geometric_construction")
        self.assertEqual(out, "BASE\n\nHINT")
        with patch(
            "mathvdiagram.describe.config.DETAILED_DESCRIPTION_PROMPT", "BASE"
        ), patch(
            "mathvdiagram.describe.config.CATEGORY_HINTS",
            {"geometric_construction": "HINT"},
        ):
            out_unknown = _build_description_prompt("unknown")
        self.assertEqual(out_unknown, "BASE")

    def test_is_valid_description_rejects_short_or_error(self):
        from mathvdiagram.describe import _is_valid_description

        self.assertFalse(_is_valid_description(""))
        self.assertFalse(_is_valid_description("short"))
        self.assertFalse(_is_valid_description("[ERROR: something]"))
        self.assertFalse(_is_valid_description("[OPENAI ERROR: x]"))
        self.assertFalse(_is_valid_description("[LLAMA ERROR: x]"))
        self.assertTrue(_is_valid_description("x" * 60))
        self.assertFalse(_is_valid_description("[ERROR: x] " + "a" * 50))

    def test_get_qwen_description_builds_image_url_payload(self):
        from mathvdiagram import config
        from mathvdiagram.describe import get_qwen_description

        client = _DummyOpenAIClient()
        with patch.object(config, "QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-instruct"), patch.object(
            config, "DESCRIPTION_MAX_TOKENS", 123
        ), patch(
            "mathvdiagram.describe.call_with_retry", side_effect=lambda fn: fn()
        ):
            out = get_qwen_description(
                client,
                image_b64="AAA",
                question_text="what is shown?",
                category="geometric_construction",
            )

        self.assertEqual(out, "ok")
        self.assertIsNotNone(client.last_kwargs)
        self.assertEqual(client.last_kwargs["model"], "qwen/qwen3-vl-235b-a22b-instruct")
        self.assertEqual(client.last_kwargs["max_tokens"], 123)

        messages = client.last_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("DO NOT solve", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(messages[1]["content"], list)
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")
        self.assertEqual(
            messages[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAA",
        )

    def test_get_llama_description_uses_groq_model(self):
        from mathvdiagram import config
        from mathvdiagram.describe import get_llama_description

        client = _DummyOpenAIClient()
        with patch.object(
            config, "LLAMA_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        ), patch.object(config, "DESCRIPTION_MAX_TOKENS", 456), patch(
            "mathvdiagram.describe.call_with_retry", side_effect=lambda fn: fn()
        ):
            out = get_llama_description(
                client,
                image_b64="BBB",
                question_text="do not answer",
                category="coordinate_plot",
            )

        self.assertEqual(out, "ok")
        self.assertEqual(
            client.last_kwargs["model"], "meta-llama/llama-4-scout-17b-16e-instruct"
        )
        messages = client.last_kwargs["messages"]
        self.assertIn("ONLY describe", messages[0]["content"])
        self.assertEqual(
            messages[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,BBB",
        )

    def test_run_description_includes_qwen_and_llama_columns(self):
        import pandas as pd
        from mathvdiagram import config
        from mathvdiagram import describe as describe_mod

        df_in = pd.DataFrame(
            [
                {
                    "image_id": "img1",
                    "question": "q1",
                    "is_math": True,
                    "final_category": "geometric_construction",
                }
            ]
        )

        dummy_openai_client = _DummyOpenAIClient()
        dummy_qwen_client = _DummyOpenAIClient()
        dummy_llama_client = _DummyOpenAIClient()
        dummy_gemini_model = MagicMock()
        dummy_gemini_model.generate_content.return_value = MagicMock(text="gem")
        dummy_claude_client = MagicMock()
        dummy_claude_client.messages.create.return_value = MagicMock(content=[MagicMock(text="cla")])

        with patch.object(config, "CLASSIFICATION_CSV", "in.csv"), patch.object(
            config, "DESCRIPTIONS_CSV", "out.csv"
        ), patch.object(
            config, "DELAY_BETWEEN_REQUESTS", 0
        ), patch.object(
            config, "CHECKPOINT_EVERY", 1000
        ), patch(
            "mathvdiagram.describe.load_mathvision"
        ), patch(
            "mathvdiagram.describe.pd.read_csv", return_value=df_in
        ), patch(
            "mathvdiagram.describe.load_checkpoint", return_value=([], set())
        ), patch(
            "mathvdiagram.describe.save_checkpoint"
        ), patch(
            "mathvdiagram.describe.get_image_pil", return_value=MagicMock()
        ), patch(
            "mathvdiagram.describe.get_image_base64", return_value=("CCC", "image/png")
        ), patch(
            "mathvdiagram.describe.get_openai_client", return_value=dummy_openai_client
        ), patch(
            "mathvdiagram.describe.get_gemini_model", return_value=dummy_gemini_model
        ), patch(
            "mathvdiagram.describe.get_claude_client", return_value=dummy_claude_client
        ), patch(
            "mathvdiagram.describe.get_qwen_client", return_value=dummy_qwen_client
        ), patch(
            "mathvdiagram.describe.get_llama_client", return_value=dummy_llama_client
        ), patch(
            "mathvdiagram.describe.call_with_retry", side_effect=lambda fn: fn()
        ), patch(
            "time.sleep"
        ):
            out_df = describe_mod.run_description(input_csv="in.csv", resume=True, delay=0)

        self.assertIn("description_qwen", out_df.columns)
        self.assertIn("description_llama", out_df.columns)
        row = out_df.iloc[0].to_dict()
        self.assertEqual(row["description_openai"], "ok")
        self.assertEqual(row["description_gemini"], "gem")
        self.assertEqual(row["description_claude"], "cla")
        self.assertEqual(row["description_qwen"], "ok")
        self.assertEqual(row["description_llama"], "ok")


if __name__ == "__main__":
    unittest.main()


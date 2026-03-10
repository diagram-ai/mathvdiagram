import unittest
from unittest.mock import MagicMock, patch


class _DummyOpenAIClient:
    def __init__(self):
        self.last_kwargs = None

        class _Completions:
            def __init__(self, outer):
                self._outer = outer

            def create(self, **kwargs):
                self._outer.last_kwargs = kwargs
                return MagicMock(choices=[MagicMock(message=MagicMock(content="aggregated"))])

        class _Chat:
            def __init__(self, outer):
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


class TestConsensus(unittest.TestCase):
    def test_is_valid_description_rejects_errors_and_short(self):
        from mathvdiagram.consensus import _is_valid_description

        self.assertFalse(_is_valid_description(""))
        self.assertFalse(_is_valid_description(None))
        self.assertFalse(_is_valid_description("[OPENAI ERROR: x]"))
        self.assertFalse(_is_valid_description("short"))
        self.assertTrue(_is_valid_description("A" * 60))

    def test_aggregate_single_image_calls_qwen_with_image_and_descriptions(self):
        from mathvdiagram import config
        from mathvdiagram.consensus import aggregate_single_image

        client = _DummyOpenAIClient()
        row = {
            "question": "q",
            "category": "geometric_construction",
            "description_openai": "desc openai " * 10,
            "description_gemini": "desc gemini " * 10,
            "description_claude": "desc claude " * 10,
        }
        with patch.object(config, "QWEN_MODEL", "qwen/vl"), patch.object(
            config, "AGGREGATION_MAX_TOKENS", 2000
        ), patch(
            "mathvdiagram.consensus.call_with_retry", side_effect=lambda fn, **kw: fn()
        ):
            out = aggregate_single_image(
                client, row, image_b64="b64", media_type="image/png"
            )

        self.assertEqual(out, "aggregated")
        self.assertEqual(client.last_kwargs["model"], "qwen/vl")
        self.assertEqual(client.last_kwargs["max_tokens"], 2000)
        messages = client.last_kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertIn("image/png;base64,b64", content[1]["image_url"]["url"])

    def test_aggregate_single_image_returns_error_when_no_valid_descriptions(self):
        from mathvdiagram.consensus import aggregate_single_image

        client = _DummyOpenAIClient()
        row = {
            "question": "q",
            "category": "unknown",
            "description_openai": "[ERROR]",
            "description_gemini": "",
            "description_claude": "short",
        }
        out = aggregate_single_image(client, row, "b64", "image/png")
        self.assertIn("AGGREGATION_ERROR", out)
        self.assertIn("No valid descriptions", out)


if __name__ == "__main__":
    unittest.main()

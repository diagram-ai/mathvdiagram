import unittest
from unittest.mock import patch


class TestApiClients(unittest.TestCase):
    @patch("mathvdiagram.api_clients.OpenAI")
    def test_get_llama_client_uses_openrouter_base_url(self, OpenAI_mock):
        from mathvdiagram import config
        from mathvdiagram.api_clients import get_llama_client

        with patch.object(config, "OPENROUTER_API_KEY", "or-key"), patch.object(
            config, "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
        ):
            _ = get_llama_client()

        OpenAI_mock.assert_called_once_with(
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )

    def test_get_llama_client_requires_openrouter_key(self):
        from mathvdiagram import config
        from mathvdiagram.api_clients import get_llama_client

        with patch.object(config, "OPENROUTER_API_KEY", ""):
            with self.assertRaises(ValueError) as ctx:
                get_llama_client()
        self.assertIn("OPENROUTER_API_KEY not set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


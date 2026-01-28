import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestActionFormattingWrapper(unittest.TestCase):
    def test_prefers_last_bracketed_line(self):
        from mindgames.envs.Hanabi.env import HanabiEnv
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(HanabiEnv())
        out = env.action("some reasoning...\n[Discard] 0\nmore text\n")
        self.assertEqual(out, "[Discard] 0")

    def test_extracts_bracketed_action_after_prefix(self):
        from mindgames.envs.Hanabi.env import HanabiEnv
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(HanabiEnv())
        out = env.action("Final Answer: [Play] 3")
        self.assertEqual(out, "[Play] 3")

    def test_normalizes_unbracketed_action(self):
        from mindgames.envs.Hanabi.env import HanabiEnv
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(HanabiEnv())
        out = env.action("Discard 0")
        self.assertEqual(out, "[Discard] 0")

    def test_empty_action(self):
        from mindgames.envs.Hanabi.env import HanabiEnv
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(HanabiEnv())
        out = env.action("\n\n  \n")
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()


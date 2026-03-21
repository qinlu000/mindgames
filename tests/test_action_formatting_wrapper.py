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
        import mindgames as mg
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(mg.make("MiniHanabi-v0-raw"))
        out = env.action("some reasoning...\n[Discard] 0\nmore text\n")
        self.assertEqual(out, "[Discard] 0")

    def test_extracts_bracketed_action_after_prefix(self):
        import mindgames as mg
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(mg.make("MiniHanabi-v0-raw"))
        out = env.action("Final Answer: [Play A]")
        self.assertEqual(out, "[Play A]")

    def test_normalizes_unbracketed_action(self):
        import mindgames as mg
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(mg.make("MiniHanabi-v0-raw"))
        out = env.action("Discard A")
        self.assertEqual(out, "[Discard A]")

    def test_empty_action(self):
        import mindgames as mg
        from mindgames.wrappers import ActionFormattingWrapper

        env = ActionFormattingWrapper(mg.make("MiniHanabi-v0-raw"))
        out = env.action("\n\n  \n")
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()

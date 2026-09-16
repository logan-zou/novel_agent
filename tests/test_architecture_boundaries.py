import ast
import unittest
from pathlib import Path


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_runtime_has_no_novel_or_external_sdk_dependency(self):
        tree = ast.parse(Path("src/harness/runtime.py").read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        forbidden = ("src.novel", "src.config", "openai", "langfuse")
        self.assertFalse([name for name in imports if name.startswith(forbidden)])


if __name__ == "__main__":
    unittest.main()

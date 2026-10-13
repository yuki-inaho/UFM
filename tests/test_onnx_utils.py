"""Unit tests for the ONNX helpers (tiny graphs; exports skip when onnx is absent)."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnx")

from uniflowmatch.utils.onnx import onnx_size_report, simplify_onnx  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.linear(x)) + 0.5 * torch.relu(self.linear(x))


def _export_tiny(path: Path) -> Path:
    model = _Tiny().eval()
    torch.onnx.export(
        model,
        (torch.randn(2, 8),),
        str(path),
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
    )
    return path


def test_onnx_size_report_counts_graph(tmp_path):
    onnx_path = _export_tiny(tmp_path / "tiny.onnx")
    report = onnx_size_report(onnx_path)
    assert report["bytes"] > 0
    assert report["external_bytes"] == 0
    assert report["total_bytes"] == report["bytes"]
    assert report["nodes"] > 0
    assert report["initializers"] >= 2


def test_simplify_onnx_writes_loadable_model(tmp_path):
    pytest.importorskip("onnxsim")
    import onnx

    onnx_path = _export_tiny(tmp_path / "tiny.onnx")
    simplified_path = simplify_onnx(onnx_path)
    assert simplified_path.is_file()
    assert simplified_path != onnx_path

    original = onnx.load(str(onnx_path))
    simplified = onnx.load(str(simplified_path))
    assert len(simplified.graph.node) <= len(original.graph.node)

    report = onnx_size_report(simplified_path)
    assert report["nodes"] == len(simplified.graph.node)
    assert report["total_bytes"] > 0

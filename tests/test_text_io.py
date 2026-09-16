# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Tests for magic-byte based text resource I/O."""

from pathlib import Path

import zstandard

from opendde.utils.text_io import read_text, write_zstd_text_atomic


def test_reads_zstd_by_magic_and_plain_text_with_zst_suffix(tmp_path: Path):
    compressed = tmp_path / "alignment.a3m"
    write_zstd_text_atomic(compressed, ">query\nACD\n")
    assert compressed.read_bytes().startswith(b"\x28\xb5\x2f\xfd")
    assert read_text(compressed) == ">query\nACD\n"

    manually_replaced = tmp_path / "alignment.a3m.zst"
    manually_replaced.write_text(">replacement\nGGG\n", encoding="utf-8")
    assert read_text(manually_replaced) == ">replacement\nGGG\n"


def test_zstd_output_is_a_standard_frame(tmp_path: Path):
    output = tmp_path / "template.cif.zst"
    write_zstd_text_atomic(output, "data_template\n#\n")

    assert zstandard.ZstdDecompressor().decompress(output.read_bytes()) == (
        b"data_template\n#\n"
    )

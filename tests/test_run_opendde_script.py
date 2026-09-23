# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Behavior tests for the root run_opendde.sh convenience wrapper."""

from pathlib import Path
import subprocess


def test_run_opendde_forwards_wrapper_options_and_trailing_arguments(tmp_path: Path):
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    captured = tmp_path / "args.txt"
    fake = tmp_path / "fake-opendde"
    fake.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURED_ARGS"\n', encoding="utf-8"
    )
    fake.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "run_opendde.sh",
            "-i",
            str(input_path),
            "-o",
            str(tmp_path / "out"),
            "-D",
            "false",
            "-P",
            "true",
            "-r",
            "101,102",
            "-s",
            "3",
            "-m",
            "2025-01-02",
            "-S",
            "true",
            "-z",
            "false",
            "-f",
            "true",
            "-a",
            "true",
            "--",
            "--device",
            "cpu",
            "--load_checkpoint_path",
            str(tmp_path / "model with spaces.pt"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "OPENDDE_BIN": str(fake),
            "CAPTURED_ARGS": str(captured),
        },
    )

    assert result.returncode == 0, result.stderr
    assert captured.read_text().splitlines() == [
        "pred",
        "--input",
        str(input_path),
        "--out_dir",
        str(tmp_path / "out"),
        "--run_data_pipeline",
        "false",
        "--run_inference",
        "true",
        "--sample",
        "3",
        "--max_template_date",
        "2025-01-02",
        "--skip",
        "true",
        "--compress_fold_input",
        "false",
        "--compress_full_confidence",
        "true",
        "--need_atom_confidence",
        "true",
        "--write_input_json",
        "true",
        "--seeds",
        "101,102",
        "--device",
        "cpu",
        "--load_checkpoint_path",
        str(tmp_path / "model with spaces.pt"),
    ]


def test_run_opendde_help_and_shell_syntax():
    syntax = subprocess.run(
        ["bash", "-n", "run_opendde.sh"], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        ["bash", "run_opendde.sh", "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "OPENDDE_BIN" in help_result.stdout
    assert "--compress_fold_input" in help_result.stdout
    assert "--compress_full_confidence" in help_result.stdout


def test_run_opendde_compression_default_and_explicit_false(tmp_path: Path):
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    captured = tmp_path / "args.txt"
    fake = tmp_path / "fake-opendde"
    fake.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURED_ARGS"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)
    environment = {
        "PATH": "/usr/bin:/bin",
        "OPENDDE_BIN": str(fake),
        "CAPTURED_ARGS": str(captured),
    }

    for extra_args, expected in (([], "false"), (["-f", "true"], "true")):
        result = subprocess.run(
            [
                "bash",
                "run_opendde.sh",
                "-i",
                str(input_path),
                "-o",
                str(tmp_path / "out"),
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        assert result.returncode == 0, result.stderr
        arguments = captured.read_text().splitlines()
        option = arguments.index("--compress_full_confidence")
        assert arguments[option + 1] == expected


def test_run_opendde_rejects_removed_write_now_option(tmp_path: Path):
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    fake = tmp_path / "fake-opendde"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "run_opendde.sh",
            "-i",
            str(input_path),
            "-o",
            str(tmp_path / "out"),
            "-w",
            "false",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "OPENDDE_BIN": str(fake)},
    )

    assert result.returncode == 2


def test_run_opendde_forwards_independent_write_switch(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("[]")
    fake = tmp_path / "opendde"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "run_opendde.sh",
            "-i",
            str(source),
            "-o",
            str(tmp_path / "out"),
            "-D",
            "false",
            "-J",
            "true",
        ],
        env={"PATH": "/usr/bin:/bin", "OPENDDE_BIN": str(fake)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[args.index("--write_input_json") + 1] == "true"

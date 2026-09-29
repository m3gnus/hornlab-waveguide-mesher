"""The ``hornlab-waveguide`` entry point itself, not just the functions it calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hornlab_mesher import cli

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.toml"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_main_builds_every_shipped_example(example, tmp_path, capfd):
    output = tmp_path / "horn.msh"

    assert cli.main([str(example), "-o", str(output)]) == 0

    out, err = capfd.readouterr()
    assert output.is_file() and output.stat().st_size > 0
    assert out.strip().startswith(f"Wrote {output} (")
    # Native libraries may write their own notices to stderr (notably on
    # Windows); a build that worked prints no traceback and no error line.
    assert "Traceback" not in err
    assert "error:" not in err


def test_print_summary_stdout_is_exactly_the_json_summary(tmp_path, capfd):
    output = tmp_path / "horn.msh"
    summary_file = tmp_path / "nested" / "summary.json"

    code = cli.main(
        [str(EXAMPLES[0]), "-o", str(output), "--print-summary",
         "--summary", str(summary_file)]
    )

    out, _err = capfd.readouterr()
    assert code == 0
    printed = json.loads(out)
    assert printed == json.loads(summary_file.read_text(encoding="utf-8"))
    assert printed["mesh_path"] == str(output)
    assert printed["n_triangles"] > 0


def test_print_summary_with_step_keeps_native_output_off_stdout(tmp_path, capfd):
    """OpenCASCADE writes its STEP transfer banner to file descriptor 1.

    With ``--print-summary`` that banner and the "Wrote ...step" line used to
    lead stdout, so ``json.load(stdout)`` failed. Both belong on stderr.
    """

    step = tmp_path / "horn.step"
    output = tmp_path / "horn.msh"

    code = cli.main(
        [str(EXAMPLES[0]), "--step", str(step), "-o", str(output), "--print-summary"]
    )

    out, err = capfd.readouterr()
    assert code == 0
    assert json.loads(out)["mesh_path"] == str(output)
    assert f"Wrote {step}" in err
    assert step.is_file()


def test_step_only_run_skips_the_mesh(tmp_path, capfd):
    step = tmp_path / "horn.step"

    assert cli.main([str(EXAMPLES[0]), "--step", str(step)]) == 0

    out, _err = capfd.readouterr()
    assert f"Wrote {step}" in out
    assert step.is_file()
    assert not list(tmp_path.glob("*.msh"))


def test_step_only_print_summary_prints_the_step_json(tmp_path, capfd):
    step = tmp_path / "horn.step"
    summary_file = tmp_path / "summary.json"

    code = cli.main(
        [str(EXAMPLES[0]), "--step", str(step), "--print-summary",
         "--summary", str(summary_file)]
    )

    out, err = capfd.readouterr()
    assert code == 0
    printed = json.loads(out)
    assert printed == json.loads(summary_file.read_text(encoding="utf-8"))
    assert printed["step_path"] == str(step)
    assert printed["n_faces"] > 0
    assert printed["units"] == "mm"
    assert f"Wrote {step}" in err
    assert not list(tmp_path.glob("*.msh"))


def test_invalid_config_is_one_line_and_exit_2(tmp_path, capfd):
    config = tmp_path / "bad.toml"
    config.write_text('formula = "NOT-A-FORMULA"\n', encoding="utf-8")

    code = cli.main([str(config), "-o", str(tmp_path / "x.msh")])

    out, err = capfd.readouterr()
    assert code == 2
    assert out == ""
    assert err.startswith("hornlab-waveguide: error: formula must be")
    assert "Traceback" not in err


def test_config_without_output_path_is_refused(tmp_path, capfd):
    config = tmp_path / "no-output.toml"
    config.write_text('formula = "OSSE"\n[profile]\nL_mm = 80.0\n', encoding="utf-8")

    code = cli.main([str(config)])

    _out, err = capfd.readouterr()
    assert code == 2
    assert "output path required; set output.path or pass -o/--output" in err


def test_unreadable_config_file_is_exit_2(tmp_path, capfd):
    code = cli.main([str(tmp_path / "missing.toml"), "-o", str(tmp_path / "x.msh")])

    _out, err = capfd.readouterr()
    assert code == 2
    assert err.startswith("hornlab-waveguide: error:")


def test_internal_error_is_not_disguised_as_a_config_error(
    tmp_path, capfd, monkeypatch
):
    """A defect used to print ``error: 'x'`` with exit 2, like a bad config."""

    def broken(*_args, **_kwargs):
        raise KeyError("x")

    monkeypatch.setattr(cli, "build_from_config", broken)

    code = cli.main([str(EXAMPLES[0]), "-o", str(tmp_path / "x.msh")])

    _out, err = capfd.readouterr()
    assert code == 1
    assert "Traceback" in err and "KeyError: 'x'" in err
    assert "internal error" in err


def test_parser_documents_every_option():
    help_text = cli._build_parser().format_help()
    for option in ("--output", "--summary", "--print-summary", "--allow-large-mesh",
                   "--step", "--step-keep-throat"):
        assert option in help_text

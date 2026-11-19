"""
Other papers' published scores reach the prose only as macros (19 Nov: the
Related-work paragraph places our scale beside T-Mem's and JustMem's on the
same 401 Cognitive items). They are constants, so they must be emitted even
with no result files at all, and with exactly the published values.
"""
import contextlib
import io
import pathlib
import runpy

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def test_published_constants_emitted_without_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "paper").mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            runpy.run_path(str(SCRIPTS / "paper_numbers.py"), run_name="__main__")
        except SystemExit:
            pass
    tex = (tmp_path / "paper" / "numbers.tex").read_text()
    for name, value in (("LiCogMax", "26.06"), ("TMemCog", "74.81"), ("JustMemCog", "62.59")):
        assert "\\newcommand{\\%s}{%s}\n" % (name, value) in tex

import subprocess
import sys


def test_cli_help_runs():
    # The command-line tool starts and lists its commands.
    result = subprocess.run(
        [sys.executable, "-m", "cmx", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "train" in result.stdout


def test_import_does_not_load_torch():
    # torch is ~1 GB. The live demo must never load it, so importing the
    # package must not pull it in. A fresh Python process is used because
    # another test could already have imported torch in this one.
    code = "import sys, cmx, cmx.__main__; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

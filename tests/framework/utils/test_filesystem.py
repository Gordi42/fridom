"""Tests for the filesystem utilities."""
import os
from pathlib import Path

import fridom.framework as fr
import fridom.framework.utils.filesystem as filesystem_module


# ================================================================
#  chdir_to_submit_dir
# ================================================================
def test_chdir_to_submit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SUBMIT_DIR", str(tmp_path))
    original = Path.cwd()
    try:
        fr.utils.chdir_to_submit_dir()
        assert Path.cwd() == tmp_path
    finally:
        os.chdir(original)


# ================================================================
#  stdout_is_file
# ================================================================
def test_stdout_is_file(monkeypatch):
    monkeypatch.setattr(filesystem_module, "get_ipython", lambda: None)

    monkeypatch.setattr(os, "isatty", lambda _fd: True)
    assert not fr.utils.stdout_is_file()

    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    assert fr.utils.stdout_is_file()


def test_stdout_is_not_file_in_ipython(monkeypatch):
    def fake_ipython():
        return object()

    monkeypatch.setattr(os, "isatty", lambda _fd: False)
    monkeypatch.setattr(filesystem_module, "get_ipython", fake_ipython)
    assert not fr.utils.stdout_is_file()

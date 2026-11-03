"""The source checkout must retain BAGEL's data loading code when cloned."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_root_data_ignore_does_not_hide_vendored_bagel_data(tmp_path):
    if not shutil.which('git'):
        pytest.skip('git unavailable')
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    (tmp_path/'.gitignore').write_bytes((root/'.gitignore').read_bytes())
    result=subprocess.run(['git','-C',str(tmp_path),'check-ignore','--stdin','-z'],
                          input=b'data/private.jsonl\0vendor/bagel/data/dataset_base.py\0',
                          capture_output=True)
    assert result.stdout.split(b'\0')==[b'data/private.jsonl',b'']

import ast
from pathlib import Path
import subprocess
import sys


def test_no_legacy_planner_imports():
    allowed={'model_test.robot_model_spec_v5'}
    for path in (Path(__file__).parents[1]).glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names=[]
            if isinstance(node,ast.Import): names=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom) and node.module: names=[node.module]
            for name in names:
                if name.startswith(('ff_prior','v6_lite','tracking_diffusion','model_test')):
                    assert name in allowed, (path,name)


def test_cli_help_and_unimplemented_fail():
    for args,expected in [(['--help'],0),(['plan','--task','unimplemented.json'],2),(['test','--suite','invalid'],2)]:
        p=subprocess.run([sys.executable,'-m','ffgd.cli',*args],capture_output=True,text=True)
        assert p.returncode==expected,p.stdout+p.stderr

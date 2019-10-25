"""
Helper script to combine coverage (as codecov seems to have problems...).
Written in python to be cross-platform
"""

import argparse
from os import chdir, listdir, environ
from pathlib import Path
import platform
from pprint import pprint
from subprocess import run, PIPE, CalledProcessError
import sys

THIS_FILE = Path(__file__)
GIT_MAIN_DIR = THIS_FILE.parent.parent.resolve()
COVERAGE_DIR = GIT_MAIN_DIR.joinpath('.coverage_dir')
PYVERSION = '{}.{}'.format(sys.version_info[0], sys.version_info[1])


def msg(*args):
    print('ERR:', *args, file=sys.stderr)


def pmsg(*args):
    pprint(*args, stream=sys.stderr)


def run_with_python(args, **kwargs):
    if platform.system() == 'Windows':
        exe = ['py', '-' + PYVERSION, '-m']
    else:
        exe = []
    cmd = exe + args
    msg("Running:", *cmd)
    try:
        res = run(cmd, check=True, stdout=PIPE, stderr=PIPE, **kwargs)
    except CalledProcessError as e:
        msg("STDOUT:")
        sys.stdout.buffer.write(e.stdout)
        msg("STDERR:")
        sys.stderr.buffer.write(e.stderr)
        raise
    else:
        msg("STDOUT:")
        sys.stdout.buffer.write(res.stdout)
        msg("STDERR:")
        sys.stderr.buffer.write(res.stderr)
        return res


def send_coverage(*, workdir, coverage_files, codecov_token):
    chdir(workdir)
    run_with_python(['coverage', 'combine'] + coverage_files)
    msg(f"Combined coverage, listing {GIT_MAIN_DIR}")
    pmsg(sorted(listdir(GIT_MAIN_DIR)))
    run_with_python(['coverage', 'xml', '--ignore-errors'])
    msg(f"Created coverage xml, listing {GIT_MAIN_DIR}")
    pmsg(sorted(listdir(GIT_MAIN_DIR)))
    codecov_args = []
    if codecov_token is not None:
        codecov_args.extend(['-t', codecov_token])
    codecov_args.extend(['--file', 'coverage.xml'])
    run_with_python(['codecov'] + codecov_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codecov-token", default=None)
    args = parser.parse_args()
    msg(f"Working in {GIT_MAIN_DIR}, listing coverage dir {COVERAGE_DIR}")
    pmsg(sorted(listdir(COVERAGE_DIR)))
    coverage_files = [str(f) for f in COVERAGE_DIR.glob('coverage-*')]
    if coverage_files:
        send_coverage(
            workdir=GIT_MAIN_DIR,
            coverage_files=coverage_files,
            codecov_token=args.codecov_token,
        )
    else:
        msg("No coverage files found")
        if environ.get("H5PY_ENFORCE_COVERAGE") is not None:
            raise RuntimeError(
                "Coverage required, no coverage found, failing..."
            )


if __name__ == '__main__':
    main()

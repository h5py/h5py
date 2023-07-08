"""
Helper script to combine coverage (as codecov seems to have problems...).
Written in python to be cross-platform
"""

import argparse
import os
from os import chdir, environ
from pathlib import Path
import platform
from pprint import pprint
import signal
from subprocess import PIPE, Popen, CalledProcessError, TimeoutExpired
import sys

THIS_FILE = Path(__file__)
GIT_MAIN_DIR = THIS_FILE.parent.parent.resolve()
COVERAGE_DIR = GIT_MAIN_DIR.joinpath('.coverage_dir')
PYVERSION = '{}.{}'.format(sys.version_info[0], sys.version_info[1])


def msg(*args):
    print('ERR:', *args, file=sys.stderr, flush=True)


def pmsg(*args):
    pprint(*args, stream=sys.stderr)
    sys.stderr.flush()


def _run(proc: Popen, timeout):
    """Run process, with several steps of signalling if timeout is reached"""
    try:
        return proc.wait(timeout=timeout)
    except TimeoutExpired:
        pass
    if sys.platform != 'win32':
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=5)
        except TimeoutExpired:
            pass

    proc.terminate()  # SIGTERM
    try:
        return proc.wait(timeout=5)
    except TimeoutExpired:
        pass

    proc.kill()  # SIGKILL
    return proc.wait(timeout=5)

def run_with_python(args, timeout=30, **kwargs):
    if platform.system() == 'Windows':
        exe = ['py', '-' + PYVERSION, '-m']
    else:
        exe = []
    cmd = exe + args
    msg("Running:", *cmd)

    proc = Popen(cmd, stdin=PIPE, **kwargs)
    proc.stdin.close()
    retcode = _run(proc, timeout)

    if retcode != 0:
        raise CalledProcessError(retcode, cmd)


def get_pr_azure_ci():
    # Get pull request number on Azure Pipelines
    # Return None if not on Azure or not a PR build.
    return (
        os.environ.get('SYSTEM_PULLREQUEST_PULLREQUESTNUMBER')
        or os.environ.get('SYSTEM_PULLREQUEST_PULLREQUESTID')
        or None
    )


def send_coverage(*, workdir, coverage_files, codecov_token):
    chdir(workdir)
    run_with_python(['coverage', 'combine'] + coverage_files)
    msg(f"Combined coverage")
    run_with_python(['coverage', 'xml', '--ignore-errors'])
    msg(f"Created coverage xml")

    # Upload coverage.xml to codecov
    codecov_args = []
    if codecov_token is not None:
        codecov_args.extend(['-t', codecov_token])
    codecov_args.extend(['--file', 'coverage.xml'])
    pr_num = get_pr_azure_ci()
    if pr_num is not None:
        codecov_args.extend(['--pr', pr_num])
    run_with_python(['codecov'] + codecov_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codecov-token", default=None)
    args = parser.parse_args()

    msg(f"Working in {GIT_MAIN_DIR}, looking for coverage files...")
    coverage_files = [str(f) for f in COVERAGE_DIR.glob('coverage-*')]
    pmsg(sorted(coverage_files))
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

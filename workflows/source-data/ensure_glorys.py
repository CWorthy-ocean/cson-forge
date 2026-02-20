#!/usr/bin/env python3
"""Prepare GLORYS source data for cson-forge, with optional batch SLURM submission."""

import argparse
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

from dask.distributed import Client
import cson_forge


def make_script(year, script_dir=None, test=False):
    """Generate a bash script to prepare GLORYS data for a single year.

    Args:
        year: Year to prepare.
        script_dir: Directory containing ensure_glorys.py. Defaults to cwd.
        test: If True, add --test to the python command (print config only).

    Returns:
        Path to the generated script file.
    """
    path_work = cson_forge.config.paths.scratch / "source-data-setup"
    path_logs = path_work / "logs"
    path_logs_str = str(path_logs)
    os.makedirs(path_logs_str, exist_ok=True)

    script_dir = script_dir or os.getcwd()

    account = cson_forge.config.machine_config.account
    queue_name = cson_forge.config.machine_config.queues.get("shared")
    if queue_name is None:
        raise ValueError(f"shared queue not found")

    sbatch_header = f"""#!/bin/bash
        #SBATCH --job-name ensure-glorys-{year}
        #SBATCH --account {account}
        #SBATCH --partition={queue_name}
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=16
        #SBATCH --time=06:00:00
        #SBATCH --output {path_logs_str}/ensure-glorys-{year}-%J.log
        #SBATCH --error {path_logs_str}/ensure-glorys-{year}-%J.log
        """

    test_mode = " --test" if test else ""
    script = textwrap.dedent(
        f"""\
        {sbatch_header}

        module load conda
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate cson-forge-v0

        cd {script_dir}
        python -u ensure_glorys.py --year {year}{test_mode}
        """
    )

    script_file = path_work / f"ensure-glorys-{year}.sh"
    script_file.write_text(script)
    return str(script_file)

def batch_download(year_start, year_end, test=False):
    """Submit or run GLORYS preparation for a range of years.

    When test=False, submits one SLURM job per year via sbatch.
    When test=True, runs each year's script directly with bash (no sbatch).

    Args:
        year_start: First year (inclusive).
        year_end: Last year (inclusive).
        test: If True, run scripts locally instead of submitting to SLURM.

    Returns:
        List of job IDs (sbatch) or status strings (test mode).
    """
    script_dir = str(Path(__file__).resolve().parent)
    jobids = []
    for year in range(year_start, year_end + 1):
        script_file = make_script(year, script_dir=script_dir, test=test)
        if test:
            result = subprocess.run(
                ["bash", script_file], capture_output=True, text=True
            )
            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
            print(f"Ran {script_file} for year {year}: {status}")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            jobids.append(status)
        else:
            result = subprocess.run(
                ["sbatch", script_file], capture_output=True, text=True
            )
            jobid = result.stdout.strip() if result.stdout else result.stderr.strip()
            print(f"Submitted job {script_file} for year {year}: {jobid}")
            jobids.append(jobid)
    
    verb = "Ran" if test else "Submitted"
    print(f"{verb} {len(jobids)} jobs for years {year_start} to {year_end}")
    return jobids


def main(year, test=False):
    """Prepare GLORYS source data for a single year.

    Args:
        year: Year to prepare.
        test: If True, print config and exit without downloading.
    """
    client = Client(n_workers=2, threads_per_worker=1)
    src = cson_forge.source_data.SourceData(
        datasets=["GLORYS"],
        clobber=False,
        start_time=datetime(year, 1, 1),
        end_time=datetime(year, 12, 31),
    )
    if test:
        print("Test mode: not preparing source data")
        print(src)
        return
    src.prepare_all()


def parse_args():
    """Parse command-line arguments for ensure_glorys."""
    parser = argparse.ArgumentParser(description="Prepare GLORYS source data.")
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to prepare.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: print config and exit without preparing data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.year, test=args.test)
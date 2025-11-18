from __future__ import annotations

import os
import shutil
from pathlib import Path

import subprocess
import sys
from dataclasses import dataclass

from datetime import datetime
import uuid

from typing import Any, Callable, Dict

import stat
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config


# =========================================================
# Data structures
# =========================================================


@dataclass
class RepoSpec:
    """
    Specification for a code repository used in the build.

    Parameters
    ----------
    name : str
        Short name for the repository (e.g., "roms", "marbl").
    url : str
        Git URL for the repository.
    default_dirname : str
        Default directory name under the code root where this repo
        will be cloned.
    checkout : str, optional
        Optional tag, branch, or commit to check out after cloning.
    """
    name: str
    url: str
    default_dirname: str
    checkout: str | None = None


@dataclass
class ModelSpec:
    """
    Description of a ROMS/MARBL model configuration.

    Parameters
    ----------
    name : str
        Logical name of the model (e.g., "roms-marbl").
    opt_base_dir : str
        Relative path (under model-configs) to the base configuration
        directory.
    conda_env : str
        Name of the conda environment used to build/run this model.
    repos : dict[str, RepoSpec]
        Mapping from repo name to its specification.
    """
    name: str
    opt_base_dir: str
    conda_env: str
    repos: Dict[str, RepoSpec]


# =========================================================
# Helper functions
# =========================================================


def _load_models_yaml(path: Path, model: str) -> ModelSpec:
    """
    Load repository specifications and model metadata from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the models.yaml file.
    model : str
        Name of the model block to load (e.g., "roms-marbl").

    Returns
    -------
    ModelSpec
        Parsed model specification including repository metadata.

    Raises
    ------
    KeyError
        If the requested model is not present in the YAML file.
    """
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    if model not in data:
        raise KeyError(f"Model '{model}' not found in models YAML file: {path}")

    repos: Dict[str, RepoSpec] = {}
    for key, val in data[model]["repos"].items():
        repos[key] = RepoSpec(
            name=key,
            url=val["url"],
            default_dirname=val.get("default_dirname", key),
            checkout=val.get("checkout"),
        )

    return ModelSpec(
        name=model,
        opt_base_dir=data[model]["opt_base_dir"],
        conda_env=data[model]["conda_env"],
        repos=repos,
    )


def _render_opt_base_dir_to_opt_dir(
    grid_name: str,
    parameters: Dict[str, Dict[str, Any]],
    opt_base_dir: Path,
    opt_dir: Path,
    overwrite: bool = False,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Stage and render model configuration templates using Jinja2.

    This function creates a working copy of the model configuration directory,
    renders selected files in place using the provided parameter dictionary,
    and preserves original file permissions. It is typically used to generate
    grid-specific ROMS/MARBL configuration files with substituted variables
    before compilation or execution.

    Workflow
    --------
    1. Copy the contents of ``opt_base_dir`` into ``opt_dir`` (optionally
       overwriting existing files).
    2. Initialize a Jinja2 environment rooted at ``opt_dir``.
    3. For each file listed in ``parameters``, treat it as a template and
       render using the provided context dictionary.
    4. Write rendered content back to the same path, preserving file
       permissions.

    Parameters
    ----------
    grid_name : str
        Name of the grid; used to ignore recursive copies of opt_<grid_name>.
    parameters : dict[str, dict[str, Any]]
        Mapping of relative filenames (e.g. ``"param.opt"``) to the
        template-variable dictionary used for rendering.
    opt_base_dir : Path
        Base (template) model configuration directory.
    opt_dir : Path
        Target directory where rendered configuration files will live.
    overwrite : bool, optional
        If True, allow existing contents in ``opt_dir`` to be overwritten.
    log_func : callable, optional
        Logging function used for messages (defaults to :func:`print`).

    Raises
    ------
    FileNotFoundError
        If a template file listed in ``parameters`` is missing from ``opt_dir``.
    jinja2.exceptions.UndefinedError
        If a template references a variable not defined in its context.
    """
    src = opt_base_dir.resolve()
    dst = opt_dir.resolve()

    if overwrite and dst.exists():
        log_func(f"[Render] Clearing existing opt_dir: {dst}")
        shutil.rmtree(dst)

    # Copy everything except an existing opt_<grid_name> directory
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(f"opt_{grid_name}"),
    )

    env = Environment(
        loader=FileSystemLoader(str(dst)),
        undefined=StrictUndefined,  # error on missing variables
        autoescape=False,           # plain text files
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )

    rendered_paths: list[str] = []

    for relname, context in parameters.items():
        relpath = Path(relname)
        target = dst / relpath
        if not target.exists():
            raise FileNotFoundError(f"Template not found in opt_dir: {target}")

        template = env.get_template(str(relpath.as_posix()))
        rendered_text = template.render(**context)

        try:
            orig_mode = target.stat().st_mode
        except FileNotFoundError:
            orig_mode = None

        target.write_text(rendered_text)

        if orig_mode is not None:
            target.chmod(stat.S_IMODE(orig_mode))

        rendered_paths.append(str(target))

    log_func("Rendered configuration files:")
    for f in rendered_paths:
        log_func(f"  - {f}")


def _run_logged(
    label: str,
    logfile: Path,
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Run a command, log stdout/stderr to a file, and fail loudly with context.

    All subprocess output is written only to ``logfile``. High-level status
    messages go through ``log_func``, which in :func:`build` is wired to
    write both to stdout and ``build.all.<token>.log``.

    Parameters
    ----------
    label : str
        Human-readable label describing this build step.
    logfile : Path
        Path to the log file that will capture stdout/stderr.
    cmd : list[str]
        Command and arguments to execute.
    env : dict[str, str] or None, optional
        Environment variables to pass to :class:`subprocess.Popen`. If None,
        the current process environment is used.
    log_func : callable, optional
        Logging function for high-level status messages.
    """
    log_func(f"[{label}] starting...")
    logfile.parent.mkdir(parents=True, exist_ok=True)

    with logfile.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        ret = proc.wait()

    if ret != 0:
        log_func(f"❌ {label} FAILED — see log: {logfile}")
        try:
            # Tail to stderr only; do not send to build.all log
            print(f"---- Last 50 lines of {logfile} ----", file=sys.stderr)
            with logfile.open() as f:
                lines = f.readlines()
            for line in lines[-50:]:
                sys.stderr.write(line)
            print("-------------------------------------", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"(could not read logfile: {e})", file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {ret}")

    log_func(f"[{label}] OK")


def _check_command_exists(name: str) -> None:
    """
    Raise a clear error if a command is not found in PATH.

    Parameters
    ----------
    name : str
        Name of the executable to look for (e.g., "conda").
    """
    from shutil import which

    if which(name) is None:
        raise RuntimeError(f"❌ Required command '{name}' not found in PATH.")


def _run(cmd: list[str], **kwargs: Any) -> str:
    """
    Convenience wrapper around :func:`subprocess.run` that returns stdout.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute.
    **kwargs
        Additional keyword arguments forwarded to :func:`subprocess.run`.

    Returns
    -------
    str
        Standard output from the command (stripped of trailing whitespace).
    """
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )
    return result.stdout.strip()


def _find_matching_build(
    builds_yaml: Path,
    fingerprint: dict,
    log_func: Callable[[str], None] = print,
) -> dict | None:
    """
    Look in builds.yaml for an entry whose configuration matches `fingerprint`.

    The comparison is done on a filtered view of each entry where the
    following keys are ignored:

      - token
      - timestamp_utc
      - exe   (we'll reuse whatever exe that entry points to)
      - clean
      - system

    Parameters
    ----------
    builds_yaml : Path
        Path to the builds.yaml file.
    fingerprint : dict
        Configuration fingerprint for the current build.
    log_func : callable, optional
        Logging function for informational messages.

    Returns
    -------
    dict or None
        The matching build entry dictionary if found and its recorded
        executable exists on disk; otherwise ``None``.
    """
    if not builds_yaml.exists():
        return None

    with builds_yaml.open() as f:
        data = yaml.safe_load(f) or []

    if not isinstance(data, list):
        data = [data]

    ignore_keys = {"token", "timestamp_utc", "exe", "clean", "system"}

    def _filtered(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in ignore_keys}

    filtered_fingerprint = _filtered(fingerprint)

    log_func(f"Found {len(data)} existing build(s) in {builds_yaml}.")

    for entry in data:
        if not isinstance(entry, dict):
            continue

        entry_cfg = _filtered(entry)

        if entry_cfg == filtered_fingerprint:
            token = entry.get("token")
            exe_raw = entry.get("exe")
            log_func(f"Matching build found: token={token}")

            if not exe_raw:
                log_func("  -> exe field missing or empty in builds.yaml entry; skipping.")
                continue

            exe_path = Path(str(exe_raw)).expanduser()
            if exe_path.exists():
                log_func(f"  -> using existing executable at: {exe_path}")
                return entry
            else:
                log_func(
                    f"  -> recorded exe does not exist on filesystem: {exe_path}; skipping."
                )

    return None


# =========================================================
# Core build function
# =========================================================


def build(
    grid_name: str,
    parameters: Dict[str, Dict[str, Any]],
    clean: bool = False,
    model_name: str = "roms-marbl",
) -> Path | None:
    """
    Build ROMS, MARBL, NHMG, and Tools-Roms for a given grid and configuration.

    The build uses paths from :mod:`config` and repository information from
    ``models.yaml``. Each build is tagged with a unique token, which is used
    to:

      - tag log filenames,
      - rename the ROMS executable,
      - record metadata in a ``builds.yaml`` file.

    If a previous build with the same configuration is found and `clean=False`,
    the existing executable is reused and the build steps are skipped.

    Parameters
    ----------
    grid_name : str
        Name of the ROMS grid. Used for paths and naming the opt/build dirs.
    parameters : dict[str, dict[str, Any]]
        Mapping from template filenames to Jinja2 context dictionaries for
        rendering the ROMS application configuration.
    clean : bool, optional
        If True, force a clean build even if a matching configuration already
        exists. This will attempt to remove any existing executable for the
        matching build before rebuilding.

    Returns
    -------
    Path or None
        Path to the build's ROMS executable (with token in the name) if the
        build or reuse succeeds; otherwise ``None`` in error cases.
    """
    # -----------------------------------------------------
    # Unique build token and logging setup
    # -----------------------------------------------------
    build_token = (
        datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    # -----------------------------------------------------
    # Load model spec and derive directories
    # -----------------------------------------------------

    model_code = _load_models_yaml(config.paths.models_yaml, model_name)

    opt_base_dir = config.paths.model_configs / model_code.opt_base_dir

    # opt_dir: rendered configuration
    opt_dir = config.paths.model_configs / "opt" / f"opt_{model_code.name}-{grid_name}"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # build_dir_tmp: clean build tree
    build_dir_final = config.paths.model_configs / "bld" / f"bld_{model_code.name}-{grid_name}"
    build_dir_tmp = config.paths.model_configs / "bld" / f"tmp_bld_{model_code.name}-{grid_name}"
    build_dir_tmp.mkdir(parents=True, exist_ok=True)
    if build_dir_tmp.exists() and clean:
        # Remove any previous build contents to guarantee a clean build tree
        shutil.rmtree(build_dir_tmp)

    roms_conda_env = model_code.conda_env
    repos = model_code.repos
    if "roms" not in repos or "marbl" not in repos:
        raise ValueError("models.yml must define at least 'roms' and 'marbl' repos.")

    logs_dir = build_dir_tmp / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    build_all_log = logs_dir / f"build.{model_code.name}.{build_token}.log"

    def log(msg: str = "") -> None:
        text = str(msg)
        print(text)
        build_all_log.parent.mkdir(parents=True, exist_ok=True)
        with build_all_log.open("a") as f:
            f.write(text + "\n")

    log(f"Build token: {build_token}")

    # -----------------------------------------------------
    # Paths from config / sanity checks
    # -----------------------------------------------------
    input_data_path = config.paths.input_data / grid_name
    if not input_data_path.is_dir():
        raise FileNotFoundError(
            f"Expected input data directory for grid '{grid_name}' at:\n"
            f"  {input_data_path}\n"
            "but it does not exist. Did you run the `model_config.gen_inputs` step?"
        )

    codes_root = config.paths.code_root
    roms_root = codes_root / repos["roms"].default_dirname
    marbl_root = codes_root / repos["marbl"].default_dirname

    log(f"Building {model_code.name} for grid: {grid_name}")
    log(f"{model_code.name} opt_base_dir : {opt_base_dir}")
    log(f"ROMS opt_dir      : {opt_dir}")
    log(f"ROMS build_dir    : {build_dir_final}")
    log(f"Input data path   : {input_data_path}")
    log(f"ROMS_ROOT         : {roms_root}")
    log(f"MARBL_ROOT        : {marbl_root}")
    log(f"Conda env         : {roms_conda_env}")
    log(f"Logs              : {logs_dir}")

    # -----------------------------------------------------
    # Check conda and define conda-run helper
    # -----------------------------------------------------
    _check_command_exists("conda")

    def _conda_run(cmd: list[str]) -> list[str]:
        """Prefix a command so that it runs inside the target conda env."""
        return ["conda", "run", "-n", roms_conda_env] + cmd

    # Create env if needed
    env_list = _run(["conda", "env", "list"])
    if roms_conda_env not in env_list:
        log(f"Creating conda env '{roms_conda_env}' from ROMS environment file...")
        env_yml = roms_root / "environments" / "conda_environment.yml"
        if not env_yml.exists():
            raise FileNotFoundError(f"Conda environment file not found: {env_yml}")
        _run(
            [
                "conda",
                "env",
                "create",
                "-f",
                str(env_yml),
                "--name",
                roms_conda_env,
            ]
        )
    else:
        log(f"Conda env '{roms_conda_env}' already exists.")

    # -----------------------------------------------------
    # Toolchain checks (inside env)
    # -----------------------------------------------------
    try:
        _run(_conda_run(["which", "gfortran"]))
        _run(_conda_run(["which", "mpifort"]))
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"❌ gfortran or mpifort not found in env '{roms_conda_env}'. "
            "Check your conda environment."
        )

    compiler_kind = "gnu"
    try:
        mpifort_version = _run(_conda_run(["mpifort", "--version"]))
        if any(token in mpifort_version.lower() for token in ["ifx", "ifort", "intel"]):
            compiler_kind = "intel"
    except Exception:
        pass

    log(f"Using compiler kind: {compiler_kind}")

    # -----------------------------------------------------
    # Build fingerprint & cache lookup
    # -----------------------------------------------------
    builds_yaml = config.paths.builds_yaml

    fingerprint = {
        "clean": bool(clean),
        "system": config.system,
        "compiler_kind": compiler_kind,
        "parameters": parameters,
        "grid_name": grid_name,
        "input_data_path": str(input_data_path),
        "logs_dir": str(logs_dir),
        "build_dir": str(build_dir_final),
        "marbl_root": str(marbl_root),
        "model_name": model_code.name,
        "opt_base_dir": str(opt_base_dir),
        "opt_dir": str(opt_dir),
        "roms_conda_env": roms_conda_env,
        "roms_root": str(roms_root),
        "repos": {
            name: {
                "url": spec.url,
                "default_dirname": spec.default_dirname,
                "checkout": spec.checkout,
            }
            for name, spec in repos.items()
        },
    }

    existing = _find_matching_build(builds_yaml, fingerprint, log_func=log)
    if existing is not None:
        exe_path = Path(existing.get("exe"))
        if not clean:
            log(
                "Found existing build matching current configuration; reusing executable."
            )
            log(f"  token : {existing.get('token')}")
            log(f"  exe   : {exe_path}")
            log("done.")
            return exe_path
        else:
            log(f"Clean build requested; attempting to remove existing executable: {exe_path}")
            try:
                if exe_path.exists() and exe_path.is_file():
                    try:
                        exe_path.chmod(0o755)
                    except OSError as e:
                        log(f"  ⚠️ chmod failed on exe before unlink: {e}")
                    exe_path.unlink()
                    log("  -> removed existing executable.")
                else:
                    log("  -> exe path missing or not a regular file; nothing to remove.")
            except OSError as e:
                log(f"⚠️ Failed to remove existing executable {exe_path}: {e}")
                log("Proceeding with clean rebuild; old exe may remain on disk.")


    # -----------------------------------------------------
    # Clone / update repos
    # -----------------------------------------------------
    if not (roms_root / ".git").is_dir():
        log(f"Cloning ROMS from {repos['roms'].url} into {roms_root}")
        _run(["git", "clone", repos["roms"].url, str(roms_root)])
    else:
        log(f"ROMS repo already present at {roms_root}")

    if repos["roms"].checkout:
        log(f"Checking out ROMS {repos['roms'].checkout}")
        _run(["git", "fetch", "--tags"], cwd=roms_root)
        _run(["git", "checkout", repos["roms"].checkout], cwd=roms_root)

    if not (marbl_root / ".git").is_dir():
        log(f"Cloning MARBL from {repos['marbl'].url} into {marbl_root}")
        _run(["git", "clone", repos["marbl"].url, str(marbl_root)])
    else:
        log(f"MARBL repo already present at {marbl_root}")

    if repos["marbl"].checkout:
        log(f"Checking out MARBL {repos['marbl'].checkout}")
        _run(["git", "fetch", "--tags"], cwd=marbl_root)
        _run(["git", "checkout", repos["marbl"].checkout], cwd=marbl_root)

    # -----------------------------------------------------
    # Sanity checks for directory trees
    # -----------------------------------------------------
    if not (roms_root / "src").is_dir():
        raise RuntimeError(f"ROMS_ROOT does not look correct: {roms_root}")
    if not (marbl_root / "src").is_dir():
        raise RuntimeError(f"MARBL_ROOT/src not found at {marbl_root}")

    # -----------------------------------------------------
    # Environment vars for builds
    # -----------------------------------------------------
    try:
        conda_prefix = _run(
            [
                "conda",
                "run",
                "-n",
                roms_conda_env,
                "python",
                "-c",
                "import os; print(os.environ['CONDA_PREFIX'])",
            ]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to determine CONDA_PREFIX for env '{roms_conda_env}'. "
            "Is the environment created correctly?"
        ) from exc

    env = os.environ.copy()
    env["ROMS_ROOT"] = str(roms_root)
    env["MARBL_ROOT"] = str(marbl_root)
    env["GRID_NAME"] = grid_name
    env["BUILD_DIR"] = str(build_dir_tmp)

    env["MPIHOME"] = conda_prefix
    env["NETCDFHOME"] = conda_prefix
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + f":{conda_prefix}/lib"
    tools_path = str(roms_root / "Tools-Roms")
    env["PATH"] = tools_path + os.pathsep + env.get("PATH", "")

    # -----------------------------------------------------
    # Optional clean helper
    # -----------------------------------------------------
    def _maybe_clean(label: str, path: Path) -> None:
        if clean:
            log(f"[Clean] {label} ...")
            try:
                subprocess.run(
                    _conda_run(["make", "-C", str(path), "clean"]),
                    check=False,
                    env=env,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ⚠️ clean failed for {label}: {e}")

    # -----------------------------------------------------
    # Builds (all via conda run)
    # -----------------------------------------------------
    log(_run(_conda_run(["conda", "list"])))

    log_marbl = logs_dir / f"build.MARBL.{build_token}.log"
    log_nhmg = logs_dir / f"build.NHMG.{build_token}.log"
    log_tools = logs_dir / f"build.Tools-Roms.{build_token}.log"
    log_roms = logs_dir / f"build.ROMS.{build_token}.log"

    # MARBL
    _maybe_clean("MARBL/src", marbl_root / "src")
    _run_logged(
        f"Build MARBL (compiler: {compiler_kind})",
        log_marbl,
        _conda_run(
            ["make", "-C", str(marbl_root / "src"), compiler_kind, "USEMPI=TRUE"]
        ),
        env=env,
        log_func=log,
    )

    # NHMG (optional nonhydrostatic lib)
    _maybe_clean("NHMG/src", roms_root / "NHMG" / "src")
    _run_logged(
        "Build NHMG/src",
        log_nhmg,
        _conda_run(["make", "-C", str(roms_root / "NHMG" / "src")]),
        env=env,
        log_func=log,
    )

    # Tools-Roms
    _maybe_clean("Tools-Roms", roms_root / "Tools-Roms")
    _run_logged(
        "Build Tools-Roms",
        log_tools,
        _conda_run(["make", "-C", str(roms_root / "Tools-Roms")]),
        env=env,
        log_func=log,
    )

    # ROMS Application/Case
    _render_opt_base_dir_to_opt_dir(
        grid_name=grid_name,
        parameters=parameters,
        opt_base_dir=opt_base_dir,
        opt_dir=opt_dir,
        overwrite=True,
        log_func=log,
    )
    _maybe_clean(f"ROMS ({build_dir_tmp})", build_dir_tmp)
    _run_logged(
        f"Build ROMS ({build_dir_tmp})",
        log_roms,
        _conda_run(["make", "-C", str(opt_dir)]),
        env=env,
        log_func=log,
    )

    # Remove existing final directory if present
    if build_dir_final.exists():
        shutil.rmtree(build_dir_final)
    build_dir_tmp.rename(build_dir_final)

    # -----------------------------------------------------
    # Rename ROMS executable with token
    # -----------------------------------------------------
    exe_path = build_dir_final / "roms"
    exe_token_path = (
        config.paths.here
        / "model-configs"
        / "exe"
        / f"{model_code.name}-{grid_name}-{build_token}"
    )
    exe_token_path.parent.mkdir(parents=True, exist_ok=True)

    if exe_path.exists():
        exe_path.rename(exe_token_path)
        log(f"{model_code.name} executable -> {exe_token_path}")
    else:
        log(f"⚠️ {model_code.name} executable not found at {exe_path}; not renamed.")


    # -----------------------------------------------------
    # Record build metadata in builds.yaml
    # -----------------------------------------------------
    if builds_yaml.exists():
        with builds_yaml.open() as f:
            builds_data = yaml.safe_load(f) or []
    else:
        builds_data = []

    if not isinstance(builds_data, list):
        builds_data = [builds_data]

    build_entry = {
        "token": build_token,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        **fingerprint,
        "exe": str(exe_token_path if exe_token_path.exists() else exe_path),
    }

    builds_data.append(build_entry)
    with builds_yaml.open("w") as f:
        yaml.safe_dump(builds_data, f)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    log("")
    log("✅ All builds completed.")
    log(f"• Build token:      {build_token}")
    log(f"• ROMS root:        {roms_root}")
    log(f"• MARBL root:       {marbl_root}")
    log(f"• App root:         {opt_base_dir}")
    log(f"• Logs:             {logs_dir}")
    log(
        f"• ROMS exe:         {exe_token_path if exe_token_path.exists() else exe_path}"
    )
    log(f"• builds.yaml:      {builds_yaml}")
    log("")

    return exe_token_path if exe_token_path.exists() else None

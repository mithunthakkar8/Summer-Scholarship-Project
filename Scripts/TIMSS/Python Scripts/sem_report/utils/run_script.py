import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_r_sem_script(
    r_script_path: str,
    r_executable: str = "Rscript",
    timeout: int | None = None,
):
    """
    Run an R SEM script and propagate FULL R errors back to Python.

    - Captures stdout + stderr
    - Raises RuntimeError with exact R failure reason
    """

    r_script_path = Path(r_script_path)

    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    cmd = [r_executable, "--vanilla", str(r_script_path)]

    logger.info("Running R SEM script")
    logger.info("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("R SEM script timed out")

    # Log everything (VERY important)
    if stdout.strip():
        logger.info("R STDOUT:\n%s", stdout)

    if stderr.strip():
        logger.error("R STDERR:\n%s", stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"""
R script failed with exit code {proc.returncode}

================= R STDERR =================
{stderr}

================= R STDOUT =================
{stdout}
"""
        )

    logger.info("R SEM script completed successfully")

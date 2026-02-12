import subprocess
import logging

logger = logging.getLogger(__name__)

def run_r_sem_script(r_script_path: str, r_executable: str = "Rscript"):
    """
    Runs an R SEM script synchronously.
    Aborts Python execution if R fails.
    """

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Running R SEM pipeline")
    logger.info(f"Script: {r_script_path}")

    try:
        result = subprocess.run(
            [r_executable, r_script_path],
            check=True,              # 🚨 fail-fast
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        logger.info("R SEM script completed successfully")
        logger.debug("R stdout:\n" + result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error("R SEM script FAILED")
        logger.error("R stdout:\n" + e.stdout)
        logger.error("R stderr:\n" + e.stderr)
        raise RuntimeError("Aborting pipeline due to R SEM failure") from e

from packaging.version import Version
from packaging import version
import subprocess
def get_driver_version():
    """
    Returns the driver version.
    """
    # Enable console printing for `hl-smi` check
    output = subprocess.run(
        "hl-smi", shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={"ENABLE_CONSOLE": "true"}
    )
    if output.returncode == 0 and output.stdout:
        return version.parse(output.stdout.split("\n")[2].replace(" ", "").split(":")[1][:-1].split("-")[0])
    return None

MIN_TGI_GAUDI_SYNAPSE_VERSION = Version("1.19.0")


def is_driver_compatible():
    driver_version = get_driver_version()
    if driver_version is not None:
        if driver_version < MIN_TGI_GAUDI_SYNAPSE_VERSION:
            return False
    return True

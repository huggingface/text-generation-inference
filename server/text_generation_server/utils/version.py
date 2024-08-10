from optimum.habana.utils import get_driver_version
from packaging.version import Version

MIN_TGI_GAUDI_SYNAPSE_VERSION=Version("1.16.0")


def is_driver_compatible():
    driver_version = get_driver_version()
    if driver_version is not None:
        if driver_version < MIN_TGI_GAUDI_SYNAPSE_VERSION:
            return False
    return True
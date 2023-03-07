import pytest

from text_generation.types import Parameters
from text_generation.errors import ValidationError


def test_parameters_validation():
    # Test repetition_penalty
    Parameters(repetition_penalty=1)
    with pytest.raises(ValidationError):
        Parameters(repetition_penalty=0)
    with pytest.raises(ValidationError):
        Parameters(repetition_penalty=-1)

    # Test seed
    Parameters(seed=1)
    with pytest.raises(ValidationError):
        Parameters(seed=-1)

    # Test temperature
    Parameters(temperature=1)
    with pytest.raises(ValidationError):
        Parameters(temperature=0)
    with pytest.raises(ValidationError):
        Parameters(temperature=-1)

    # Test top_k
    Parameters(top_k=1)
    with pytest.raises(ValidationError):
        Parameters(top_k=0)
    with pytest.raises(ValidationError):
        Parameters(top_k=-1)

    # Test top_p
    Parameters(top_p=1)
    with pytest.raises(ValidationError):
        Parameters(top_p=0)
    with pytest.raises(ValidationError):
        Parameters(top_p=-1)

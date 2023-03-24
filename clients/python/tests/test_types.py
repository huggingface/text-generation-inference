import pytest

from text_generation.types import Parameters, Request
from text_generation.errors import ValidationError


def test_parameters_validation():
    # Test best_of
    Parameters(best_of=1)
    with pytest.raises(ValidationError):
        Parameters(best_of=0)
    with pytest.raises(ValidationError):
        Parameters(best_of=-1)
    Parameters(best_of=2, do_sample=True)
    with pytest.raises(ValidationError):
        Parameters(best_of=2)
    with pytest.raises(ValidationError):
        Parameters(best_of=2, seed=1)

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
    Parameters(top_p=0.5)
    with pytest.raises(ValidationError):
        Parameters(top_p=0)
    with pytest.raises(ValidationError):
        Parameters(top_p=-1)
    with pytest.raises(ValidationError):
        Parameters(top_p=1)

    # Test truncate
    Parameters(truncate=1)
    with pytest.raises(ValidationError):
        Parameters(truncate=0)
    with pytest.raises(ValidationError):
        Parameters(truncate=-1)

    # Test typical_p
    Parameters(typical_p=0.5)
    with pytest.raises(ValidationError):
        Parameters(typical_p=0)
    with pytest.raises(ValidationError):
        Parameters(typical_p=-1)
    with pytest.raises(ValidationError):
        Parameters(typical_p=1)


def test_request_validation():
    Request(inputs="test")

    with pytest.raises(ValidationError):
        Request(inputs="")

    Request(inputs="test", stream=True)
    Request(inputs="test", parameters=Parameters(best_of=2, do_sample=True))

    with pytest.raises(ValidationError):
        Request(
            inputs="test", parameters=Parameters(best_of=2, do_sample=True), stream=True
        )

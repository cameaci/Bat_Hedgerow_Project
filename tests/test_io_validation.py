from pathlib import Path

import pytest

from hedge_features.exceptions import InputValidationError
from hedge_features.io import read_input_geodata


def test_rejects_single_shx(tmp_path: Path):
    shx = tmp_path / "hedges.shx"
    shx.write_bytes(b"dummy")
    with pytest.raises(InputValidationError) as exc:
        read_input_geodata(shx)
    assert ".shx index file alone" in str(exc.value)


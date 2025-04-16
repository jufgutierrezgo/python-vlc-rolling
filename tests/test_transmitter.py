import numpy as np
import pytest

from vlc_rolling.constants import Constants as Kt
from vlc_rolling.transmitter import Transmitter  # adjust import path
from vlc_rolling.indoorenv import Indoorenv      # adjust import path


@pytest.fixture
def valid_room():
    color = np.ones(Kt.NO_WAVELENGTHS)
    wall = ('diffuse', color)
    return Indoorenv(
        name="Room",
        size=[5, 5, 3],
        resolution=0.2,
        ceiling=wall,
        west=wall,
        east=wall,
        north=wall,
        south=wall,
        floor=wall
    )


@pytest.fixture
def valid_params(valid_room):
    return {
        "name": "TX1",
        "room": valid_room,
        "position": [1.0, 2.0, 3.0],
        "normal": [0.0, 0.0, -1.0],
        "wavelengths": [460.0, 530.0, 630.0],  # assuming Kt.NO_LEDS = 3
        "fwhm": [20.0, 20.0, 20.0],
        "mlambert": 1.0,
        "modulation": "ieee16",
        "frequency": 1000,
        "no_symbols": 100,
        "luminous_flux": 10.0
    }


def test_valid_transmitter(valid_params):
    tx = Transmitter(**valid_params)
    assert tx._name == "TX1"
    assert tx._order_csk == 16
    assert tx._luminous_flux == 10.0


def test_invalid_room(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["room"] = "not_a_room"
    with pytest.raises(ValueError, match="Indoor environment attribute must be"):
        Transmitter(**invalid_params)


def test_invalid_position(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["position"] = [1.0, 2.0]  # only 2D
    with pytest.raises(ValueError, match="Position must be"):
        Transmitter(**invalid_params)


def test_invalid_normal(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["normal"] = [1.0, 0.0]  # too short
    with pytest.raises(ValueError, match="Normal must be"):
        Transmitter(**invalid_params)


def test_invalid_lambert_scalar(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["mlambert"] = np.array([1.0, 2.0])  # not scalar
    with pytest.raises(ValueError, match="Lambert number must be scalar float."):
        Transmitter(**invalid_params)


def test_invalid_lambert_value(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["mlambert"] = 0.0
    with pytest.raises(ValueError, match="Lambert number must be greater than zero."):
        Transmitter(**invalid_params)


def test_invalid_wavelength_size(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["wavelengths"] = np.array([450.0, 530.0])  # wrong size
    with pytest.raises(ValueError, match="Dimension of wavelengths array must be"):
        Transmitter(**invalid_params)


def test_invalid_wavelength_range(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["wavelengths"] = np.array([360.0, 530.0, 630.0])  # below 380
    with pytest.raises(ValueError, match="Wavelengths must be between"):
        Transmitter(**invalid_params)


def test_invalid_fwhm_size(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["fwhm"] = np.array([20.0, 30.0])  # wrong size
    with pytest.raises(ValueError, match="Dimension of FWHM array must be"):
        Transmitter(**invalid_params)


def test_invalid_fwhm_values(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["fwhm"] = np.array([20.0, -1.0, 30.0])  # negative value
    with pytest.raises(ValueError, match="FWDM must be non-negative."):
        Transmitter(**invalid_params)


def test_invalid_modulation(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["modulation"] = "invalid-mod"
    with pytest.raises(ValueError, match="Modulation is not valid."):
        Transmitter(**invalid_params)


def test_invalid_frequency(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["frequency"] = -10.0
    with pytest.raises(ValueError, match="Frequency must be non-negative."):
        Transmitter(**invalid_params)


def test_invalid_no_symbols_type(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["no_symbols"] = "1000"  # str, not int
    with pytest.raises(ValueError, match="No. of symbols must be a positive integer."):
        Transmitter(**invalid_params)


def test_invalid_no_symbols_value(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["no_symbols"] = 0
    with pytest.raises(ValueError, match="No. of symbols must be greater than zero."):
        Transmitter(**invalid_params)


def test_invalid_luminous_flux(valid_params):
    invalid_params = valid_params.copy()
    invalid_params["luminous_flux"] = -1.0
    with pytest.raises(ValueError, match="The luminous flux must be non-negative."):
        Transmitter(**invalid_params)

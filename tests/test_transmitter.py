import numpy as np
import pytest

from vlc_rolling.transmitter import Transmitter

from vlc_rolling.constants import Constants as Kt

from vlc_rolling.indoorenv import Indoorenv 

from vlc_rolling.sightpy import *


# Mock Indoorenv class if not available
def DummyIndoorEnv():
    
    green_wall = rgb(0.0, 1.0, 0.0)
    red_wall = rgb(0.0, 1.0, 0.0)
    white_wall = rgb(0.8, 0.8, 0.8)
    floor_wall = rgb(0.1, 0.1, 0.1)

    # Create indoor environment and 3d scene

    room = Indoorenv(
        name="Matisse-CornellBox",
        size=[5, 5, 3],
        no_reflections=10,
        resolution=1/10,
        ceiling=('diffuse', white_wall),
        west=('diffuse', white_wall),
        north=('diffuse', green_wall),
        east=('diffuse', white_wall),
        south=('diffuse', red_wall),
        floor=('diffuse', floor_wall)
            )
    
    return room

def test_transmitter_default_init():
    room = DummyIndoorEnv()
    tx = Transmitter(
        room=room,
        name="TestLED",
        led_type="gaussian",
        position=[0.5, 0.5, 2.0],
        normal=[0, 0, -1],
        wavelengths=[450, 550, 650],
        fwhm=[20, 20, 20],
        mlambert=1,
        constellation="ieee16",
        luminous_flux=100
    )

    assert tx.name == "TestLED"
    assert np.allclose(tx.position, np.array([0.5, 0.5, 2.0], dtype=np.float32))
    assert np.allclose(tx.normal, np.array([[0, 0, -1]], dtype=np.float32))
    assert tx.mlambert == 1
    assert np.allclose(tx.wavelengths, np.array([450, 550, 650], dtype=np.float32))
    assert np.allclose(tx.fwhm, np.array([20, 20, 20], dtype=np.float32))
    assert tx.constellation.shape == (3, 16)
    assert tx.luminous_flux == 100
    assert isinstance(str(tx), str)

def test_invalid_room_type():
    with pytest.raises(ValueError):
        Transmitter(room="not_a_room")

def test_invalid_wavelength_range():
    room = DummyIndoorEnv()
    with pytest.raises(ValueError):
        Transmitter(
            room=room,
            wavelengths=[300, 550, 600]  # 300 is out of visible range
        )

def test_custom_led_reference_not_supported():
    room = DummyIndoorEnv()
    with pytest.raises(ValueError):
        Transmitter(
            room=room,
            led_type="custom",
            reference="NotSupported"
        )

def test_invalid_constellation_format():
    room = DummyIndoorEnv()
    with pytest.raises(ValueError):
        Transmitter(
            room=room,
            constellation=np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 rows, should be 3
        )

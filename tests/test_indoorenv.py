# test_indoorenv.py
import numpy as np
import pytest

from vlc_rolling.constants import Constants as Kt
from vlc_rolling.indoorenv import Indoorenv  # Assuming your module is named indoorenv.py


@pytest.fixture
def valid_color():
    return np.ones(Kt.NO_WAVELENGTHS)

@pytest.fixture
def valid_wall(valid_color):
    return ('diffuse', valid_color)

@pytest.fixture
def default_size():
    return [5.0, 4.0, 3.0]

@pytest.fixture
def default_resolution():
    return 0.5

def test_valid_initialization(valid_wall, default_size, default_resolution):
    env = Indoorenv(
        name="TestRoom",
        size=default_size,
        resolution=default_resolution,
        ceiling=valid_wall,
        west=valid_wall,
        north=valid_wall,
        east=valid_wall,
        south=valid_wall,
        floor=valid_wall
    )
    assert env.name == "TestRoom"
    assert np.array_equal(env.size, np.array(default_size))
    assert env.resolution == default_resolution
    assert env.ceiling[0] == 'diffuse'
    assert np.array_equal(env.ceiling[1], valid_wall[1])

def test_invalid_size(valid_wall, default_resolution):
    with pytest.raises(ValueError, match="Size of the indoor environment must be"):
        Indoorenv(
            name="BadRoom",
            size=[5.0, 4.0],  # Invalid
            resolution=default_resolution,
            ceiling=valid_wall,
            west=valid_wall,
            north=valid_wall,
            east=valid_wall,
            south=valid_wall,
            floor=valid_wall
        )

def test_invalid_resolution(valid_wall, default_size):
    with pytest.raises(ValueError, match="Resolution of points must be a real number"):
        Indoorenv(
            name="BadResolution",
            size=default_size,
            resolution=-1.0,  # Invalid
            ceiling=valid_wall,
            west=valid_wall,
            north=valid_wall,
            east=valid_wall,
            south=valid_wall,
            floor=valid_wall
        )

def test_invalid_wall_type(valid_color, default_size, default_resolution):
    with pytest.raises(ValueError, match="type must be one of"):
        Indoorenv(
            name="BadWall",
            size=default_size,
            resolution=default_resolution,
            ceiling=('mirror', valid_color),  # Invalid wall type
            west=('diffuse', valid_color),
            north=('diffuse', valid_color),
            east=('diffuse', valid_color),
            south=('diffuse', valid_color),
            floor=('diffuse', valid_color)
        )

def test_invalid_wall_color_size(default_size, default_resolution):
    bad_color = np.ones(Kt.NO_WAVELENGTHS + 1)  # Wrong size
    with pytest.raises(ValueError, match="Color array for ceiling must have size"):
        Indoorenv(
            name="BadWallColor",
            size=default_size,
            resolution=default_resolution,
            ceiling=('diffuse', bad_color),
            west=('diffuse', bad_color),
            north=('diffuse', bad_color),
            east=('diffuse', bad_color),
            south=('diffuse', bad_color),
            floor=('diffuse', bad_color)
        )

def test_setters(valid_wall, valid_color, default_size, default_resolution):
    env = Indoorenv(
        name="SetterRoom",
        size=default_size,
        resolution=default_resolution,
        ceiling=valid_wall,
        west=valid_wall,
        north=valid_wall,
        east=valid_wall,
        south=valid_wall,
        floor=valid_wall
    )

    env.name = "UpdatedRoom"
    assert env.name == "UpdatedRoom"

    new_size = [6.0, 5.0, 4.0]
    env.size = new_size
    assert np.array_equal(env.size, new_size)

    new_color = np.zeros(Kt.NO_WAVELENGTHS)
    env.ceiling = ('glossy', new_color)
    assert env.ceiling[0] == 'glossy'
    assert np.array_equal(env.ceiling[1], new_color)

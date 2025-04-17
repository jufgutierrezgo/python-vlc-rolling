# import numpy library
import numpy as np

from vlc_rolling.constants import Constants as Kt

from vlc_rolling.indoorenv import Indoorenv

from vlc_rolling.imagesensor import Imagesensor

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

def test_valid_initialization():
 
    cam = Imagesensor(
        name = 'TestRoom',
        focal_length = 0.1,
        pixel_size = 1,
        image_height = 200,
        image_width = 200,
        camera_center = vec3(278, 278, 800),
        camera_look_at =  vec3(278, 278, 0),
        room = DummyIndoorEnv(),
        sensor='SonyStarvisBSI'
            )

    # Assertions for each property
    assert cam.name == 'TestRoom'
    assert cam.focal_length == 0.1
    assert cam.pixel_size == 1.0
    assert cam.image_height == 200
    assert cam.image_width == 200
    # assert np.allclose(cam.camera_center, vec3(278, 278, 800))
    # assert np.allclose(cam.camera_look_at, vec3(278, 278, 0))
    # assert cam.room == DummyIndoorEnv
    # assert isinstance(cam.quantum_efficiency, np.ndarray)
    # assert cam.quantum_efficiency.ndim == 2  # assuming it is 2D: shape (wavelengths, channels)
    
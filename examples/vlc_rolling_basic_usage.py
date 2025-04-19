# Import Transmitter
from vlc_rolling.transmitter import Transmitter
# Import Photodetector
# from vlc_rm.photodetector import Photodetector
# Import Indoor Environment
from vlc_rolling.indoorenv import Indoorenv

from vlc_rolling.imagesensor import Imagesensor
# Import REcursiveModel
# from vlc_rm.recursivemodel import Recursivemodel
# Import Symbol Constants
from vlc_rolling.constants import Constants as Kt

from vlc_rolling.sightpy import *

# Import numpy
import numpy as np

#Import luxpy   
import luxpy as lx

# Import Matplotlob
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['figure.facecolor']='white'



green_wall = rgb(0.0, 1.0, 0.0)
red_wall = rgb(1.0, 0.0, 0.0)
white_wall = rgb(0.8, 0.8, 0.8)
floor_wall = rgb(0.1, 0.1, 0.1)

# room dimensions in milimeters
WIDTH = 5e2
LENGTH = 5e2
HEIGHT = 5e2

# camera parameters
# CAMERA_CENTER=[]

# Create indoor environment and 3d scene

room = Indoorenv(
    name="Matisse-CornellBox",
    size=[LENGTH, WIDTH, HEIGHT],
    no_reflections=10,
    resolution=1/10,
    ceiling=('diffuse', white_wall),
    west=('diffuse', red_wall),
    north=('diffuse', white_wall),
    east=('diffuse', green_wall),
    south=('diffuse', white_wall),
    floor=('diffuse', floor_wall)
        )
room.create_environment()
print(room)

# Create a transmitter-type object
transmitter = Transmitter(
    room=room,
    name="Led1",
    led_type='gaussian',
    reference='RGB-Phosphor',
    position=[LENGTH/2, HEIGHT*0.99, WIDTH/2],
    normal=[0, 0, -1],
    mlambert=1.3,
    wavelengths=[620, 530, 475],
    fwhm=[20, 30, 20],
    constellation='ieee16',
    luminous_flux=5000
            )
print(transmitter)

img_sensor = Imagesensor(
    name="Camera",
    focal_length = 1e0,
    pixel_size = 1,
    image_height = 2*25,
    image_width = 3*25,
    camera_center = vec3(WIDTH/2, 0.49*LENGTH, HEIGHT + HEIGHT/20),
    camera_look_at = vec3(WIDTH/2, LENGTH/2, HEIGHT),
    room = room,
    sensor='SonyStarvisBSI'
    )
print(img_sensor)

img_sensor.take_picture(plot='true')
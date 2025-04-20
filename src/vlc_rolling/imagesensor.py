
from vlc_rolling.constants import Constants as Kt

#import transmitter module
from vlc_rolling.transmitter import Transmitter

from vlc_rolling.indoorenv import Indoorenv

from vlc_rolling.sightpy import *

# numeric numpy library
import numpy as np

# Library to plot 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Scipy import
import scipy.signal as signal
from scipy.stats import norm
# Skiimage import
from skimage import data

from typing import Optional

import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

class Imagesensor:
    """
    This class defines the camera properties
    """

    _DECIMALS = 2  # how many decimal places to use in print

    def __init__(
        self,
        name: str,
        focal_length: float,
        pixel_size: float,
        image_height: float,
        image_width: float,
        camera_center: np.ndarray,
        camera_look_at: np.ndarray,
        room: Indoorenv,
        transmitter: Transmitter,
        sensor: str
        ) -> None:
        
        self._name = name

        # Focal length check
        self._focal_length = np.float32(focal_length)
        if self._focal_length <= 0:
            raise ValueError("Focal length must be positive.")

        # Pixel size check
        self._pixel_size = np.float32(pixel_size)
        if self._pixel_size <= 0:
            raise ValueError("Pixel size must be positive.")

        # Image height
        self._image_height = image_height
        if self._image_height <= 0:
            raise ValueError("Image height must be positive.")

        # Image width
        self._image_width = image_width
        if self._image_width <= 0:
            raise ValueError("Image width must be positive.")

        # Camera position and orientation
        # TODO: Create validation for these two parameters
        self._camera_center = camera_center
        self._camera_look_at = camera_look_at

        # Indoor type check
        if not isinstance(room, Indoorenv):
            raise ValueError("Indoorenv attribute must be of type Indoorenv.")
        self._room = room

        # Transmitter type check
        self._transmitter = transmitter
        if not type(transmitter) is Transmitter:
            raise ValueError(
                "Transmiyyer attribute must be an object type Transmitter.")

        # Sensor QE loading
        if sensor == 'SonyStarvisBSI':
            self._quantum_efficiency = np.loadtxt(Kt.SENSOR_PATH + "SonyStarvisBSI.txt")
        else:
            raise ValueError(f"Sensor '{sensor}' is not supported.")        
        # -------------- I N I T I A L - C O D E -------------------         
           
        self._rgb_responsivity = self._compute_responsivity(
                self._quantum_efficiency
                ) 

        # Init function for image sensor
        self._add_camera_to_scene()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value


    @property
    def focal_length(self) -> float:
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        value = np.float32(value)
        if value <= 0:
            raise ValueError("Focal length must be positive.")
        self._focal_length = value


    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        value = np.float32(value)
        if value <= 0:
            raise ValueError("Pixel size must be positive.")
        self._pixel_size = value


    @property
    def image_height(self) -> float:
        return self._image_height

    @image_height.setter
    def image_height(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Image height must be positive.")
        self._image_height = value


    @property
    def image_width(self) -> float:
        return self._image_width

    @image_width.setter
    def image_width(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Image width must be positive.")
        self._image_width = value


    @property
    def camera_center(self) -> np.ndarray:
        return self._camera_center

    @camera_center.setter
    def camera_center(self, value: np.ndarray) -> None:
        self._camera_center = value  # Add validation if needed


    @property
    def camera_look_at(self) -> np.ndarray:
        return self._camera_look_at

    @camera_look_at.setter
    def camera_look_at(self, value: np.ndarray) -> None:
        self._camera_look_at = value  # Add validation if needed


    @property
    def room(self) -> Indoorenv:
        return self._room

    @room.setter
    def room(self, value: Indoorenv) -> None:
        if not isinstance(value, Indoorenv):
            raise ValueError("Room must be an instance of Indoorenv.")
        self._room = value


    @property
    def quantum_efficiency(self) -> np.ndarray:
        return self._quantum_efficiency


    @property
    def rgb_responsivity(self) -> np.ndarray:
        return self._rgb_responsivity


    @property
    def idark(self) -> float:
        return self._idark

    @idark.setter
    def idark(self, idark):
        self._idark = idark
        if not (isinstance(self._idark, (float))) or self._idark <= 0:
            raise ValueError(
                "Dark current curve must be float and non-negative.")

    def _add_camera_to_scene(self):
        """
        This funtion is used for add/define the camera in the 3d scene 
        for the ray-tracer.
        """
        self._room._scene_rt.add_Camera(
            screen_width = self._image_width,
            screen_height = self._image_height,
            look_from = self._camera_center, 
            look_at = self._camera_look_at, 
            focal_distance= self._focal_length, 
            field_of_view= 80
            )

    def take_picture(self, plot='false') -> np.ndarray:
        """This function computes the projected image on the image sensor and
        computes the intensity distribution."""    
    
        self._npimage_rgblinear_gain = self._room.render_environment(plot=plot)

        self._crosstalk = self._compute_crosstalk(
            spd_led=self._transmitter._spd_1lm,
            color_filter_response=self._rgb_responsivity
            )
        
        self._image_bayer_mask, self._image_bayer_crosstalk = self._create_bayern_filter(
            Hmat=self._crosstalk,
            height=self._image_height,
            width=self._image_width
            )
        
        # Calculate the image considering raytracer + crosstalk_bayern
        # this 3d numpy array can be used to multiple csk symbols to 
        # compute the electrical current per pixel 
        self._image_H_rgblinear = np.multiply(
            self._image_bayer_crosstalk, 
            self._npimage_rgblinear_gain
            )
        
        # Example log messages
        logging.info(f"image_H_rgblinear matrix shape: {self._image_H_rgblinear.shape}")
        logging.info(f"image H rgblinear matrix:\n {self._image_H_rgblinear}")

    def plot_rgblinear_image(self):

        logging.info(f"RGBLinear Image: {self._npimage_rgblinear_gain}")   

        plt.imshow(self._npimage_rgblinear_gain)
        plt.title('RGBLinear Image')        
        plt.show()
        

    def plot_crosstalk_rgblinear_image(self):

        logging.info(f"Crosstalk+RGBLinear Image: {self._image_H_rgblinear}")   

        plt.imshow(self._image_H_rgblinear)
        plt.title('Crosstalk+RGBLinear Image')        
        plt.show()
        
    def _compute_pixel_power(
            self, 
            pos_cam: np.ndarray,
            n_cam: np.ndarray, 
            pos_led: np.ndarray, 
            n_led: np.ndarray,
            surface_points: np.ndarray,
            n_surface: np.ndarray,
            area_surf: float,
            m_lambert: float,
            pixel_area: float,
            luminous_flux: float
                ) -> np.ndarray:
        
        print("Computing the irradiance in each pixel inside of polygon...")

        dist_led = np.linalg.norm(surface_points - pos_led, axis=1)
        dist_cam = np.linalg.norm(pos_cam - surface_points, axis=1)

        no_points = len(dist_led)
        delta_area = area_surf / no_points

        unit_vled = np.divide(
            surface_points - pos_led,
            dist_led.reshape((-1, 1))
            )
        
        unit_vcam = np.divide(
            pos_cam - surface_points,
            dist_cam.reshape((-1, 1))
            )

        # print("Diff:")
        # print(surface_points - pos_led)
        # print("Unit Vector LED")
        # print(unit_vled)
        # print("Distance")
        # print(dist_led)

        cos_phi_led = (unit_vled).dot(n_led)
        cos_theta_surface = (-unit_vled).dot(n_surface)
        cos_phi_surface = (unit_vcam).dot(n_surface)
        cos_theta_pixel = (-unit_vcam).dot(n_cam)

        # print("Cos-Phi LED:")
        # print(cos_phi_led)
        # print("Cos-Theta Surface:")
        # print(cos_theta_surface)
        # print("Cos-Phi Surface:")
        # print(cos_phi_surface)
        # print("Cos-Theta Pixel:")
        # print(cos_theta_pixel)

        power_pixel = (luminous_flux)*(
            (m_lambert+1)/(2*np.pi*dist_led**2) *
            (cos_phi_led**m_lambert) *
            cos_theta_surface *
            delta_area *
            cos_phi_surface *
            cos_theta_pixel *
            1/(2*np.pi*dist_cam**2) *
            pixel_area
            )
        
        # print("Power in each pixel")
        # print(power_pixel)

        return power_pixel
     
    def plot_image_intensity(self):
        """ Plot the image of the received power. """

        normalized_power_image = self._power_image / np.max(self._power_image)       
        
        # Plot power image
        plt.imshow(normalized_power_image, cmap='gray', interpolation='nearest')
        plt.title("Image of the normalized received power")        
        plt.show()
    
    def plot_quantum_efficiency(self) -> None:
        for i in range(Kt.NO_LEDS):
            plt.plot(        
                self._quantum_efficiency[:, 0],
                self._quantum_efficiency[:, i+1],
                color=['r', 'g', 'b'][i],
                linestyle='dashed'
            )
        plt.title("Spectral Quantum Efficiency")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Response")
        plt.grid()
        plt.show()    

    def _compute_responsivity(self, qe) -> np.ndarray:
        """
        This functions computes the responsivity for each color channel.
        The R array has a follow specification:
            column 0: wavelengths array
            column 1: R responsivity array
            column 2: B responsivity array
            column 3: G responsivity array
        """    

        # Define an array to store the responsivity data
        R = np.zeros_like(qe)

        # Define constants
        h = Kt.H    # Planck's constant (J.s)
        c = Kt.SPEED_OF_LIGHT        # Speed of light (m/s)
        e = Kt.QE  # Elementary charge (C)

        # Define wavelengths array according to the qe array [Agrawal]
        wavelengths = qe[:, 0] * 1e-9 # wavelength range from 400 to 700 nm
        R[:, 0] = wavelengths

        # Compute spectral responsivity
        R[:, 1] = (wavelengths * qe[:, 1] * e) / (h * c)
        R[:, 2] = (wavelengths * qe[:, 2] * e) / (h * c)
        R[:, 3] = (wavelengths * qe[:, 3] * e) / (h * c)
        

        return R

    def plot_responsivity(self) -> None:
        """ This function plots the responsivity of the color channels. """
        
        for i in range(Kt.NO_LEDS):
            plt.plot(
                self._rgb_responsivity[:, 0] * 1e9,
                self._rgb_responsivity[:, i+1],
                color=['r', 'g', 'b'][i],
                linestyle='solid'
            )
        plt.title(" Spectral RGB responsivity")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Response [A/W]")
        plt.grid()
        plt.show()

    def _compute_crosstalk(self, spd_led, color_filter_response):
        """
        This function computes an spectral factor from the SPD of LED, 
        the spectral reflectance of the surface, the spectral response
        of the color channel, and the responsivity of the substrate.
        """
        print("Computing crosstalk ... ")
        
        H = np.zeros((3, 3))

        # print(spd_led.shape)
        # print(reflectance.shape)
        # print(channel_response.shape)

        for i in range(Kt.NO_LEDS):
            for j in range(Kt.NO_DETECTORS):
                H[j, i] = np.sum(
                    spd_led[:, i] *  
                    color_filter_response[:, j+1]
                )

        print("Crosstalk matrix:\n", H)        

        return H
        
    def _create_bayern_filter(self, Hmat, height, width):
        
        # Define the Bayer filter pattern according SONY-IMX219PQ
        bayer_block = np.array([[0, 1],
                                [1, 2]])
        
        # Define the color filter array
        cfa = np.zeros((2, 2, 3))

        # Assign color filter transmission values based on the Bayer filter pattern
        cfa[bayer_block == 0, 0] = Hmat[0, 0]
        cfa[bayer_block == 0, 1] = Hmat[0, 1]
        cfa[bayer_block == 0, 2] = Hmat[0, 2]

        cfa[bayer_block == 1, 0] = Hmat[1, 0]
        cfa[bayer_block == 1, 1] = Hmat[1, 1]
        cfa[bayer_block == 1, 2] = Hmat[1, 2]

        cfa[bayer_block == 2, 0] = Hmat[2, 0]
        cfa[bayer_block == 2, 1] = Hmat[2, 1]
        cfa[bayer_block == 2, 2] = Hmat[2, 2]

        # Print the color filter array
        # print("Color Filter Array - Crosstalk")
        # print(cfa)

        # Create a Bayer filter pattern for the entire image
        num_blocks_x = width // 2
        num_blocks_y = height // 2

        # print(width, num_blocks_x)
        # print(height, num_blocks_y)

        image_bayern_mask = np.tile(bayer_block, (num_blocks_y, num_blocks_x))
        
        # create an zeros array to store crosstalk and bayer filter
        image_bayer_crosstalk = np.zeros((height, width, 3))
        
        image_bayer_crosstalk[:, :, 0] = np.tile(
            cfa[:, :, 0], (num_blocks_y, num_blocks_x)) 
        image_bayer_crosstalk[:, :, 1] = np.tile(
            cfa[:, :, 1], (num_blocks_y, num_blocks_x))
        image_bayer_crosstalk[:, :, 2] = np.tile(
            cfa[:, :, 2], (num_blocks_y, num_blocks_x))        

        return image_bayern_mask, image_bayer_crosstalk

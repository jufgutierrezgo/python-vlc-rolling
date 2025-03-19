
from constants import Constants as Kt

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

#import transmitter module
from transmitter import Transmitter as Transmitter

#import surface module
from vlc_rolling.indoorenv import Surface as Surface

# import camera module
from camera_models import *  # our package

from typing import Optional

import logging

# logging.basicConfig(format=FORMAT)

class Camera:
    """
    This class defines the camera properties
    """

    _DECIMALS = 2  # how many decimal places to use in print

    def __init__(
        self,
        name: str,
        focal_length: float,
        pixel_size: float,
        px: float,
        py: float,
        mx: float,
        my: float,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        centre: np.ndarray,
        image_height: float,
        image_width: float,        
        transmitter: Transmitter,
        surface: Surface,
        sensor: str
            ) -> None:

        self._name = name

        self._focal_length = np.float32(focal_length)        
        if self._focal_length <= 0:
            raise ValueError("The luminous flux must be non-negative.")
        
        self._pixel_size = np.float32(pixel_size)        
        if self._pixel_size <= 0:
            raise ValueError("Pixel size must be non-negative.")

        # principal point x-coordinate
        self._px = px
        if self._px <= 0:
            raise ValueError("The PX must be non-negative.")

        # principal point y-coordinate
        self._py = py
        if self._py <= 0:
            raise ValueError("The PY must be non-negative.")

        # number of pixels per unit distance in image coordinates in x direction
        self._mx = mx
        if self._mx <= 0:
            raise ValueError("The MX must be non-negative.")

        # number of pixels per unit distance in image coordinates in y direction
        self._my = my
        if self._my <= 0:
            raise ValueError("The PX must be non-negative.")

        # roll angle
        self._theta_x = theta_x

        # pitch angle
        self._theta_y = theta_y

        # yaw angle
        self._theta_z = theta_z

        # camera centre
        self._centre = np.array(centre,  dtype=np.float32)
        if not (isinstance(self._centre, np.ndarray)) or self._centre.size != 3:
            raise ValueError("Camera centre must be an 1d-numpy array [x y z] dtype= float or int.")        

        # image height
        self._image_height = image_height
        if self._image_height <= 0:
            raise ValueError("The IMAGE LENGTH must be non-negative.")

        # image width
        self._image_width = image_width
        if self._image_width <= 0:
            raise ValueError("The IMAGE WIDTH must be non-negative.")
        
        # resolution height
        self._resolution_h = round(self._image_height)
        
        
        # resolution width
        self._resolution_w = round(self._image_width)        
        

        self._surface = surface             
        if not type(surface) is Surface:
            raise ValueError(
                "Surface attribute must be an object type Surface.")

        self._transmitter = transmitter
        if not type(transmitter) is Transmitter:
            raise ValueError(
                "Transmiyyer attribute must be an object type Transmitter.")
        
        if sensor == 'SonyStarvisBSI':
            # read text file into NumPy array
            self._quantum_efficiency = np.loadtxt(
                Kt.SENSOR_PATH+"SonyStarvisBSI.txt")         
        
        # -------------- I N I T I A L - C O D E -------------------         
        self._pixel_area = (1/self._mx) * (1/self._my)                
        self._rgb_responsivity = self._compute_responsivity(
                self._quantum_efficiency
                ) 
        
    @property
    def idark(self) -> float:
        return self._idark

    @idark.setter
    def idark(self, idark):
        self._idark = idark
        if not (isinstance(self._idark, (float))) or self._idark <= 0:
            raise ValueError(
                "Dark current curve must be float and non-negative.")

    def take_picture(self) -> np.ndarray:
        """This function computes the projected image on the image sensor and
        computes the intensity distribution."""    
      
    def plot_binary_image(self, pixels, height, width):        
    
        binary_image = np.zeros((height, width))
        binary_image[pixels[1, :], pixels[0, :]] = 1

        # Plot binary matrix
        plt.imshow(binary_image, cmap='gray', interpolation='nearest')
        plt.title("Binary image of the projected area")
        # plt.scatter(pixels[1,:],pixels[0,:])
        # plt.xlim([0, self._resolution_w])
        # plt.ylim([0, self._resolution_h])
        plt.show()

        return binary_image
        
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

    def _draw3d_led(
            self, 
            origin_led = np.array([0, 0, 0]),
            ax: Optional[Axes3D] = None,
            name: str = "LED") -> Axes3D:
        
        if ax is None:
            ax = plt.gca(projection="3d")

        # Define the 8 vertices of the rectangular parallelepiped
        vertices = np.array([
            (-0.5, -0.5, 0), 
            (-0.5, 0.5, 0), 
            (0.5, 0.5, 0), 
            (0.5, -0.5, 0),     
            (-0.5, -0.5, 0.1),
            (-0.5, 0.5, 0.1),
            (0.5, 0.5, 0.1),
            (0.5, -0.5, 0.1)      
            ])/5 + origin_led

        # Define the 12 edges of the rectangular parallelepiped
        edges = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])

        # Plot the rectangular parallelepiped
        for edge in edges:
            ax.plot3D(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 'blue')        
        
        ax.text(*(origin_led+[0, 1, 0]), name)

        return ax

    def _compute_power_image(self, pixels_power, pixels_inside, height, width) -> np.ndarray:
        """ This function computes the image with the received power by each pixel. """

        power_image = np.zeros((height, width))

        for i in range(len(pixels_power)):
            power_image[pixels_inside[1, i], pixels_inside[0, i]] = pixels_power[i]

        no_blurred_image =  power_image

        return power_image, no_blurred_image
     
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

    def _compute_crosstalk(self, spd_led, reflectance, channel_response):
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
                    reflectance[:, 1] * 
                    channel_response[:, j+1]
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

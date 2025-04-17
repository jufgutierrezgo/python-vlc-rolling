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

import cv2

import sys
sys.path.insert(0, './camera-models')

# import transmitter module
from transmitter import Transmitter as Transmitter
# import surface module 
from vlc_rolling.indoorenv import Surface as Surface
# import camera module 
from camera import Camera

from typing import Optional



import logging

# logging.basicConfig(format=FORMAT)

class RollingShutter:
    """
    This class defines the rolling shutter adquisition properties
    """    
    
    def __init__(
        self,
        name: str,
        t_exposure: float,
        t_rowdelay: float,
        t_start: float,
        iso: float,
        adc_resolution: int,        
        gain_pixel: float,
        temperature: float,
        idark: float,
        transmitter: Transmitter,
        camera: Camera        
            ) -> None:

        self._name = name

        self._t_exposure = np.float32(t_exposure)        
        if self._t_exposure <= 0:
            raise ValueError("The exposure time must be a float non-negative.")
        
        self._t_rowdelay = np.float32(t_rowdelay)        
        if self._t_rowdelay <= 0:
            raise ValueError("The row delay time must be a float non-negative.")
        self._bandwidth = 1/self._t_rowdelay

        self._t_start = np.float32(t_start)        
        if self._t_rowdelay <= 0:
            raise ValueError(
                "The row adquisition start time must be non-negative."
                )

        self._iso = np.int(iso)        
        if self._iso <= 0:
            raise ValueError(
                "The ISO must be integer non-negative."
                )
        
        self._adc_resolution = adc_resolution
        self.MAX_ADC = 2 ** self._adc_resolution - 1
        self._gain_pixel = gain_pixel
        
        self._temperature = temperature

        self._idark = np.float32(idark)
        if self._idark <= 0:
            raise ValueError(
                "Dark current curve must be float and non-negative.")

        self._transmitter = transmitter
        if not type(transmitter) is Transmitter:
            raise ValueError(
                "Transmiyyer attribute must be an object type Transmitter.")
        
        self._camera = camera
        if not type(camera) is Camera:
            raise ValueError(
                "Camera attribute must be an object type Camera.")
        
        self._index_row_bins = self._compute_row_bins()

        self._image_current = self._compute_image_current(
            symbols_csk=self._transmitter._symbols_csk,
            im_bayern_crostalk=self._camera._image_bayern_crosstalk,
            im_gain=camera._power_image,
            index_bins=self._index_row_bins,
            height=self._camera._resolution_h,
            width=self._camera._resolution_w,
            t_rowdelay=self._t_rowdelay
        )
        
        self._rgb_image = self._bayerGBGR_to_RGB(
            bayer=self._image_current,
            height=self._camera._resolution_h,
            width=self._camera._resolution_w
            )

        self._noisy_image = self._add_noise_to_raw_image(
            raw_image=self._rgb_image,
            current_image=self._image_current,
            idark=self._idark,
            gain=self._gain_pixel,
            B=self._bandwidth,
            T=self._temperature
            )
        
    def _compute_row_bins(self) -> np.ndarray:
        """ This function computes the row bins respect to each of symbols. """

        # compute the time of each symbol. It is equal to 1/f.
        t_symbol = 1/self._transmitter._frequency

        # compute the symbol corresponding to the last row 
        last_symbol = int(
                ((self._t_rowdelay * self._camera._resolution_h)
                + self._t_start) / t_symbol 
            ) + 1

        no_symbol = np.arange(0, last_symbol+1)
        
        index_bins = (
            ((no_symbol*t_symbol) - self._t_start - self._t_exposure/2)
            / self._t_rowdelay        
            ).astype(int) + 1
        index_bins[-1] = self._camera._resolution_h
        index_bins[0] = 0

        print("Row bins:")
        print(index_bins)

        return index_bins

    def _compute_image_current(
            self, 
            symbols_csk, 
            im_bayern_crostalk, 
            im_gain,
            index_bins,
            height,
            width,
            t_rowdelay) -> np.ndarray:
        """ This function computes the image with the transmitter symbols. """

        image_current = np.zeros((height, width))
        image_color = np.zeros((height, width))

        for symbol in np.arange(1,np.size(index_bins)):
            image_color[
                index_bins[symbol-1]: index_bins[symbol], :
                ] = np.sum(
                    np.multiply(
                        im_bayern_crostalk[
                            index_bins[symbol-1]: index_bins[symbol], :, :],
                        symbols_csk[:, symbol-1].reshape(1, 1, 3)
                    ),
                axis=2
                )
            # print(symbols_csk[:, symbol-1].reshape(1, 1, 3))

        image_current = im_gain * image_color 
        image_bayer_norm = image_current / np.max(image_current)        

        return image_current
    
    def plot_current_image(self):
        """ Plot the image of the photocurrent by each pixel. """        
        
        # Plot power image
        plt.imshow(self._image_current, cmap='gray', interpolation='nearest')
        plt.title("Image of the normalized received power")        
        plt.show()
  
    def _bayerGBGR_to_RGB(self, bayer, height, width):
        """ Plot the image of the photocurrent by each pixel. """        
        
        # bayer = bayer / np.max(bayer)
        voltage_bayer = self._gain_pixel * self._t_exposure * bayer
        voltage_bayer[voltage_bayer > 255] = 255

        print("Maximum value of Bayer image:")     
        print(np.max(voltage_bayer))       
        
        bayer_8bits = (voltage_bayer).astype(np.uint8)
        
        rgb_cv = cv2.cvtColor(bayer_8bits, cv2.COLOR_BAYER_RG2BGR)

        return rgb_cv
    
    def _add_noise_to_raw_image(
            self,
            raw_image,
            current_image,
            idark,
            gain,
            B,
            T) -> np.ndarray:
        
        i_mean = np.mean(current_image) 
        print("Current mean:")
        print(i_mean)

        # Computes the squared standard deviation
        thermal_sigma2 = 4 * Kt.KB * T * B / gain
        shot_sigma2 = 2 * Kt.QE * (i_mean + idark) * B

        print("Variance of noise")
        print(thermal_sigma2, shot_sigma2)

        # Computes total sigma
        sigma = (gain * (thermal_sigma2 + shot_sigma2)) ** 0.5

        # Generate noise samples        
        noise = np.random.normal(loc=0, scale=sigma, size=raw_image.shape)

        # Scale the noise samples to fit within the range of uint8 (0-255)
        max_value = np.iinfo(np.uint8).max
        min_value = np.iinfo(np.uint8).min
        
        scaled_noise = np.clip(noise, min_value, max_value)      

        # Add noise to the raw image        
        noisy_raw_image = raw_image.astype(float) + scaled_noise

        # Convert the image back to uint8 data type
        noisy_raw_image = noisy_raw_image.astype(np.uint8)

        return noisy_raw_image
        
    def plot_color_image(self):

        plt.imshow(self._noisy_image)
        plt.title('RGB Image')        
        plt.show()
    
    def add_blur(self, size=7, center=3.5, sigma=1.5) -> np.ndarray:    
        """ This function applies the point spread function of the power image. """
                
        print("Adding blur effect ...")
        # Apply Gaussian blur to the image
        blurred_image = cv2.GaussianBlur(
            self._noisy_image,
            (size, size),
            sigma, sigma
            )

        self._blurred_image = blurred_image

        return blurred_image        
    
    def plot_blurred_image(self) -> None:
        """ Plot the original image and the blurred image """
       
        plt.imshow(self._blurred_image)
        plt.title('Blurred image with PSF')        
        plt.show()

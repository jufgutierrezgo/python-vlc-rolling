from vlc_rolling.constants import Constants as Kt

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
from vlc_rolling.transmitter import Transmitter
# import surface module 
from vlc_rolling.indoorenv import Indoorenv
# import image sensor module 
from vlc_rolling.imagesensor import Imagesensor


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
        t_rowshift: float,
        t_offset: float,
        iso: float,
        adc_resolution: int,        
        gain_pixel: float,
        temperature: float,
        idark: float,
        transmitter: Transmitter,
        imagesensor: Imagesensor        
            ) -> None:

        self._name = name

        self._t_exposure = np.float32(t_exposure)        
        if self._t_exposure <= 0:
            raise ValueError("The exposure time must be a float non-negative.")
        
        self._t_rowshift = np.float32(t_rowshift)        
        if self._t_rowshift <= 0:
            raise ValueError("The row delay time must be a float non-negative.")
        self._bandwidth = 1/self._t_rowshift

        self._t_offset = np.float32(t_offset)        
        if self._t_offset < 0:
            raise ValueError(
                "The row adquisition start time must be non-negative."
                )

        self._iso = np.int32(iso)        
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
                "Transmitter attribute must be an object type Transmitter.")
        
        self._imagesensor = imagesensor
        if not type(imagesensor) is Imagesensor:
            raise ValueError(
                "Image sensor attribute must be an object type Image sensor.")
        
        self._init_rollingshutter()
            
    def _init_rollingshutter(self) -> None:
        
        self._image_current = self._compute_image_current(
            symbols_csk=self._transmitter._symbols_csk,
            im_crosstalk_rgblinear=self._imagesensor._image_H_rgblinear,
            height=self._imagesensor._image_height,
            width=self._imagesensor._image_width,
            t_rowshift=self._t_rowshift,
            t_offset=self._t_offset,
            t_exposure=self._t_exposure
            )
        
        self._rgb_image = self._bayerGBGR_to_RGB(
            current_image=self._image_current,
            height=self._imagesensor._image_height,
            width=self._imagesensor._image_width
            )

        self._noisy_image = self._add_noise_to_raw_image(
            raw_image=self._rgb_image,
            current_image=self._image_current,
            idark=self._idark,
            gain=self._gain_pixel,
            B=self._bandwidth,
            T=self._temperature
            )

    def _compute_image_current(
            self, 
            symbols_csk, 
            im_crosstalk_rgblinear,
            height,
            width,
            t_rowshift,
            t_offset,
            t_exposure) -> np.ndarray:
        """ This function computes the image with the transmitter symbols. """

        image_current = np.zeros((height, width))
        image_color = np.zeros((height, width))

        logging.info(f"Symbols CSK array shape: {symbols_csk.shape}")

        # compute the time of each symbol. It is equal to 1/f.
        self._t_symbol = 1/self._transmitter._frequency
        t_symbol = self._t_symbol

        for row in range(0,height):

            t_start = t_offset + row * t_rowshift
            t_end = t_start + t_exposure

            t_switch = np.ceil(
                ((row * t_rowshift) + t_offset) / t_symbol
                ) / t_symbol

            no_symbol = np.int32(
                np.floor(
                    ((row * t_rowshift) + t_offset) / t_symbol
                )
            )          
            
            if t_end <= t_switch:
                # results = c()*si
                symbol_csk = np.reshape(symbols_csk[:, no_symbol],(1,3))
                
                # symbol_csk = np.transpose(symbol_csk)
                image_current[row, :] = np.sum(
                    im_crosstalk_rgblinear[row, :, :] * symbol_csk,
                    axis=1
                    ).T

                logging.debug(f"Symbol CSK shape:{symbol_csk.shape}")

            else:
                # results = c()*si
                coeff1 = ((t_switch - t_start)/t_exposure)
                coeff2 = ((t_end - t_switch)/t_exposure)
                symbol_csk = np.int32(coeff1*symbols_csk[:, no_symbol] + coeff2*symbols_csk[:, no_symbol+1])
                
                # symbol_csk = np.transpose(symbol_csk)
                image_current[row, :] = np.sum(
                    im_crosstalk_rgblinear[row, :, :] * symbol_csk,
                    axis=1
                    ).T
                
                logging.debug(f"Symbol CSK shape:{symbol_csk.shape}")
                
        logging.info(f"No Symbol:{no_symbol}")
        logging.info(f"Im_crosstalk_rgblinear shape: {im_crosstalk_rgblinear.shape}")
        logging.info(f"Image current shape: {image_current}")

        return image_current
    
    def plot_current_image(self):
        """ Plot the image of the photocurrent by each pixel. """        
                
        # Plot power image
        plt.imshow(self._image_current, cmap='gray', interpolation='nearest')
        plt.title("Image of the normalized received power")        
        plt.show()
  
    def _bayerGBGR_to_RGB(self, current_image, height, width):
        """ Plot the image of the photocurrent by each pixel. """        
        
        # bayer = bayer / np.max(bayer)
        voltage_bayer = self._gain_pixel * self._t_exposure * current_image
        voltage_bayer[voltage_bayer > 255] = 255
                
        bayer_8bits = (voltage_bayer).astype(np.uint8)
        
        rgb_cv = cv2.cvtColor(bayer_8bits, cv2.COLOR_BAYER_RG2BGR)

        logging.info(f"Maximum value of Bayer image:{np.max(voltage_bayer)}")     
        logging.info(f"Voltage image: {voltage_bayer}")

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
    
    def plot_color_image(self, save='off'):

        logging.info(f"RGB image: {self._rgb_image}")   

        plt.imshow(self._rgb_image)
        plt.axis('off')  # Hide axis and ticks
        plt.tight_layout(pad=0)  # Remove padding around image

        if save.lower() == 'on':
            plt.savefig("color_rolling_image.pdf", bbox_inches='tight', pad_inches=0)

        plt.show()

    def plot_color_noisy_image(self):

        logging.info(f"Noisy image:", self._noisy_image)   

        plt.imshow(self._rgb_image)
        plt.axis('off')  # Hide axis and ticks
        plt.tight_layout(pad=0)  # Remove padding around image

        plt.imshow(self._noisy_image)
        plt.title('RGB Image with Noise')        
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

        plt.imshow(self._rgb_image)
        plt.axis('off')  # Hide axis and ticks
        plt.tight_layout(pad=0)  # Remove padding around image

        plt.imshow(self._blurred_image)
        # plt.title('Blurred image with PSF')        
        plt.show()

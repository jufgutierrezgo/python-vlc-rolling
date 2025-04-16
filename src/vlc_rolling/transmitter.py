from vlc_rolling.constants import Constants as Kt

from vlc_rolling.indoorenv import Indoorenv 

from vlc_rolling.sightpy import *

# Numeric Numpy library
import numpy as np

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt

from scipy import stats

import luxpy as lx


# from indoorenv import scene_rt

# from sightpy import Sphere, rgb, vec3

class Transmitter:
    """
    This class defines the transmitter features
    """

    def __init__(
        self,
        name: str,
        room: Indoorenv,
        position: np.ndarray,
        normal: np.ndarray,
        wavelengths: np.ndarray,
        fwhm: np.ndarray,
        mlambert: float = 1,        
        modulation: str = 'ieee16',
        frequency: float = 1000,
        no_symbols: int = 1000,
        luminous_flux: float = 1,        
            ) -> None:

        self._name = name

        self._room = room
        if not type(self._room) is Indoorenv:
            raise ValueError(
                "Indoor environment attribute must be an object type IndoorEnv.")

        self._position = np.array(position, dtype=np.float32)
        if self._position.size != 3:
            raise ValueError("Position must be an 1d-numpy array [x y z].")

        self._normal = np.array(normal,  dtype=np.float32)
        if not (isinstance(self._normal, np.ndarray)) or self._normal.size != 3:
            raise ValueError("Normal must be an 1d-numpy array [x y z] dtype= float or int.")        

        self._mlambert = np.float32(mlambert)
        if self._mlambert.size > 1:
            raise ValueError("Lambert number must be scalar float.")
        elif mlambert <= 0:
            raise ValueError("Lambert number must be greater than zero.")        

        self._wavelengths = np.array(wavelengths, dtype=np.float32)
        if self._wavelengths.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of wavelengths array must be equal to the number of LEDs.")
        elif (np.any(self._wavelengths > 780) or np.any(self._wavelengths < 380)):
            raise ValueError(
                "Wavelengths must be between 380nm and 780 nm.")

        self._fwhm = np.array(fwhm, dtype=np.float32)
        if self._fwhm.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of FWHM array must be equal to the number of LEDs.")
        elif np.any(self._fwhm <= 0):
            raise ValueError(
                "FWDM must be non-negative.")


        self._modulation = modulation
        # define the modulation
        if self._modulation == 'ieee16':
            self._constellation = Kt.IEEE_16CSK
            self._order_csk = 16
        elif self._modulation == 'ieee8':
            self._constellation = Kt.IEEE_8CSK
            self._order_csk = 8
        elif self._modulation == 'ieee4':
            self._constellation = Kt.IEEE_4CSK
            self._order_csk = 4
        elif self._modulation == 'warm-16':
            self._constellation = Kt.WARM_16CSK
            self._order_csk = 16
        elif self._modulation == 'white-16':
            self._constellation = Kt.WHITE_16CSK
            self._order_csk = 16
        else:
            raise ValueError("Modulation is not valid.")

        self._frequency = np.float32(frequency)        
        if self._frequency <= 0:
            raise ValueError("Frequency must be non-negative.")

        if isinstance(no_symbols, (int, float)):
            self._no_symbols = int(no_symbols)        
        else:
            raise ValueError(
                "No. of symbols must be a positive integer.")
        
        if self._no_symbols <= 0:
            raise ValueError(
                "No. of symbols must be greater than zero.")

        self._luminous_flux = np.float32(luminous_flux)        
        if self._luminous_flux <= 0:
            raise ValueError("The luminous flux must be non-negative.")


        # Initial functions
        self._create_spd_1w()
        self._compute_iler(self._spd_1w)
        self._avg_power_color()
        self._create_spd_1lm()
        # self._create_random_symbols()
        self._create_test_symbols()
        self._compute_cct_cri()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position,  dtype=np.float32)
        if self._position.size != 3:
            raise ValueError("Position must be a 3d-numpy array.")

    @property
    def normal(self) -> np.ndarray:
        return self._normal

    @normal.setter
    def normal(self, normal):
        self._normal = np.array(normal,  dtype=np.float32)
        if self._normal.size != 3:
            raise ValueError("Normal must be a 3d-numpy array.")

    @property
    def mlambert(self) -> float:
        return self._mlambert

    @mlambert.setter
    def mlambert(self, mlabert):
        if mlabert <= 0:
            raise ValueError("Lambert number must be greater than zero.")
        self._mlambert = mlabert

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        self._wavelengths = np.array(wavelengths,  dtype=np.float32)
        if self._wavelengths.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of wavelengths array must be equal to the number of LEDs.")

    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        self._fwhm = np.array(fwhm,  dtype=np.float32)
        if self._fwhm.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of FWHM array must be equal to the number of LEDs.") 

    @property
    def modulation(self) -> str:
        return self._modulation

    @modulation.setter
    def modulation(self, modulation):
        self._modulation = modulation
        # define the modulation
        if self._modulation == 'ieee16':
            self._constellation = Kt.IEEE_16CSK
            self._order_csk = 16
        elif self._modulation == 'ieee8':
            self._constellation = Kt.IEEE_8CSK
            self._order_csk = 8
        elif self._modulation == 'ieee4':
            self._constellation
            self._order_csk = 4
        else:
            print("Modulation name is not valid")

    @property
    def luminous_flux(self) -> float:
        return self._luminous_flux

    @luminous_flux.setter
    def luminous_flux(self, luminous_flux):
        if luminous_flux < 0:
            raise ValueError("The luminous flux must be non-negative.")
        self._luminous_flux = luminous_flux

    def __str__(self) -> str:
        return (
            f'\n List of parameters for LED transmitter: \n'
            f'Name: {self._name}\n'
            f'Position [x y z]: {self._position} \n'
            f'Normal Vector [x y z]: {self._normal} \n'
            f'Lambert Number: {self._mlambert} \n'            
            f'Central Wavelengths [nm]: {self._wavelengths} \n'
            f'FWHM [nm]: {self._fwhm}\n'
            f'Luminous Flux [lm]: {self._luminous_flux}\n'
            f'ILER [W/lm]: \n {self._iler_matrix} \n'
            f'Average Power per Channel Color: \n {self._luminous_flux*self._avg_power} \n'
            f'Total Power emmited by the Transmitter [W]: \n {self._total_power} \n'
            f'Color Temperature: \n {self._cct}'
            
        )

    def plot_led_pattern(self) -> None:
        """ Function to create a 3d radiation pattern of the LED source.

        The LED for recurse channel model is assumed as lambertian radiator.
        The number of lambert defines the directivity of the light source.

        """

        theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi/2, 40)
        THETA, PHI = np.meshgrid(theta, phi)
        R = (self._mlambert + 1)/(2*np.pi)*np.cos(PHI)**self._mlambert
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)

        plt.show()
        
    def _create_light_source_in_scene(self):
        """
        This function creates the light source into the 
        3D scene in the ray-tracer module
        """
        emissive_white =Emissive(color = rgb(20., 20., 20.))

        Sc.add(
            Plane(
                material = emissive_white,  
                center = vec3(213 + 130/2, 554, -227.0 - 105/2), 
                width = 130.0, 
                height = 105.0, 
                u_axis = vec3(1.0, 0.0, 0), 
                v_axis = vec3(0.0, 0, 1.0)), 
            importance_sampled = True
            )


    def _create_spd_1w(self):
        """
        This function creates the  spectrum of the LEDs 
        from central wavelengths and FWHM. 
        """
        # Array for wavelenght points from 380nm to (782-2)nm with 1nm steps
        self._array_wavelenghts = np.arange(400, 701, 1)

        # Numpy Array to save the spectral power distribution of each color channel
        self._spd_1w = np.zeros((self._array_wavelenghts.size, Kt.NO_LEDS))
        self._spd_normalized = np.zeros((self._array_wavelenghts.size, Kt.NO_LEDS))

        for i in range(Kt.NO_LEDS):
            # Arrays to estimates the normalized spectrum of LEDs
            self._spd_1w[:, i] = stats.norm.pdf(
                self._array_wavelenghts, self._wavelengths[i], self._fwhm[i]/2)
            
            self._spd_normalized[:, i] = self._spd_1w[:, i]/np.max(self._spd_1w[:, i])
    
    def _create_spd_1lm(self):
        """
        This function computes the Spectral Power Distribution in each 
        LED to produce 1 lumen.
        """  
        self._spd_1lm = np.zeros((self._array_wavelenghts.size, Kt.NO_LEDS))

        for i in range(Kt.NO_LEDS):
            self._spd_1lm[:, i] = self._avg_power[i] * self._spd_1w[:, i]

    def _create_light_source_raytracer(self)->None:
        " This function define the light source for the raytracer algorithm "
        
        # Define a difusse material for the light source
        emissive_white =Emissive(color = rgb(15., 15., 15.))

        # Define the plane for the light source
        light_source_plane = Plane(
                material = emissive_white, 
                center = vec3(213 + 130/2, 554, -227.0 - 105/2), 
                width = 330.0, 
                height = 305.0, 
                u_axis = vec3(1.0, 0.0, 0), 
                v_axis = vec3(0.0, 0, 1.0))
        # Define the light suource based on the emissive material
        self._scene_raytracer.add(
            # Plane(
            #     material = emissive_white, 
            #     center = vec3(213 + 130/2, 554, -227.0 - 105/2), 
            #     width = 330.0, 
            #     height = 305.0, 
            #     u_axis = vec3(1.0, 0.0, 0), 
            #     v_axis = vec3(0.0, 0, 1.0)), 
            # importance_sampled = True)
            light_source_plane,
            importance_sampled = True)

    def plot_spd_at_1lm(self):
        # plot red spd data
        for i in range(Kt.NO_LEDS):
            plt.plot(self._array_wavelenghts, self._spd_1lm[:, i])
        
        plt.title("Spectral Power Distribution at 1 Lumen/channel")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Power [W]")
        plt.grid()
        plt.show()
    
    def plot_spd_normalized(self):
        # plot red spd data
        for i in range(Kt.NO_LEDS):
            plt.plot(
                self._array_wavelenghts,
                # self._spd_normalized[:, i]
                self._spd_1w[:, i]
                )
        
        plt.title("Normalized Spectral Power Distribution")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Normalied Power [W]")
        plt.grid()
        plt.show()
    
    
    def _compute_iler(self, spd_data) -> None:        
        """
        This function computes the inverse luminous efficacy radiation (LER) matrix.
        This matrix has a size of NO_LEDS x NO_LEDS
        """
        self._photometric = lx.spd_to_power(
            np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, 0]
                    ]),
            'pu'
        )
        self._radiometric = lx.spd_to_power(
            np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, 0]
                    ]),
            'ru'
        )
        self._iler_matrix = np.zeros((Kt.NO_LEDS, Kt.NO_LEDS))

        for i in range(Kt.NO_LEDS):
            self._iler_matrix[i, i] = 1/lx.spd_to_ler(
                np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, i]
                    ])
                )

    def _avg_power_color(self):
        """
        This function computes the average radiometric power emmitted by 
        each color channel in the defined constellation.
        """
        
        self._avg_lm = np.mean(
                    self._constellation,
                    axis=1
                    )
        self._avg_power = np.transpose(
            np.matmul(
                self._iler_matrix,
                self._avg_lm
                )
            )

        self._total_power = self._luminous_flux*np.sum(self._avg_power)
        # Manual setted of avg_power by each color channels
        #self._avg_power = np.array([1, 1, 1])

    def _create_random_symbols(self) -> None:
        """
        This function creates the symbols array to transmit.
        """
        # create a random symbols identifier (decimal) for payload
        self._symbols_decimal = np.random.randint(
                0,
                self._order_csk-1,
                (self._no_symbols),
                dtype='int16'
            )

        self._symbols_payload = np.zeros((Kt.NO_LEDS, self._no_symbols))

        self._constellation = self._constellation
        
        for index, counter in zip(self._symbols_decimal, range(self._no_symbols)):
            self._symbols_payload[:, counter] = self._constellation[:, index]

        # Define the number of symbols for delimiter header
        self._delimiter_set = 3

        # add to the payload three base-set of symbols
        self._symbols_csk = np.concatenate((
                np.identity(Kt.NO_LEDS),
                np.identity(Kt.NO_LEDS),
                np.identity(Kt.NO_LEDS),
                self._symbols_payload),
                axis=1
            )
    
    def _create_test_symbols(self) -> None:
        """ This function create a symbol test frame """
        
        # Create the elementary frame
        elementary_frame = np.array(
            [15, 5, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            )
        
        # create a random symbols identifier (decimal) for payload
        self._symbols_decimal = np.tile(elementary_frame, 8)

        # create an array with the symbols
        self._symbols_csk = np.zeros((Kt.NO_LEDS, len(self._symbols_decimal)))

        for index, counter in zip(self._symbols_decimal, range(len(self._symbols_decimal))):
            self._symbols_csk[:, counter] = self._constellation[:, index]        

    def _compute_cct_cri(self) -> None:
        """ This function calculates a CCT and CRI of the QLED SPD."""

        # Computing the xyz coordinates from SPD-RGBY estimated spectrum
        self._XYZ_uppper = lx.spd_to_xyz(
            [
                self._array_wavelenghts,
                np.sum(self._spd_1lm, axis=1)
            ])

        # Example of xyx with D65 illuminant     
        # self._xyz = lx.spd_to_xyz(
        #    lx._CIE_ILLUMINANTS['D65']
        #    )

        self._xyz = self._XYZ_uppper/np.sum(self._XYZ_uppper)

        # Computing the CRI coordinates from SPD-RGBY estimated spectrum
        self._cri = lx.cri.spd_to_cri(
            np.vstack(
                    [
                        self._array_wavelenghts,
                        np.sum(self._spd_1lm, axis=1)

                    ]
                )
            )
        
        # Example of CRI for D65 illuminant
        # self._cri = lx.cri.spd_to_cri(lx._CIE_ILLUMINANTS['D65'])

        # Computing the CCT coordinates from SPD-RGBY estimated spectrum
        self._cct = lx.xyz_to_cct_ohno2014(self._xyz)

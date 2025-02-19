# import numpy library
import numpy as np


class Indoorenv:
    """
    This class defines the indoor environment features, and computes
    the points grid and the cosine and distance pair-waise.

    """

    def __init__(
        self,
        name: str,
        size: np.ndarray,
        resolution: float,
        ceiling: np.ndarray,
        west: np.ndarray,
        north: np.ndarray,
        east: np.ndarray,
        south: np.ndarray,
        floor: np.ndarray,
        no_reflections: int = 3        
    ) -> None:

        self._name = name
        self.deltaA = 'Non defined, create grid.'
        self.no_points = 'Non defined, create grid.'

        self._size = np.array(size, dtype=np.float32)
        if self._size.size != 3:
            raise ValueError(
                "Size of the indoor environment must be an 1d-numpy array [x y z]")        

        self._no_reflections = no_reflections
        if not isinstance(self._no_reflections, int):
            raise ValueError(
                "No of reflections must be a positive integer between 0 and 10.")
        if self._no_reflections < 0 or self._no_reflections > 20:
            raise ValueError(
                "No of reflections must be a real integer between 0 and 10.")
        
        self._resolution = np.float32(resolution)
        if self._resolution > min(self._size):
            raise ValueError(
                "Resolution of points must be less or equal to minimum size of the rectangular indoor space.")
        if self._resolution <= 0:
            raise ValueError(
                "Resolution of points must be a real number greater than zero.")

        self._ceiling = np.array(ceiling)
        if self._ceiling.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of ceiling reflectance array must be equal to the number of wavelengths.")

        self._west = np.array(west)
        if self._west.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of north reflectance array must be equal to the number of wavelengths.")
                        
        self._north = np.array(north)
        if self._north.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of north reflectance array must be equal to the number of wavelengths.")
        
        self._east = np.array(east)
        if self._east.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of east reflectance array must be equal to the number of wavelengths.")

        self._south = np.array(south)
        if self._south.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of south reflectance array must be equal to the number of wavelengths.")

        self._floor = np.array(floor)
        if self._floor.size != Kt.NO_WAVELENGTHS:
            raise ValueError(
                "Dimension of floor reflectance array must be equal to the number of wavelengths.")

    @property
    def name(self):
        """The name property"""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def size(self) -> np.ndarray:
        """The size property"""
        return self._size

    @size.setter
    def size(self, size):
        self._size = np.array(size)
        if self._size.size != 3:
            raise ValueError(
                "Size of the indoor environment must be an 1d-numpy array [x y z]")

    @property
    def no_reflections(self) -> int:
        """The number of reflections property"""
        return self._no_reflections

    @no_reflections.setter
    def no_reflections(self, no_reflections):
        self._no_reflections = no_reflections
        if self._no_reflections <= 0:
            raise ValueError(
                "Resolution of points must be a real integer between 0 and 10.")

    @property
    def resolution(self) -> float:
        """The resolution property"""
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        self._resolution = resolution
        if self._resolution <= 0:
            raise ValueError(
                "Resolution of points must be a real number greater than zero.")

    @property
    def ceiling(self) -> np.ndarray:
        """The ceiling property"""
        return self._ceiling

    @ceiling.setter
    def ceiling(self, ceiling):
        self._ceiling = np.array(ceiling)
        if self._ceiling.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of ceiling reflectance array must be equal to the number of LEDs.")

    @property
    def west(self) -> np.ndarray:
        """The west property"""
        return self._west

    @west.setter
    def west(self, west):
        self._west = np.array(west)
        if self._ceiling.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of west reflectance array must be equal to the number of LEDs.")

    @property
    def north(self) -> np.ndarray:
        """The north property"""
        return self._north

    @north.setter
    def north(self, north):
        self._north = np.array(north)
        if self._north.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of north reflectance array must be equal to the number of LEDs.")

    @property
    def east(self) -> np.ndarray:
        """The east property"""
        return self._east

    @east.setter
    def east(self, east):
        self._east = np.array(east)
        if self._east.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of east reflectance array must be equal to the number of LEDs.")

    @property
    def south(self) -> np.ndarray:
        """The east property"""
        return self._south

    @south.setter
    def south(self, south):
        self._south = np.array(south)
        if self._south.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of south reflectance array must be equal to the number of LEDs.")

    @property
    def floor(self) -> np.ndarray:
        """The floor property"""
        return self._floor

    @floor.setter
    def floor(self, floor):
        self._floor = np.array(floor)
        if self._floor.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of ceiling reflectance array must be equal to the number of LEDs.")


    
    def __str__(self) -> str:
        return (
            f'\n List of parameters for indoor envirionment {self._name}: \n'
            f'Name: {self._name}\n'
            f'Size [x y z] -> [m]: {self._size} \n'
            f'Order reflection: {self._no_reflections} \n'
            f'Resolution points [m]: {self._resolution}\n'
            f'Smaller Area [m^2]: {self.deltaA}\n'
            f'Number of points: {self.no_points}\n'
        )    

    def create_environment(
        self,
        tx: Transmitter,
        rx: Photodetector,
        mode: str = 'new'
            ) -> None:

        self._tx = tx
        if not type(self._tx) is Transmitter:
            raise ValueError(
                "Tranmistter attribute must be an object type Transmitter.")

        self._rx = rx
        if not type(self._rx) is Photodetector:
            raise ValueError(
                "Receiver attribute must be an object type Photodetector.")
        
        if (mode != 'new') and (mode != 'modified'):
            raise ValueError(
                "Mode attribute must be 'new' or 'modified'.")

        print("\n Creating parameters of indoor environment ...")

        self._create_grid(
            self._tx._position,
            self._rx._position,
            self._tx._normal,
            self._rx._normal
            )
        
        self._compute_parameters(self._rx._fov, mode)
        print("Parameters created!\n")

    
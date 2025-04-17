# import numpy library
import numpy as np

from vlc_rolling.constants import Constants as Kt

from vlc_rolling.sightpy import *

class Indoorenv:
    """
    This class defines the indoor environment features.
    """

    def __init__(
        self,
        name: str,
        size: np.ndarray,
        resolution: float,
        ceiling: tuple,
        west: tuple,
        north: tuple,
        east: tuple,
        south: tuple,
        floor: tuple,
        no_reflections: int = 3        
    ) -> None:

        VALID_WALL_TYPES = ('diffuse', 'glossy')

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

        self._ceiling = self._validate_wall(ceiling, "ceiling")
        self._west    = self._validate_wall(west, "west")
        self._north   = self._validate_wall(north, "north")
        self._east    = self._validate_wall(east, "east")
        self._south   = self._validate_wall(south, "south")
        self._floor   = self._validate_wall(floor, "floor")

    def _validate_wall(self, wall: tuple, wall_name: str):
        if not isinstance(wall, tuple) or len(wall) != 2:
            raise ValueError(f"{wall_name} must be a tuple: (type, color).")
        
        wall_type, color = wall

        if wall_type not in ('diffuse', 'glossy'):
            raise ValueError(f"{wall_name} type must be one of {('diffuse', 'glossy')}. Got '{wall_type}'.")

        # color = np.array(color)
    
        # if color.size != Kt.NO_WAVELENGTHS:
        #     raise ValueError(
        #         f"Color array for {wall_name} must have size equal to the number of wavelengths ({Kt.NO_WAVELENGTHS})."
        #     )

        return (wall_type, color)


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
    def ceiling(self) -> tuple:
        """The ceiling property (type, color array)"""
        return self._ceiling

    @ceiling.setter
    def ceiling(self, ceiling: tuple):
        self._ceiling = self._validate_wall(ceiling, "ceiling")

    @property
    def west(self) -> tuple:
        """The west property (type, color array)"""
        return self._west

    @west.setter
    def west(self, west: tuple):
        self._west = self._validate_wall(west, "west")

    @property
    def north(self) -> tuple:
        """The north property (type, color array)"""
        return self._north

    @north.setter
    def north(self, north: tuple):
        self._north = self._validate_wall(north, "north")

    @property
    def east(self) -> tuple:
        """The east property (type, color array)"""
        return self._east

    @east.setter
    def east(self, east: tuple):
        self._east = self._validate_wall(east, "east")

    @property
    def south(self) -> tuple:
        """The south property (type, color array)"""
        return self._south

    @south.setter
    def south(self, south: tuple):
        self._south = self._validate_wall(south, "south")

    @property
    def floor(self) -> tuple:
        """The floor property (type, color array)"""
        return self._floor

    @floor.setter
    def floor(self, floor: tuple):
        self._floor = self._validate_wall(floor, "floor")
    
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

    def create_environment(self) -> None:

                
        green_diffuse=Diffuse(diff_color = rgb(.12, .45, .15))
        red_diffuse=Diffuse(diff_color = rgb(.65, .05, .05))
        white_diffuse=Diffuse(diff_color = rgb(.73, .73, .73))
        emissive_white =Emissive(color = rgb(20., 20., 20.))
        emissive_blue =Emissive(color = rgb(2., 2., 3.5))
        blue_glass =Refractive(n = vec3(1.5 + 0.05e-8j,1.5 +  0.02e-8j,1.5 +  0.j))
        
        # Initialize of the scene for the indoor environment 
        self._scene_rt = Scene(ambient_color = rgb(0.00, 0.00, 0.00))

        # Add walls for the indoor environment
        self._scene_rt.add(Plane(material = white_diffuse,  center = vec3(555/2, 555/2, -555.0), width = 555.0,height = 555.0, u_axis = vec3(0.0, 1.0, 0), v_axis = vec3(1.0, 0, 0.0)))


    def render_environment(self) -> None:

        img = self._scene_rt.render(
            samples_per_pixel = 100,
            # progress_bar = True
            )

        # img.save("cornell_box.png")

        img.show()
    
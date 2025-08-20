==================
python-vlc-rolling
==================

.. image:: https://img.shields.io/pypi/v/vlc_rolling.svg
        :target: https://pypi.python.org/pypi/vlc_rolling

.. image:: https://img.shields.io/travis/jufgutierrezgo/vlc_rolling.svg
        :target: https://travis-ci.com/jufgutierrezgo/vlc_rolling

.. image:: https://readthedocs.org/projects/vlc-rolling/badge/?version=latest
        :target: https://vlc-rolling.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


**python-vlc-rolling** is a Python package for simulating Visible Light Communication (VLC) and Optical Camera Communication (OCC) systems using rolling shutter cameras and Color Shift Keying (CSK) modulation. The package provides classes to define indoor environments (including **NLOS** links with reflective surfaces), multicolor LED transmitters, image sensors, and the rolling shutter sampling process, facilitating research and experimentation in wireless optical communications.

The figure illustrates a typical OCC scenario with an NLOS link, where the user receives information by capturing the light reflected from a surface using a smartphone camera:

.. image:: https://github.com/jufgutierrezgo/rs-vlc-model/blob/main/images/OCC-rs-csk-nlos.png?raw=true
        :alt: OCC scenario with rolling shutter, CSK, and NLOS link via reflective surfaces
        :align: center


What are VLC and OCC?
---------------------

*Visible Light Communication (VLC)* uses light sources (e.g., LEDs) to transmit data by modulating their intensity. In *OCC*, a camera acts as the receiver, capturing the modulated light. With a **rolling shutter** sensor, the row-by-row sequential readout converts high-frequency temporal modulation into spatial patterns (stripes) on the image, which can then be decoded. Color-based techniques such as **CSK** exploit multiple channels (e.g., RGB) to increase the system’s spectral efficiency.


Key Features
------------

- **Modular framework**: separate classes for indoor environment, transmitters, sensors, and rolling shutter process, making it easy to extend and adapt.
- **LoS and NLoS links**: modeling **reflections** with configurable reflectance surfaces (walls, ceiling, floor).
- **Color modulation (CSK)**: 8-CSK, 16-CSK, and higher constellations with configurable LED spectra by wavelength.
- **Realistic sensor models**: CMOS sensor quantum efficiency (QE) curves and functions for responsivity and image formation.
- **Channel and noise analysis**: optical gains, shot and thermal noise, and color channel crosstalk effects.
- **Configurable rolling shutter**: exposure and frame readout times to study temporal–spatial aliasing and detectable bands.
- **Examples and notebooks**: ready-to-run scripts to build scenarios, visualize CSK constellations, and decode patterns.


Installation
------------

Install the stable version from PyPI:

.. code-block:: bash

    pip install vlc_rolling

To install the development version from the ``main`` branch:

.. code-block:: bash

    pip install https://github.com/jufgutierrezgo/python-vlc-rolling/archive/main.zip

For local editable development:

.. code-block:: bash

    git clone https://github.com/jufgutierrezgo/python-vlc-rolling.git
    cd python-vlc-rolling
    python -m venv venv
    source venv/bin/activate  # or ``venv\Scripts\activate`` on Windows
    pip install -e .[dev]


Basic Usage
-----------

Minimal example setting up a room with an RGB transmitter and a rolling shutter camera:

.. code-block:: python

    import vlc_rolling as vlc

    # Indoor environment 5 m × 5 m × 3 m with neutral walls
    env = vlc.Indoorenv(
        room_dimensions=(5, 5, 3),
        wall_reflectance=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7)  # (N, S, E, W, ceiling, floor)
    )

    # Transmitter at (2.5, 2.5, 3) pointing down (-Z axis)
    tx = vlc.Transmitter(
        wavelengths=(450, 520, 620),         # nm (RGB peaks)
        luminous_flux=1000,                  # lumens
        modulation='8-CSK',                  # CSK constellation
        frequency=1000,                      # Hz (carrier/clock)
        position=(2.5, 2.5, 3.0),            # meters (X, Y, Z)
        orientation=(0, 0, -1)               # unit vector
    )

    # Image sensor (e.g., Sony Starvis BSI model)
    sensor = vlc.Imagesensor(
        model='starvis_bsi',
        resolution=(1920, 1080),             # pixels (width, height)
        pixel_size=3.75e-6,                  # meters
        focal_length=0.012                   # meters
    )

    # Rolling shutter parameters
    rs = vlc.Rollingshutter(
        exposure_time=1/1000,                # s
        readout_time=1/60                    # s (per frame)
    )

    # Run simulation and obtain RGB pattern
    image = rs.capture(env, tx, sensor)

    # Visualize result (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title("RS-CSK Pattern (example)")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.show()


NLoS Example with Reflective Surfaces
-------------------------------------

This example shows a non-line-of-sight (NLoS) link using higher wall and ceiling reflectances. Useful to study how the rolling shutter captures reflected information.

.. code-block:: python

    import vlc_rolling as vlc

    env = vlc.Indoorenv(
        room_dimensions=(6, 4, 3),
        wall_reflectance=(0.85, 0.85, 0.85, 0.85, 0.9, 0.6)  # more reflective walls and ceiling
    )

    tx = vlc.Transmitter(
        wavelengths=(455, 525, 625),
        luminous_flux=1500,
        modulation='16-CSK',
        frequency=2000,
        position=(3.0, 2.0, 2.9),
        orientation=(0.2, 0.0, -0.98)  # slight tilt towards a wall
    )

    sensor = vlc.Imagesensor(
        model='starvis_bsi',
        resolution=(2560, 1440),
        pixel_size=2.9e-6,
        focal_length=0.006
    )

    rs = vlc.Rollingshutter(
        exposure_time=1/800,
        readout_time=1/30
    )

    image = rs.capture(env, tx, sensor)

    # (Optional) CSK symbol decoding/estimation
    # symbols = vlc.decode_csk(image, method="ml")  # if available in API

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title("RS-CSK Pattern in NLoS Link (reflections)")
    plt.show()


Notes on Rolling Shutter and Detectable Frequency
-------------------------------------------------

- The ``readout_time`` parameter represents the **frame readout time**. Row time is approximated as:
  ``t_row = readout_time / resolution[1]``.
- To avoid excessive aliasing, the effective detectable frequency is roughly bounded by:
  ``f_max ≈ 1 / (2 * t_row)``.
- The choice of ``frequency`` (modulation) and ``exposure_time`` impacts stripe visibility and signal-to-noise ratio.


Documentation
-------------

The complete documentation (tutorials, API reference, and theoretical background) is available at Read the Docs:

https://vlc-rolling.readthedocs.io

If the site is unavailable, you can build the documentation locally with Sphinx:

.. code-block:: bash

    cd docs
    make html

Generated HTML will be in ``docs/_build/html``.


Contributing
------------

Contributions are welcome! To report bugs, suggest improvements, or add support for new sensors/constellations:

- Fork the repository and create a new branch for your contribution.
- Follow the project’s code style and include tests.
- Run ``make lint`` and ``make test`` before submitting a pull request.
- Update documentation and examples when adding new features.
- See ``CONTRIBUTING.rst`` for more details on the workflow.


License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for more details.


Authors
-------

`Juan-Felipe Gutiérrez-Gómez <jufgutierrezgo@unal.unal.edu.co>`_ is the creator and main maintainer. The full list of contributors can be found in ``AUTHORS.rst``.


Citation
--------

If you use this package or derived results in academic work, please cite:

**Published paper (OCC NLoS with RS and CSK)**

J. F. Gutierrez, D. Sandoval, and J. M. Quintero, *“An Analytical Performance Study of a Non-Line-of-Sight Optical Camera Communication System Based on Rolling Shutter and Color Shift Keying,”* in **2023 IEEE Sustainable Smart Lighting World Conference & Expo (LS18)**, Mumbai, India, 2023, pp. 1–6. doi: 10.1109/LS1858153.2023.10170645.

.. code-block:: bibtex

    @inproceedings{Gutierrez2023OCC_RS_CSK,
      author    = {J. F. Gutierrez and D. Sandoval and J. M. Quintero},
      title     = {An Analytical Performance Study of a Non-Line-of-Sight Optical Camera Communication System Based on Rolling Shutter and Color Shift Keying},
      booktitle = {2023 IEEE Sustainable Smart Lighting World Conference \& Expo (LS18)},
      address   = {Mumbai, India},
      year      = {2023},
      pages     = {1--6},
      doi       = {10.1109/LS1858153.2023.10170645}
    }

**Repository and package**

Cite this repository as: *python-vlc-rolling: OCC Simulation with Rolling Shutter and CSK in LoS/NLoS links*. Include the project URL and the PyPI version number you used.

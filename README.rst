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


**python-vlc-rolling** es un paquete de Python para simular sistemas de comunicación por luz visible (VLC) y comunicación óptica por cámara (OCC) utilizando cámaras de obturador rodante (*rolling shutter*) y modulación por desplazamiento de color (CSK). El paquete proporciona clases para definir entornos interiores (incluyendo enlaces **NLOS** con superficies reflectantes), transmisores LED multicolor, sensores de imagen y el proceso de muestreo *rolling shutter*, facilitando la investigación y la experimentación en comunicaciones ópticas inalámbricas.

La figura ilustra un escenario típico OCC con enlace NLOS, donde el usuario recibe información capturando la luz reflejada en una superficie mediante la cámara de un smartphone:

.. image:: https://github.com/jufgutierrezgo/rs-vlc-model/blob/main/images/OCC-rs-csk-nlos.png?raw=true
        :alt: Escenario OCC con obturador rodante, CSK y enlace NLOS mediante superficies reflectantes
        :align: center


¿Qué son VLC y OCC?
--------------------

La *comunicación por luz visible (VLC)* emplea fuentes de luz (p. ej., LED) para transmitir datos modulando su intensidad. En *OCC*, una cámara actúa como receptor capturando la luz modulada. Con un sensor de **obturador rodante**, la lectura secuencial por filas convierte la modulación temporal de alta frecuencia en patrones espaciales (franjas) sobre la imagen, que luego pueden decodificarse. Técnicas basadas en color como **CSK** permiten aprovechar múltiples canales (p. ej., RGB) para incrementar la eficiencia espectral del sistema.


Características destacadas
--------------------------

- **Framework modular**: clases separadas para entorno interior, transmisores, sensores y el proceso de obturador rodante, facilitando su extensión y adaptación.
- **Enlaces LoS y NLoS**: modelado de **reflexiones** mediante superficies con reflectancia configurable (paredes, techo, piso).
- **Modulación por color (CSK)**: constelaciones 8-CSK, 16-CSK y superiores con espectros LED configurables por longitud de onda.
- **Modelos de sensores realistas**: curvas de eficiencia cuántica (QE) de sensores CMOS y funciones para responsividad y formación de imagen.
- **Análisis de canal y ruido**: ganancias ópticas, ruido de disparo y térmico, y efectos de diafonía entre canales de color.
- **Rolling shutter configurable**: tiempos de exposición y lectura por cuadro para estudiar aliasing temporal–espacial y bandas detectables.
- **Ejemplos y notebooks**: scripts listos para ejecutar para construir escenarios, visualizar constelaciones CSK y decodificar patrones.


Instalación
-----------

Instala la versión estable desde PyPI:

.. code-block:: bash

    pip install vlc_rolling

Para instalar la versión de desarrollo desde la rama ``main``:

.. code-block:: bash

    pip install https://github.com/jufgutierrezgo/python-vlc-rolling/archive/main.zip

Para trabajar localmente en modo editable:

.. code-block:: bash

    git clone https://github.com/jufgutierrezgo/python-vlc-rolling.git
    cd python-vlc-rolling
    python -m venv venv
    source venv/bin/activate  # o ``venv\Scripts\activate`` en Windows
    pip install -e .[dev]


Uso básico
----------

Ejemplo mínimo que configura una habitación con un transmisor RGB y una cámara de obturador rodante:

.. code-block:: python

    import vlc_rolling as vlc

    # Entorno interior de 5 m × 5 m × 3 m con paredes neutras
    env = vlc.Indoorenv(
        room_dimensions=(5, 5, 3),           # metros
        wall_reflectance=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7)  # (N, S, E, O, techo, piso)
    )

    # Transmisor en (2.5, 2.5, 3) apuntando hacia abajo (eje -Z)
    tx = vlc.Transmitter(
        wavelengths=(450, 520, 620),         # nm (picos RGB)
        luminous_flux=1000,                  # lúmenes
        modulation='8-CSK',                  # constelación CSK
        frequency=1000,                      # Hz (portadora/clock de modulación)
        position=(2.5, 2.5, 3.0),            # metros (X, Y, Z)
        orientation=(0, 0, -1)               # vector unitario
    )

    # Sensor de imagen (ej.: modelo Sony Starvis BSI)
    sensor = vlc.Imagesensor(
        model='starvis_bsi',
        resolution=(1920, 1080),             # píxeles (ancho, alto)
        pixel_size=3.75e-6,                   # metros
        focal_length=0.012                    # metros
    )

    # Parámetros del obturador rodante
    rs = vlc.Rollingshutter(
        exposure_time=1/1000,                # s
        readout_time=1/60                    # s (lectura por cuadro)
    )

    # Ejecutar la simulación y obtener el patrón RGB
    image = rs.capture(env, tx, sensor)

    # Visualizar el resultado (requiere matplotlib)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title("Patrón RS-CSK (ejemplo)")
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.show()


Ejemplo NLoS con superficies reflectantes
-----------------------------------------

Este ejemplo muestra un enlace no línea de vista (NLoS) usando reflectancias más altas en paredes y techo. Útil para estudiar cómo el *rolling shutter* captura información reflejada.

.. code-block:: python

    import vlc_rolling as vlc

    env = vlc.Indoorenv(
        room_dimensions=(6, 4, 3),
        wall_reflectance=(0.85, 0.85, 0.85, 0.85, 0.9, 0.6)  # paredes y techo más reflectivos
    )

    tx = vlc.Transmitter(
        wavelengths=(455, 525, 625),
        luminous_flux=1500,
        modulation='16-CSK',
        frequency=2000,
        position=(3.0, 2.0, 2.9),
        orientation=(0.2, 0.0, -0.98)  # leve inclinación hacia una pared
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

    # (Opcional) Decodificación/estimación de símbolos CSK a partir del patrón
    # symbols = vlc.decode_csk(image, method="ml")  # si está disponible en la API

    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title("Patrón RS-CSK en enlace NLoS (reflexiones)")
    plt.show()


Notas sobre *rolling shutter* y frecuencia detectable
-----------------------------------------------------

- El parámetro ``readout_time`` representa el **tiempo de lectura por cuadro**. El tiempo por fila se aproxima como:
  ``t_fila = readout_time / resolution[1]``.
- Para evitar aliasing excesivo, la frecuencia efectiva detectable está acotada aproximadamente por ``f_max ≈ 1 / (2 * t_fila)``.
- La elección de ``frequency`` (modulación) y ``exposure_time`` impacta la visibilidad del patrón de franjas y la relación señal-ruido.


Documentación
-------------

La documentación completa (tutoriales, referencia de API y fundamentos teóricos) está en Read the Docs:

https://vlc-rolling.readthedocs.io

Si el sitio no está disponible, puedes construir la documentación localmente con Sphinx:

.. code-block:: bash

    cd docs
    make html

El HTML generado estará en ``docs/_build/html``.


Cómo contribuir
---------------

¡Las contribuciones son bienvenidas! Para reportar errores, sugerir mejoras o añadir soporte a nuevos sensores/constelaciones:

- Haz un *fork* del repositorio y crea una rama nueva para tu contribución.
- Sigue el estilo de código del proyecto e incluye pruebas.
- Ejecuta ``make lint`` y ``make test`` antes de enviar un *pull request*.
- Actualiza la documentación y los ejemplos al agregar nuevas funcionalidades.
- Consulta ``CONTRIBUTING.rst`` para más detalles sobre el flujo de trabajo.


Licencia
--------

Este proyecto está licenciado bajo MIT. Consulta el archivo ``LICENSE`` para más detalles.


Autores
-------

`Juan-Felipe Gutiérrez-Gómez <jufgutierrezgo@unal.unal.edu.co>`_ es el creador y mantenedor principal. La lista completa de colaboradores se encuentra en ``AUTHORS.rst``.


Citación
--------

Si utilizas este paquete o resultados derivados en trabajos académicos, por favor cita:

**Artículo publicado (OCC NLoS con RS y CSK)**

J. F. Gutierrez, D. Sandoval y J. M. Quintero, *“An Analytical Performance Study of a Non-Line-of-Sight Optical Camera Communication System Based on Rolling Shutter and Color Shift Keying,”* en **2023 IEEE Sustainable Smart Lighting World Conference & Expo (LS18)**, Mumbai, India, 2023, pp. 1–6. doi: 10.1109/LS1858153.2023.10170645.

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

**Repositorio y paquete**

Cita este repositorio como: *python-vlc-rolling: Simulación de OCC con obturador rodante y CSK en enlaces LoS/NLoS*. Incluye la URL del proyecto y el número de versión de PyPI que utilizaste.

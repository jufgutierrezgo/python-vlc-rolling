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




Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://vlc-rolling.readthedocs.io.


Features
--------

* TODO

python-vlc-rolling
==================

**python-vlc-rolling** es un paquete de Python para simular sistemas de comunicación por luz visible (VLC) y comunicación óptica por cámara (OCC) utilizando cámaras de obturador rodante y modulación por desplazamiento de color (CSK). El paquete proporciona clases para definir entornos interiores, transmisores LED, sensores de imagen y el proceso de detección con obturador rodante, lo que facilita la investigación y la experimentación en comunicaciones ópticas inalámbricas.

¿Qué son VLC y OCC?
--------------------

La *comunicación por luz visible* emplea fuentes de luz como los LED para transmitir datos mediante la modulación de su intensidad. En OCC, una cámara actúa como receptor capturando la luz modulada. Cuando una cámara con sensor de obturador rodante observa patrones de parpadeo, la lectura secuencial de filas transforma la modulación temporal en patrones espaciales que pueden decodificarse.

Características destacadas
--------------------------

- **Framework modular**: clases separadas para el entorno, los transmisores, los sensores y el procesamiento del obturador rodante, lo que facilita su extensión y adaptación a nuevos escenarios.
- **Admite varias longitudes de onda y modulación CSK**: constelaciones 8‑CSK, 16‑CSK y superiores, con espectros LED configurables.
- **Simulación de entornos interiores**: dimensiones de sala y reflectancias de paredes personalizables para modelar línea de vista y reflexiones.
- **Modelos de sensores realistas**: se incluyen curvas de eficiencia cuántica de sensores CMOS comerciales y funciones para calcular la responsividad.
- **Análisis de ruido y diafonía**: considera ganancias de canal, ruido de disparo y ruido térmico para evaluar el rendimiento de la comunicación.
- **Ejemplos y notebooks**: scripts listos para ejecutar que muestran cómo construir escenarios, visualizar constelaciones CSK y decodificar patrones.

Instalación
-----------

Puedes instalar la versión estable desde PyPI con:

.. code-block:: bash

    pip install vlc_rolling

Esto instalará la última versión publicada en PyPI. Para instalar la versión en desarrollo directamente desde la rama ``main`` de GitHub:

.. code-block:: bash

    pip install https://github.com/jufgutierrezgo/python-vlc-rolling/archive/main.zip

Para trabajar en el desarrollo localmente, clona el repositorio e instala en modo editable:

.. code-block:: bash

    git clone https://github.com/jufgutierrezgo/python-vlc-rolling.git
    cd python-vlc-rolling
    python -m venv venv
    source venv/bin/activate  # o ``venv\Scripts\activate`` en Windows
    pip install -e .[dev]

Uso básico
----------

Este es un ejemplo mínimo que configura una habitación simple con un LED azul y una cámara de obturador rodante:

.. code-block:: python

    import vlc_rolling as vlc

    # Entorno interior de 5 m × 5 m × 3 m con paredes neutras
    env = vlc.Indoorenv(
        room_dimensions=(5, 5, 3),
        wall_reflectance=(0.7, 0.7, 0.7, 0.7, 0.7, 0.7)
    )

    # Transmisor en (2.5, 2.5, 3) apuntando hacia abajo
    tx = vlc.Transmitter(
        wavelengths=(450, 520, 620),
        luminous_flux=1000,
        modulation='8-CSK',
        frequency=1000,
        position=(2.5, 2.5, 3),
        orientation=(0, 0, -1)
    )

    # Sensor de imagen (modelo Sony Starvis BSI)
    sensor = vlc.Imagesensor(
        model='starvis_bsi',
        resolution=(1920, 1080),
        pixel_size=3.75e-6,
        focal_length=0.012
    )

    # Parámetros del obturador rodante
    rs = vlc.Rollingshutter(
        exposure_time=1/1000,
        readout_time=1/60
    )

    # Ejecutar la simulación y obtener el patrón RGB
    image = rs.capture(env, tx, sensor)

    # Visualizar el resultado (requiere matplotlib)
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

Consulta los directorios ``examples`` y ``examples/notebooks`` para más demostraciones y notebooks interactivos.

Documentación
-------------

La documentación completa, con tutoriales, referencia de API y fundamentos teóricos, se encuentra en Read the Docs:

https://vlc-rolling.readthedocs.io

Si el sitio no está disponible, puedes construir la documentación localmente con Sphinx:

.. code-block:: bash

    cd docs
    make html

El HTML generado estará en ``docs/_build/html``.

Cómo contribuir
---------------

¡Las contribuciones son bienvenidas! Para reportar errores, sugerir mejoras o añadir soporte a nuevos sensores/constelaciones, sigue estas pautas:

- Haz un *fork* del repositorio y crea una rama nueva para tu contribución.
- Asegúrate de seguir el estilo de código del proyecto e incluir pruebas.
- Ejecuta ``make lint`` y ``make test`` antes de enviar un *pull request*.
- Actualiza la documentación y los ejemplos al agregar nuevas funcionalidades.
- Consulta ``CONTRIBUTING.rst`` para más detalles sobre el flujo de trabajo.

Licencia
--------

Este proyecto está licenciado bajo MIT. Consulta el archivo ``LICENSE`` para más detalles.

Autores
-------

`Juan‑Felipe Gutiérrez‑Gómez <jufgutierrezgo@unal.unal.edu.co>`_ es el creador y mantenedor principal. La lista completa de colaboradores se encuentra en ``AUTHORS.rst``.

Citación
--------

Si utilizas este paquete en trabajos académicos, por favor cita nuestro artículo sobre simulación de OCC con obturador rodante (en preparación) y referencia este repositorio.



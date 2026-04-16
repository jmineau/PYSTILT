.. _tutorial_column:

Tutorial: Column / Satellite Receptors (X-STILT-style)
=======================================================

This tutorial demonstrates how to use PYSTILT for atmospheric column
simulations, the Python equivalent of what
`X-STILT <https://github.com/uataq/X-STILT>`_ does in R
(`Wu et al. 2018 <https://doi.org/10.5194/gmd-11-4843-2018>`_).

Column footprints are required when working with satellite-retrieved column
concentrations (e.g., OCO-2/3 XCO₂, TROPOMI CH₄ / CO, GOSAT) or with
aircraft profiles. Instead of releasing particles from a single height, we
release them from many levels spanning the atmospheric column and weight the
resulting footprints by the satellite's averaging kernel and pressure weighting
function.

**What you'll learn:**

* How to define a slant receptor from observation LOS geometry.
* How to apply vertical weighting with declarative footprint transforms.
* How to interpret column footprints.


Defining the Column Receptor
-----------------------------

For a nadir-pointing instrument we use a vertical column (same lon/lat at each
level). For an off-nadir geometry we define a line of sight from viewing angles
and altitude sampling, then convert that LOS into a transport-level multipoint
receptor.

.. code-block:: python

   import pandas as pd
   import stilt
   from stilt.observations import (
       LineOfSight,
       Observation,
       ViewingGeometry,
       build_slant_receptor,
   )

   # OCO-2-like overpass: sounding over Salt Lake City on 15 July 2023
   overpass_time = pd.Timestamp("2023-07-15 21:06", tz="UTC")

   observation = Observation(
       sensor="oco2",
       species="xco2",
       time=overpass_time,
       latitude=40.77,
       longitude=-111.85,
       viewing=ViewingGeometry(
           viewing_zenith_angle=0.0,
           viewing_azimuth_angle=0.0,
       ),
       line_of_sight=LineOfSight(
           start_altitude=0.0,
           end_altitude=10000.0,
           count=50,
           altitude_ref="agl",
       ),
   )
   receptor = build_slant_receptor(observation)

   print(f"Column receptor: {len(receptor)} levels")

Here the LOS anchor is inferred from ``Observation.altitude`` when provided.
For MSL LOS definitions, PYSTILT also uses an MSL observation altitude as the
default lower clipping bound, so below-surface samples do not need to be
manually filtered in common cases.


Averaging Kernel Weighting
---------------------------

Satellite column retrievals are sensitive to different parts of the atmosphere
according to an averaging kernel (AK) and a pressure weighting function (PWF).
We apply these as a built-in transform on the footprint product:

.. code-block:: python

   from stilt.config import VerticalOperatorTransformSpec

   # Example AK and PWF arrays (length must match receptor count)
   n_levels = 50
   ak  = np.ones(n_levels)                        # simplified: uniform AK
   pwf = np.exp(-np.linspace(0, 5, n_levels))    # simplified: exponential PWF
   pwf /= pwf.sum()                               # normalise
   column_transform = VerticalOperatorTransformSpec(
       kind="vertical_operator",
       mode="ak_pwf",
       levels=np.linspace(0.0, 10000.0, n_levels).tolist(),
       values=(ak * pwf).tolist(),
       coordinate="xhgt",
   )


Project Setup and Run
----------------------

.. code-block:: python

   config = stilt.ModelConfig(
       n_hours=-72,
       numpar=200,
       hnf_plume=True,

       mets={
           "hrrr": stilt.MetConfig(
               directory="/data/met/hrrr",
               file_format="hrrr_%Y%m%d.arl",
               file_tres="1h",
           ),
       },

       footprints={
           "column": stilt.FootprintConfig(
               grid=stilt.Grid(
                   xmin=-115.0, xmax=-108.0,
                   ymin=38.0,   ymax=43.0,
                   xres=0.01,   yres=0.01,
               ),
               transforms=[column_transform],
           ),
       },
   )

   model = stilt.Model(
       project="./column_project",
       receptors=[receptor],
       config=config,
   )
   model.run(mets=["hrrr"], footprints=["column"])


Interpreting Column Footprints
--------------------------------

A column footprint has units of ppm / (µmol m⁻² s⁻¹) and represents the
sensitivity of the column-averaged dry-air mole fraction (XCO₂, XCH₄, etc.)
to surface fluxes. Multiplying by a flux inventory gives the expected
concentration enhancement in the column.

.. code-block:: python

   import matplotlib.pyplot as plt

   sim = list(model.simulations.values())[0]
   foot = sim.get_footprint("column")
   integrated = foot.integrate_over_time(*foot.time_range)

   integrated.data.plot(cmap="YlOrRd")
   plt.title(f"Column footprint — XCH₄ — {overpass_time:%Y-%m-%d %H:%M} UTC")
   plt.show()


.. seealso::

   :doc:`../advanced/column_receptors`: column receptor API reference.

   :doc:`../advanced/custom_hooks`: runtime particle-transform reference.

   `Wu et al. (2018) <https://doi.org/10.5194/gmd-11-4843-2018>`_:
   X-STILT paper with details on averaging kernel weighting and error analysis.

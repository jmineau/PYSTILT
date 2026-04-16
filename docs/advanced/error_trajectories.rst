.. _error_trajectories:

Transport Error Trajectories
=============================

PYSTILT can quantify uncertainty in footprints arising from errors in the
modelled wind field and boundary-layer height. This follows the error
propagation methodology of
`Lin and Gerbig (2005) <https://doi.org/10.1029/2004GL021127>`_ and is the
Python equivalent of X-STILT's ``run_hor_err`` / ``run_ver_err`` workflow.

The approach runs *two* sets of HYSPLIT trajectories per receptor:

1. **Nominal trajectories** — standard simulation.
2. **Error trajectories** — nominal run with random perturbations added to the
   horizontal wind (σ_u, σ_v) and the mixed-layer height (σ_zi).

The difference between the two footprints provides an estimate of transport
uncertainty.


Configuring Error Parameters
------------------------------

Error parameters are set in :class:`~stilt.config.ErrorParams` (inherited by
:class:`~stilt.ModelConfig`):

.. code-block:: python

   config = stilt.ModelConfig(
       n_hours=-24,
       numpar=100,
       mets={"hrrr": hrrr_met},

       # Horizontal wind error statistics
       siguverr=0.5,    # standard deviation of u/v wind error [m s⁻¹]
       tluverr=1.0,     # Lagrangian time scale of wind error [hours]
       zcoruverr=0.0,   # vertical correlation length scale [m]
       horcoruverr=5.0, # horizontal correlation length scale [km]

      # Mixed-layer height error statistics
      sigzierr=
   )

.. note::

   Error statistics (σ_uv, τ_L) are ideally derived from comparisons between
   the meteorological model and radiosonde observations. The NOAA radiosonde
   archive is available at https://ruc.noaa.gov/raobs/.


Running Error Trajectories
---------------------------

Error trajectories are run as a second pass automatically when error parameters
are supplied. Access them via :attr:`~stilt.Simulation.error_trajectories`:

.. code-block:: python

   model.run_trajectories(mets=["hrrr"])   # runs both passes

   for sim_id, sim in model.simulations.items():
       nominal_trajs = sim.trajectories
       error_trajs = sim.error_trajectories   # Trajectories | None


Computing Footprint Uncertainty
---------------------------------

Set ``error=True`` in :class:`~stilt.config.FootprintConfig` to compute error
footprints alongside nominal footprints in every batch call:

.. code-block:: python

   config = stilt.ModelConfig(
       ...
       footprints={
           "default": stilt.FootprintConfig(
               grid=grid,
               error=True,   # also generate a "default_error" footprint
           ),
       },
   )

Each batch call to :meth:`~stilt.Model.generate_footprints` (or ``stilt run``)
will then produce both ``"default"`` and ``"default_error"`` footprints per
simulation:

.. code-block:: python

   model.generate_footprints("default")

   nominal_foot = model.footprints[sim_id]["default"]
   error_foot   = model.footprints[sim_id]["default_error"]

   # Footprint uncertainty [ppm / (µmol m⁻² s⁻¹)]
   uncertainty = abs(nominal_foot.data - error_foot.data)

You can also generate error footprints one at a time at the simulation level:

.. code-block:: python

   sim = list(model.simulations.values())[0]
   error_foot = sim.generate_footprint("default", error=True)

.. seealso::

   `Lin & Gerbig (2005) <https://doi.org/10.1029/2004GL021127>`_:
   original methodology paper for propagating wind errors into concentration
   uncertainty.

   `Wu et al. (2018) <https://doi.org/10.5194/gmd-11-4843-2018>`_:
   X-STILT paper extending the error analysis to column retrievals.

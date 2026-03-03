from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from stilt.simulation import Trajectory


def extract_flux(fluxes, particles):
    """
    Extracts flux values from a flux dataset for given particle locations.

    Parameters
    ----------
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.
    particles : pandas.DataFrame
        DataFrame containing particle trajectory data with 'lati' and 'long' columns.

    Returns
    -------
    np.ndarray
        Numpy array of flux values corresponding to particle locations.
    """
    # TODO include time dimension if needed
    lat_indexer = xr.DataArray(particles.lati.values, dims="point")
    lon_indexer = xr.DataArray(particles.long.values, dims="point")
    selected_flux = fluxes.sel(lat=lat_indexer, lon=lon_indexer, method="nearest")
    return selected_flux.values


def calculate_particle_concentrations(
    trajectory_path: Path, fluxes: xr.DataArray
) -> pd.DataFrame:
    """
    Calculates the concentration of dCH4 for particles in a given trajectory file.

    Parameters
    ----------
    trajectory_path : Path
        Path to the trajectory parquet file.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    pd.DataFrame
        Concentration of dCH4 for the particles.
    """
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    # Load trajectory data
    particles = Trajectory.from_path(trajectory_path).data

    # Extract flux and calculate dCH4
    particles["flux"] = extract_flux(fluxes=fluxes, particles=particles)
    particles["dCH4"] = particles["foot"] * particles["flux"]

    # Sum dCH4 for each particle
    particle_sums = particles.groupby("indx")["dCH4"].sum()

    return particle_sums


def calculate_particle_variance(trajectory_path: Path, fluxes: xr.DataArray) -> float:
    """
    Calculates the variance of dCH4 for particles in a given trajectory file.

    Parameters
    ----------
    trajectory_path : Path
        Path to the trajectory parquet file.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    float
        Variance of dCH4 for the particles.
    """
    # Calculate particle concentrations
    particles = calculate_particle_concentrations(trajectory_path, fluxes=fluxes)

    # Calculate variance of concentrations between particles
    return float(particles.var())


def plot_particle_variances(sim_dir: Path | str, fluxes: xr.DataArray) -> plt.Axes:
    """
    Plots the variances of dCH4 for regular and error particles.

    Parameters
    ----------
    sim_dir : Path | str
        Directory containing the simulation data.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    plt.Axes
        Matplotlib Axes object with the plot.
    """
    import seaborn as sns

    # Get simulation directory and ID
    sim_dir = Path(sim_dir)
    sim_id = sim_dir.name

    regular_path = sim_dir / f"{sim_id}_trajec.parquet"
    error_path = sim_dir / f"{sim_id}_error.parquet"

    # Calculate particle concentrations
    regular = calculate_particle_concentrations(regular_path, fluxes=fluxes)
    error = calculate_particle_concentrations(error_path, fluxes=fluxes)

    # Create DataFrame for plotting
    df = pd.DataFrame({"Regular Particles": regular, "Error Particles": error}).melt(
        var_name="Particle Type", value_name="CH4 Concentration"
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        df, ax=ax, x="CH4 Concentration", hue="Particle Type", bins=100, kde=True
    )

    return ax


def calculate_transport_error(sim_dir: Path | str, fluxes: xr.DataArray) -> float:
    """
    Calculates the transport error for a given STILT simulation ID.

    Parameters
    ----------
    sim_dir : Path | str
        Directory containing the simulation data.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    float
        The calculated transport error.
    """
    # Get simulation directory and ID
    sim_dir = Path(sim_dir)
    sim_id = sim_dir.name

    regular_path = sim_dir / f"{sim_id}_trajec.parquet"
    error_path = sim_dir / f"{sim_id}_error.parquet"

    # Calculate variances
    regular_var = calculate_particle_variance(regular_path, fluxes=fluxes)
    error_var = calculate_particle_variance(error_path, fluxes=fluxes)

    # Calculate transport error
    return error_var - regular_var

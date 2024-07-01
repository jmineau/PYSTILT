"""
lair.uataq.stilt
~~~~~~~~~~~~~~~~

Module for preparing data for STILT runs & processing STILT output
"""

import datetime as dt
from functools import cached_property
import numpy as np
import os
import pandas as pd
import subprocess
import xarray as xr

from lair.config import STILT_DIR
from lair.uataq import site_config
from lair.utils.grid import add_latlon_ticks
from lair.utils.plotter import log10formatter
from lair.utils.records import Cacher

# TODO I think the classes should be just Footprint and Receptor?
#   and then STILT is a wrapper around those two classes?



TIME_FORMAT = '%Y%m%d%H%M'


def init_project(project, repo='https://github.com/jmineau/stilt', branch='main'):
    '''
    Initialize STILT project
    
    Python implementation of Rscript -e "uataq::stilt_init('project')"

    Parameters
    ----------
    project : str
        Name/path of STILT project.
    repo : str, optional
        URL of STILT project repo. The default is jmineau/stilt.
    branch : str, optional
        Branch of STILT project repo. The default is main.
    '''
    
    # Extract project name and working directory
    project_name = os.path.basename(project)
    wd = os.path.dirname(project)
    if wd == '':
        wd = os.getcwd()
        
    if os.path.exists(project):
        raise FileExistsError(f'{project} already exists')

    # Clone git repository
    cmd = f'git clone -b {branch} --single-branch --depth=1 {repo} {project}'
    subprocess.check_call(cmd, shell=True)

    # Run setup executable
    os.chdir(project)
    os.chmod('setup', 0o755)
    subprocess.check_call('./setup')

    # Render run_stilt.r template with project name and working directory
    with open('r/run_stilt.r') as f:
        run_stilt = f.read()
    run_stilt = run_stilt.replace('{{project}}', project_name)
    run_stilt = run_stilt.replace('{{wd}}', wd)
    with open('r/run_stilt.r', 'w') as f:
        f.write(run_stilt)
    os.chdir(wd)
    

def extract_simulation_id(simulation):
    '''
    Extract simulation id from simulation name

    Parameters
    ----------
    simulation : str
        Name of simulation.

    Returns
    -------
    sim_id : dict
        Dictionary with keys: time, lati, long, zagl.

    '''
    
    simulation = os.path.basename(simulation)

    if simulation.count('_') > 3:
        simulation = '_'.join(simulation.split('_')[:4])

    # Extract simulation id
    run_time, long, lati, zagl = simulation.split('_')
    sim_id = {'run_time': run_time,
              'long': float(long),
              'lati': float(lati),
              'zagl': float(zagl)}

    return sim_id


def fix_sim_links(old_out_dir, new_out_dir):
    '''
    Fix sim symlinks if stilt wd was changed

    Parameters
    ----------
    old_out_dir : str
        old out directory where symlinks currently point to and shouldnt.
    new_out_dir : str
        new out directory where symlinks should point to.

    Returns
    -------
    None.

    '''

    for subdir in ['footprints', 'particles']:
        for file in os.listdir(os.path.join(new_out_dir, subdir)):
            filepath = os.path.join(new_out_dir, subdir, file)

            # Check if file is link
            if os.path.islink(filepath):
                old_link = os.readlink(filepath)

                # Check if old_link is to file in old_out_dir
                if old_link.startswith(old_out_dir):
                    # Get sim_id of simulation
                    sim_id = os.path.basename(os.path.dirname(old_link))

                    # Remove old_link
                    os.remove(filepath)

                    # Create new link to new_out_dir
                    new_link = os.path.join(new_out_dir, 'by-id', sim_id, file)
                    os.symlink(new_link, filepath)


class Receptors:
    # TODO what about multilocation receptors such as TRAX?
    def __init__(self, loc, times):

        self.data = self.generate(loc, times)

    def generate(self, loc, times: pd.Series):
        '''
        Generate receptor dataframe when given a location and list of datetimes
        '''

        # Supply loc as SID or (lati, long, zagl)
        if isinstance(loc, str):
            lati, long, zagl = Receptors.get_site_location(loc)
        else:
            lati, long, zagl = loc

        N = len(times)  # number of receptors

        receptors = pd.DataFrame({'run_time': times,
                                  'long': [long] * N,
                                  'lati': [lati] * N,
                                  'zagl': [zagl] * N})

        return receptors.sort_values('run_time')

    def update(self, receptors):
        '''
        Update data with more receptors from a different Receptors instance
        '''

        assert type(receptors) == Receptors

        self.data = pd.concat([self.data, receptors.data])

        return None

    def save(self, out_path):
        self.data.set_index('run_time').to_csv(out_path)

        return None

    @staticmethod
    def get_site_location(site):
        # FIXME what about mobile sites
        site_config = site_config.loc[site]

        lati = site_config.lati.astype(float)
        long = site_config.long.astype(float)
        zagl = site_config.zagl.astype(float)

        return lati, long, zagl

    @staticmethod
    def nearest_mesowest(loc):
        # TODO
        pass


class Footprints:

    def __init__(self, footprint_dir, subset=False, engine='rasterio',
                 cache=False, reload_cache=False):

        assert os.path.exists(footprint_dir)  # make sure footprint dir exists

        self.footprint_dir = footprint_dir
        self.files = None

        if cache:
            if cache is True:
                cache = os.path.join(footprint_dir, 'footprints_cache.pkl')
            self.read = Cacher(self.read, cache, reload=reload_cache,
                               verbose=True)
        self.cache = cache

        self.foots = self.read(footprint_dir, subset, engine)
        self.weighted = self.apply_weights()

        self.receptor_locs = {}

    def __repr__(self):
        # TODO I think you should be able to recreate instance from repr
        #   which probably isnt true here
        # I think this should be __str__
        from math import log10, floor

        def round_to_1(x):
            return round(x, -int(floor(log10(abs(x)))))

        start = str(self.foots.run_time.values[0])[:10]
        end = str(self.foots.run_time.values[-1])[:10]
        time_range = f'{start} ~ {end}'

        resolution = round_to_1(self.foots.rio.resolution()[0])
        digits = len(str(resolution).split('.')[1])
        bounds = [round(x, digits) for x in self.foots.rio.bounds()]

        return (f'Footprints(time range="{time_range}", '
                f'resolution="{resolution}", bounds="{bounds}", '
                f'dir="{self.footprint_dir}", cache="{self.cache}")')

    def read(self, footprint_dir, subset, engine):

        def _preprocess(ds):
            filename = ds.encoding['source']
            sim_id = extract_simulation_id(filename)
            run_time = dt.datetime.strptime(sim_id['run_time'], TIME_FORMAT)

            if subset:
                ds = self._sub(ds, subset)

            ds = ds.sum(dim='time')  # temporal sum
            
            # Create run_time dimension to concat footprints
            ds = ds.expand_dims({'run_time': [run_time]})

            return ds

        print(f'Reading footprints from {footprint_dir}')

        self.files = Footprints.get_files(footprint_dir)

        processed_foots = [_preprocess(xr.open_dataset(file, engine=engine))
                           for file in self.files]
        foots = xr.concat(processed_foots, dim='run_time').foot
        foots = foots.sortby('run_time')

        return foots

    def _sub(self, ds, subset):
        if ds.rio.y_dim == 'y':
            y_slicer = slice(subset[3], subset[1])
        elif ds.rio.y_dim[:3].lower() == 'lat':
            y_slicer = slice(subset[1], subset[3])

        return ds.sel({ds.rio.x_dim: slice(subset[0], subset[2]),
                       ds.rio.y_dim: y_slicer})

    def sub(self, subset):
        return self._sub(self.foots, subset)

    def get_receptors_locs(self):
        assert len(self.files) >= 1

        sim_ids = [extract_simulation_id(file) for file in self.files]

        # TODO I feel like this could be down better
        # Get sim_locs of all the files without time - list of dicts
        sim_locs = [{'lati': sim_id['lati'],
                     }
                    for var in ['lati', 'long', 'zagl']
                    for sim_id in sim_ids]

        sim_locs = [{k: v for k, v in ID.items() if k != 'time'}
                    for ID in sim_ids]

        # remove duplicate locs
        sim_locs = [dict(t) for t in {tuple(loc.items()) for loc in sim_locs}]

        return sim_locs

    @cached_property
    def area(self):
        from lair.utils.grid import area_DataArray

        return area_DataArray(self.foots)

    def apply_weights(self):
        
        weights = np.cos(np.deg2rad(self.foots[self.foots.rio.y_dim]))
        weighted = self.foots.weighted(weights)
        return weighted

    def validate_transport(self):
        from synoptic.services import stations_timeseries

        # TODO
        #   first lets just use the station closest to the receptor
        #   later, we can try to combine all the stations within the STILT domain

        pass

    @staticmethod
    def get_files(footprint_dir):
        '''Get footprint files in footprint dir'''
        footprint_files = [os.path.join(footprint_dir, file)
                           for file in os.listdir(footprint_dir)
                           if file.endswith('foot.nc')]

        return sorted(footprint_files)

    @staticmethod
    def plot(foot, ax=None, crs=None, label_deci=1, tiler=None, tiler_zoom=9,
             bounds=None, x_buff=0.1, y_buff=0.1, labelsize=None,
             more_lon_ticks=0, more_lat_ticks=0):
        import cartopy.crs as ccrs
        from functools import partial
        from matplotlib.ticker import FuncFormatter as FFmt
        import matplotlib.pyplot as plt

        if tiler is not None:
            crs = tiler.crs
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': crs})
            ax.add_image(tiler, tiler_zoom)
        elif crs is None:
            crs = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': crs})

        if bounds is None:
            bounds = foot.rio.bounds()
        extent = [bounds[0]-x_buff, bounds[2]+x_buff,
                  bounds[1]-y_buff, bounds[3]+y_buff]

        label = r'footprint [ppm$ \left/ \frac{\mu mol}{m^2 s}\right.$]'
        log10formatter_part = partial(log10formatter, deci=label_deci)

        np.log10(foot).plot(ax=ax, transform=ccrs.PlateCarree(), alpha=0.5,
                            cmap='cool',
                            cbar_kwargs={'orientation': 'vertical',
                                         'pad': 0.03, 'label': label,
                                         'format': FFmt(log10formatter_part)})

        add_latlon_ticks(ax, extent, x_rotation=30, labelsize=labelsize,
                         more_lon_ticks=more_lon_ticks,
                         more_lat_ticks=more_lat_ticks)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set(ylabel=None,
               xlabel=None)

        return ax


class STILT:
    def __init__(self, project, directory=None, symlink_dir=None):
        self.project = project
        directory = directory or STILT_DIR or os.getcwd()
        self.stilt_wd = os.path.join(directory, self.project)

        if not os.path.exists(self.stilt_wd):
            print('Initializing STILT project...')
            init_project(self.stilt_wd)
            
            if symlink_dir is not None:
                os.symlink(self.stilt_wd, os.path.join(symlink_dir, 'STILT'))

    def __repr__(self):
        return f'STILT(project="{self.project}")'

    def generate_receptors(self, loc, times):
        self.receptors = Receptors(loc, times)

        return self.receptors
    
    def catalog_output(self, output_wd=None):
        'Catalog STILT output'
        
        
    def get_sims(self, id_dir=None):
        # id_dir is relative to stilt_wd/out
        id_dir = os.path.join(self.stilt_wd, 'out', id_dir or 'by-id')
        
        sims = [extract_simulation_id(sim) for sim in os.listdir(id_dir)]
        
        # Convert to dataframe
        sims = pd.DataFrame(sims)
        # sims['run_time'] = sims['run_time'].astype(str)
        # for column in ['lati', 'long', 'zagl']:
        #     sims[column] = sims[column].astype(float)
        sims['run_time'] = pd.to_datetime(sims['run_time'], format=TIME_FORMAT)
        sims.sort_values('run_time', inplace=True)
        return sims

    def get_missing_sims(self, receptors: pd.DataFrame, id_dir=None):
        '''Get missing simulations from STILT output directory'''

        # Get existing simulations
        existing_sims = self.get_sims(id_dir)

        # Merge the dataframes and add an indicator column
        merged = receptors.merge(existing_sims, how='outer', indicator=True)

        # Get the rows that are in receptors but not in existing_sims
        missing_sims = merged[merged['_merge'] == 'left_only']

        return missing_sims
    
    def read_footprints(self, footprint_dir=None, subset=False, engine='rasterio',
                        cache=False, reload_cache=False):
        
        footprint_dir = os.path.join(self.stilt_wd, 'out', footprint_dir or 'footprints')
        
        footprints = Footprints(footprint_dir, subset, engine, cache, reload_cache)
        
        return footprints

    def get_foots(self, footprint_dir=None, subset=False, engine='rasterio',
                   cache=False, reload_cache=False):
        
        footprints = self.read_footprints(footprint_dir, subset, engine,
                                          cache, reload_cache)

        return footprints.foots


def Lin2021(dCH4, footprints, weight=False, filter_sims=False):
    '''Following this logic, we estimate the Basin-averaged CH4 emissions
    Φ by dividing the CH4 enhancement measured at HPL by the total
    footprint integrating over all gridcells i within the Uinta Basin
    (defined as between 39.9N to 40.5N and 110.6W to 109W), where Φ at
    daily timescales was determined by dividing the ∆CH4 averaged over the
    afternoon (13:00–16:00 MST) by the total footprint averaged over the
    same hours'''

    if weight:
        # AREA WEIGHTED

        # Multiply by cell area, divide by total area, sum
        # f_weighted = (STILT.footprints.foots * STILT.footprints.area)\
        #     / STILT.footprints.area.sum()
        # f_sum = f_weighted.sum(dim=['x', 'y'])

        # Simple cosine weighting
        f_sum = footprints.weighted.sum(dim=['x', 'y'])

    else:
        # NOT AREA WEIGHTED
        f_sum = footprints.foots.sum(dim=['x', 'y'])

    f_sum_daily = f_sum.resample(run_time='1D').mean().to_pandas()

    if filter_sims:
        # Filter simulations when footprint is too weak
        f_thres = f_sum_daily.quantile(0.1)  # 10% quantile
        f_sum_daily[f_sum_daily < f_thres] = np.nan

        # Filter simulations when transport errors are large
        dir_thres = 45  # deg
        ws_thres = 1  # m s-1

    emis = (dCH4 / f_sum_daily).to_frame('CH4_emis')
    emis.index.name = 'Time_UTC'

    return emis

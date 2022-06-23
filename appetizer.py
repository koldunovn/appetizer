import gribscan
import numcodecs
import xarray as xr
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
mpl.use('Agg')
import gc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import pandas as pd
import os.path
import cmocean.cm as cmo
import sys
import yaml
import logging
from pprint import pformat
import time

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def get_cmap(cmap=None):
    """Get the color map.
    Parameters
    ----------
    cmap: str, mpl.colors.Colormap
        The colormap can be provided as the name (should be in matplotlib or cmocean colormaps),
        or as matplotlib colormap object.
    Returns
    -------
    colormap:mpl.colors.Colormap
        Matplotlib colormap object.
    """
    if cmap:
        if isinstance(cmap, (mpl.colors.Colormap)):
            colormap = cmap
        elif cmap in cmo.cmapnames:
            colormap = cmo.cmap_d[cmap]
        elif cmap in plt.colormaps():
            colormap = plt.get_cmap(cmap)
        else:
            raise ValueError(
                "Get unrecognised name for the colormap `{}`. Colormaps should be from standard matplotlib set of from cmocean package.".format(
                    cmap
                )
            )
    else:
        colormap = plt.get_cmap("Spectral_r")

    return colormap


def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z

def create_indexes_and_distances(model_lon, model_lat, lons, lats, k=1, workers=2):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.
    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.
    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.
    """
    xs, ys, zs = lon_lat_to_cartesian(model_lon, model_lat)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k, workers=workers)

    return distances, inds

def proj_selection(projection):
    if projection == "pc":
        projection_ccrs = ccrs.PlateCarree()
    if projection == "mer":
        projection_ccrs = ccrs.Mercator()
    elif projection == "np":
        projection_ccrs = ccrs.NorthPolarStereo()
    elif projection == "sp":
        projection_ccrs = ccrs.SouthPolarStereo()
    return projection_ccrs

def region_cartopy(box, res, projection="pc"):
    """ Computes coordinates for the region 
    Parameters
    ----------
    box : list
        List of left, right, bottom, top boundaries of the region in -180 180 degree format
    res: list
        List of two variables, defining number of points along x and y
    projection : str
        Options are:
            "pc" : cartopy PlateCarree
            "mer": cartopy Mercator
            "np" : cartopy NorthPolarStereo
            "sp" : cartopy SouthPolarStereo
    Returns
    -------
    x : numpy.array
        1 d array of coordinate values along x
    y : numpy.array
        1 d array of coordinate values along y
    lon : numpy.array
        2 d array of longitudes
    lat : numpy array
        2 d array of latitudes
    """
    projection_ccrs = proj_selection(projection)

    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 500, 500
    left, right, down, up = box
    logging.info('Box %s, %s, %s, %s', left, right, down, up)
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw=dict(projection=projection_ccrs),
        constrained_layout=True,
        figsize=(10, 10),
    )
    ax.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    # res = scl_fac * 300. # last number is the grid resolution in meters (NEEDS TO BE CHANGED)
    # nx = int((xmax-xmin)/res)+1; ny = int((ymax-ymin)/res)+1
    x = np.linspace(xmin, xmax, lonNumber)
    y = np.linspace(ymin, ymax, latNumber)
    x2d, y2d = np.meshgrid(x, y)

    npstere = ccrs.PlateCarree()
    transformed2 = npstere.transform_points(projection_ccrs, x2d, y2d)
    lon = transformed2[:, :, 0]  # .ravel()
    lat = transformed2[:, :, 1]  # .ravel()
    fig.clear()
    plt.close(fig)
   
    return x, y, lon, lat
    
config_path = sys.argv[1]
logging.info('Input file: %s', config_path)

with open(config_path, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

logging.debug(pformat(config))

# json file was already prepared with gribscan-index and gribscan-build command line tools
experiment_id = config['experiment_id']
datapath = config['datapath']
datazarr = config['datazarr'] # all 2D surface fields, this is the whole dataset!
outpath = config['outpath']

sstart_in = config['sstart_in']
sstop_in = config['sstop_in']

variable_name = config['variable_name']

left = config['left']
right = config['right']
bottom = config['bottom']
top = config['top']

projection = config['projection']
coastlines = config['coastlines']
res = config['res']

cmap=get_cmap(config['cmap'])
vmin=config['vmin']
vmax=config['vmax']

textxy = config['textxy']
textsize = config['textsize']
text_color = config['text_color']
dpi = config['dpi']

path_to_dist = f'{outpath}/distances_{experiment_id}_{projection}_{left}_{right}_{bottom}_{top}_{res[0]}_{res[1]}.npy'
path_to_inds = f'{outpath}/inds_{experiment_id}_{projection}_{left}_{right}_{bottom}_{top}_{res[0]}_{res[1]}.npy'

if sstart_in==sstop_in:
    startstop = None
else:
    startstop = [int(sstart_in), int(sstop_in)]
    

ds = xr.open_zarr("reference::"+datazarr, consolidated=False)

logging.info('Reading coordinates')
model_lon = ds.lon.values
model_lat = ds.lat.values

logging.info('Remove nans')
nonan = ~(np.isnan(model_lon) | np.isnan(model_lat)) 

lat_nonan = model_lat[nonan]
lon_nonan = model_lon[nonan]

logging.info('Getting the region')
x, y, x2, y2 = region_cartopy([left, right, bottom, top], res, projection=projection)

logging.info('Shape along x: %s', x.shape[0])
logging.info('Shape along y: %s', y.shape[0])


data = ds[variable_name]

if os.path.isfile(path_to_inds):
    distances = np.load(path_to_dist)
    inds = np.load(path_to_inds)
    logging.info('loading dist and inds')
else:
    logging.info('Inds and dist do not exist, creating')
    distances, inds = create_indexes_and_distances(lon_nonan, lat_nonan, x2, y2, k=1, workers=10)
    np.save(path_to_dist, distances)
    np.save(path_to_inds, inds)
    logging.info('saving dist and inds')

if startstop is None:
    sstart = 0
    sstop = data.shape[0]
else:
    sstart = startstop[0]
    sstop = startstop[1]

logging.info('Start/stop time indexes: %s, %s', sstart, sstop)

if not os.path.exists(f'{outpath}/{variable_name}/images/'):
    os.makedirs(f'{outpath}/{variable_name}/images/')

if not os.path.exists(f'{outpath}/{variable_name}/netcdf/'):
    os.makedirs(f'{outpath}/{variable_name}/netcdf/')

logging.info('Start main loop')    
for ttime in range(sstart, sstop):
    start_time = time.time()
    
    data_nan = data[ttime].values
    data_nonan = data_nan[nonan]
    interpolated_data_fesom = data_nonan[inds]
    interpolated_data_fesom.shape = x2.shape
    strtime = data[ttime].time.values.astype('str')[:16]
    
    if coastlines:       
        projection_ccrs = proj_selection(projection)
        fig, ax = plt.subplots(
                    1,
                    1,
                    subplot_kw=dict(projection=projection_ccrs),
                    constrained_layout=True,
                    figsize=(res[0]/dpi, res[1]/dpi),
                )

        ax.imshow(np.flipud(interpolated_data_fesom), cmap=cmap, vmin=vmin, vmax=vmax, 
                  extent=(x.min(), x.max(), y.min(), y.max()),  transform=projection_ccrs)
        ax.coastlines(resolution='50m')
        print(textxy[0])
        print(textxy[1])
        ax.text(textxy[0], textxy[1], f'{strtime} [{ttime}]', {'size':textsize, 'color':text_color}, transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(
            1,
            1,
            constrained_layout=True,
            figsize=(res[0]/dpi, res[1]/dpi),
        )

        ax.imshow(np.flipud(interpolated_data_fesom), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.text(textxy[0], textxy[1], f'{strtime} [{ttime}]', {'size':textsize, 'color':text_color})
    
    
    ax.axis('off');
    ostrtime = strtime.replace(':','_')
    plt.savefig(f'{outpath}/{variable_name}/images/{experiment_id}_{projection}_{variable_name}_{ostrtime}_{str(ttime).zfill(5)}.png', dpi=dpi)
    logging.info('Image created')
    fig.clear()
    plt.close(fig)
#     plt.close()
#     del interpolated_u_fesom
    out1 = xr.Dataset(
            {variable_name: (["time", "lat", "lon"], np.expand_dims(interpolated_data_fesom, 0))},
            coords={
                "time": np.atleast_1d(data[ttime].time),
                # "depth": realdepths,
                "lon": (["lon"], x),
                "lat": (["lat"], y),
                "longitude": (["lat", "lon"], x2),
                "latitude": (["lat", "lon"], y2),
            },
            attrs={'experiment_id':experiment_id, 'interpolation':'nearest neighbour'},
    )
    out1.to_netcdf(f'{outpath}/{variable_name}/netcdf/{experiment_id}_{projection}_{variable_name}_{ostrtime}_{str(ttime).zfill(5)}.nc', 
                  encoding={
            "time": {"dtype": np.dtype("double")},
            "lat": {"dtype": np.dtype("double")},
            "lon": {"dtype": np.dtype("double")},
            "longitude": {"dtype": np.dtype("double")},
            "latitude": {"dtype": np.dtype("double")},
            variable_name: {"zlib": True, "complevel": 1, "dtype": np.dtype("single")},
        },)
    logging.info('NetCDF file created')
    del interpolated_data_fesom
    gc.collect()
    elapsed_time = time.time() - start_time
    logging.info('Finish processing time step %s, %s, in %s',str(ttime).zfill(5), ostrtime, time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    

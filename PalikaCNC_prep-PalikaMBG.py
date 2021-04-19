"""
SOURCE : 
[1]:    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
[2]:    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/14675742#14675742

"""

import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
import numpy as np
import sys, json
import geopandas as gpd
gdal.PushErrorHandler('CPLQuietErrorHandler')
from math import log10, floor

import os
import numpy as np
from scipy.spatial import ConvexHull

def create_polygon(MBG_bbox):   
    
    latitudes = MBG_bbox[:,0]
    longitudes = MBG_bbox[:,1]
    
    coords = zip(latitudes, longitudes)
    
    ring = ogr.Geometry(ogr.wkbLinearRing)
    
    for coord in coords:
        ring.AddPoint(coord[0], coord[1])
        

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()

def write_shapefile(df, out_shp, extra_cols=[]):
    """
    https://gis.stackexchange.com/a/52708/8104
    """
    
    
    
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    # Remove output shapefile if it already exists
    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)
        
    ds = driver.CreateDataSource(out_shp)
    
    

    layer = ds.CreateLayer('MBG', None, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('Z_MIN', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('Z_MAX', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('Z_RANGE', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('Z_MEAN', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('MBG_W_m', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('MBG_L_m', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('MBG_Orient', ogr.OFTReal))
    
    
    for xcols in extra_cols:
        # Add one attribute
        layer.CreateField(ogr.FieldDefn(xcols, ogr.OFTString))
        
 
    for index, row in df.iterrows():
        defn = layer.GetLayerDefn()

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('Z_MIN', row['Z_MIN'])
        feat.SetField('Z_MAX', row['Z_MAX'])
        feat.SetField('Z_RANGE', row['Z_RANGE'])
        feat.SetField('Z_MEAN', row['Z_MEAN'])
        feat.SetField('MBG_W_m', row['MBG_W_m'])
        feat.SetField('MBG_L_m', row['MBG_L_m'])
        feat.SetField('MBG_Orient', row['MBG_Orient'])
        #for xcols in extra_cols:
        #    feat.SetField(xcols, row[xcols])
            

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkt(row['MBG_poly'])
        feat.SetGeometry(geom)
        
        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

def truncate(number, digits, step_up = True) -> float:
    stepper = 10.0 ** digits    
    if(step_up):
        return math.trunc(math.ceil(stepper * number)) / stepper
    else:
        return math.trunc(stepper * number) / stepper
      
        
def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing



def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

   
    
    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
    
     # find angle of longest side of rectangle    
    angle = (pi2 - angles[best_idx]) if ( (x1-x2) >= (y1 - y2) ) else (np.pi - angles[best_idx]) 
    edge  = edges[best_idx]
    

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)
        
    return {"bbox":rval, "max_x" : x1, "min_x":x2, "max_y":y1, "min_y":y2, "r":r, "angle": angle,"rval":rval , "edge": edge}

def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


def zonal_stats(vector_path, raster_path, cols = [], nodata_value=None, global_src_extent=False):
    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert(rds)
    rb = rds.GetRasterBand(1)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(vector_path, GA_ReadOnly)  # TODO maybe open update if we want to write stats
    assert(vds)
    vlyr = vds.GetLayer(0)

    # create an in-memory numpy array of the source raster data
    # covering the whole extent of the vector layer
    if global_src_extent:
        # use global source extent
        # useful only when disk IO or raster scanning inefficiencies are your limiting factor
        # advantage: reads raster data in one pass
        # disadvantage: large vector extents may have big memory requirements
        src_offset = bbox_to_pixel_offsets(rgt, vlyr.GetExtent())
        src_array = rb.ReadAsArray(*src_offset)

        # calculate new geotransform of the layer subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5]
        )

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    while feat is not None:

        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)

            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5]
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource('out')
        mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())
        
        

        # Rasterize it
        rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(
                src_array == nodata_value,
                np.logical_not(rv_array)
            )
        )
                
        env =feat.geometry().GetEnvelope()
        
        ring_dict = json.loads(feat.ExportToJson())
        points = np.array(ring_dict['geometry']['coordinates'][0])
        MBG_dict = minimum_bounding_rectangle(points)
        MBG_bbox= MBG_dict['bbox']
        
        ### For Plotting
        """
        plt.scatter(points[:,0], points[:,1])
        plt.fill(MBG_bbox[:,0], MBG_bbox[:,1], alpha=0.2)
        plt.axis('equal')
        plt.show()
        """
        mbg_poly = create_polygon(MBG_bbox)
        
     
        
        
        # print (points[0])
        
        # print ("minX: %d, minY: %d, maxX: %d, maxY: %d" %(env[0],env[2],env[1],env[3]))

        
        MBG_Width_m = float(MBG_dict['max_y']-MBG_dict['min_y']) if float(MBG_dict['max_y']-MBG_dict['min_y']) < float(MBG_dict['max_x']-MBG_dict['min_x']) else float(MBG_dict['max_x']-MBG_dict['min_x']) ,
        MBG_Length_m = float(MBG_dict['max_x']-MBG_dict['min_x']) if float(MBG_dict['max_x']-MBG_dict['min_x'])> float(MBG_dict['max_y']-MBG_dict['min_y']) else  float(MBG_dict['max_y']-MBG_dict['min_y']) ,  
         
        feature_stats = {
            'Z_MIN': float(masked.min()),
            'Z_MAX': float(masked.max()),
            'Z_RANGE': float(masked.max())-float(masked.min()),
            'Z_MEAN': float(masked.mean()),
            'Z_STD': float(masked.std()),
            'Z_SUM': float(masked.sum()),
            'Z_COUNT': int(masked.count()),
            
            'MBG_poly':mbg_poly,
            'MBG_W_m':MBG_Width_m[0],
            'MBG_L_m':MBG_Length_m[0],
            'MBG_Orient':math.degrees(MBG_dict['angle']),
            'Map scale 1:':truncate(float(MBG_Length_m[0]), -2, True),
            'Model Z depth (m)':(float(masked.max())-float(masked.min()))/truncate(float(MBG_Length_m[0]), -2, True),
            'Model_width to fit 1 m': MBG_Width_m[0]/ truncate(float(MBG_Length_m[0]), -2, True),          
            'Model_length to fit 1 m':MBG_Length_m[0]/ truncate(float(MBG_Length_m[0]), -2, True),   
            'Width @ 1:10000':MBG_Width_m[0]/10000, 
            'Length @ 1:10000':MBG_Length_m[0]/10000, 
            'Height @ 1:10000':(float(masked.max())-float(masked.min()))/10000,
            'No. 25 mm slices':((float(masked.max())-float(masked.min()))/10000)/0.025,
            'No. 18 mm slices':((float(masked.max())-float(masked.min()))/10000)/0.018,
                        
             #'fid': int(feat.GetFID()),
            'BBOX_Width' : float(env[1]-env[0]) if float(env[1]-env[0]) <  float(env[3]-env[2]) else float(env[3]-env[2]),
            'BBOX_Length' : float(env[3]-env[2]) if float(env[3]-env[2]) > float(env[1]-env[0]) else float(env[1]-env[0])
        }
        
        for col in cols:
            feature_stats[col] = feat[col]
        stats.append(feature_stats)
        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    vds = None
    rds = None
    return stats

from shapely import speedups
speedups.disable()

dem_fp = r"C:\work\CNC\app\data\DEM\20m_aw3d1.tif"  


try:
    import pandas as pd
    from pandas import DataFrame
    
    palikas_fp=r"C:\work\CNC\app\data\ECHO_GIS_Data\ECHO_Project_palikas.shp"
    cols_palikas = ['FIRST_DIST','FIRST_GaPa']
    stats_palikas = zonal_stats(palikas_fp, dem_fp, cols = cols_palikas, nodata_value=-32768)
    df_palikas = DataFrame(stats_palikas)
    df_palikas.to_excel(r"C:\work\CNC\app\PalikasMBG.xlsx")
    write_shapefile(df_palikas,r"C:\work\CNC\app\PalikasMBG.shp")
    
    wards_fp = r"C:\work\CNC\app\data\ECHO_GIS_Data\ECHO_Project_wards.shp"
    cols_wards = ['DISTRICT','GaPa_NaPa','NEW_WARD_N']
    stats_wards = zonal_stats(wards_fp, dem_fp, cols = cols_wards, nodata_value=-32768)
    df_wards = DataFrame(stats_wards)
    df_wards.to_excel(r"C:\work\CNC\app\WardsMBG.xlsx")
    write_shapefile(df_wards,r"C:\work\CNC\app\WardsMBG.shp")
    

except ImportError:
    import json
    print (json.dumps(stats, indent=2))

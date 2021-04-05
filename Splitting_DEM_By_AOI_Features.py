import glob
import fiona
from subprocess import Popen

with fiona.open(r"C:\Users\Ram_DFID\Downloads\Boundaries-20210405T082136Z-001\Boundaries\ECHO_GIS_Data\ECHO_GIS_Data\ECHO_Project_wards.shp", 'r') as dst_in:
    for index, feature in enumerate(dst_in):
        ward = str(feature['properties']["NEW_WARD_N"])
        gapanapa = feature['properties']["GaPa_NaPa"]
        name = gapanapa+"-"+ward
        with fiona.open(r'C:\work\CNC\DEM\separated\{}_{}.shp'.format(gapanapa,ward), 'w', **dst_in.meta) as dst_out:
            dst_out.write(feature)

polygons = glob.glob(r'C:\work\CNC\DEM\separated\*.shp')  ## Retrieve all the .shp files
import os
for polygon in polygons:
    feat = fiona.open(polygon, 'r')
    name =os.path.basename(polygon).split('.')[0]

    
    command = 'gdalwarp -dstnodata -9999 -cutline "{}" ' \
              '-crop_to_cutline -of GTiff "C:\\work\\CNC\DEM\\20m_aw3d1.tif" "C:\\work\\CNC\\DEM\\outputraster\\{}_20m_aw3d1.tiff"'.format(polygon, name)
    
    print (command)
    Popen(command, shell=True)

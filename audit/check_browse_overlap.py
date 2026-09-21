"""Compare native TMC geotags + subsampled browse support with OHRC grids.
Browse nonzero is a diagnostic support proxy, not a science validity mask.
"""
import json
import math
from pathlib import Path
import numpy as np
from osgeo import gdal, ogr
base=Path(__file__).resolve().parent
gdal.UseExceptions()
meta=json.loads((base/'metadata_analysis.json').read_text())
geo=json.loads((base/'geometry_analysis.json').read_text())
tags=json.loads((base/'tmc_geotags.json').read_text())['gdal']
ds=gdal.Open(str(next((base/'snapshot').glob('ch2_tmc*/browse/**/*.png'))))
a=ds.ReadAsArray()
gt=tags['geoTransform']
sx=tags['size'][0]/ds.RasterXSize;sy=tags['size'][1]/ds.RasterYSize
dx=gt[1]*sx;dy=gt[5]*sy
result={'method':'Assume archive subsampled browse spans native raster extent; positive browse pixels are support proxy only',
        'browse_lon_step':dx,'browse_lat_step':dy,'pairs':{}}
for k in ['OHRC_2021','OHRC_2024']:
    poly=ogr.CreateGeometryFromWkt(geo[k]['grid']['boundary_wkt'])
    xmin,xmax,ymin,ymax=poly.GetEnvelope()
    positive=0;candidate=0;gaps=[];rows=[]
    y0=max(0,int((ymax-gt[3])/dy));y1=min(a.shape[0],int((ymin-gt[3])/dy)+1)
    for y in range(y0,y1):
        lat=gt[3]+(y+.5)*dy
        line=ogr.Geometry(ogr.wkbLineString);line.AddPoint_2D(xmin-1,lat);line.AddPoint_2D(xmax+1,lat)
        cut=poly.Intersection(line)
        if cut.IsEmpty():continue
        lo,hi,_,_=cut.GetEnvelope()
        xs=gt[0]+(np.arange(a.shape[1])+.5)*dx
        in_poly=(xs>=lo)&(xs<=hi)
        candidate+=int(in_poly.sum());positive+=int(((a[y]>0)&in_poly).sum())
        nz=np.flatnonzero(a[y]>0)
        if len(nz):
            left=gt[0]+nz[0]*dx
            gap=(left-hi)*math.pi*1737400/180*math.cos(math.radians(lat))
            gaps.append(gap)
            rows.append({'browse_row':y,'latitude':lat,'left_positive_longitude':left,'ohrc_right_longitude':hi,'gap_m':gap})
    result['pairs'][k]={'candidate_browse_centres_inside_ohrc':candidate,'positive_browse_centres_inside_ohrc':positive,
                       'eastward_gap_m_min_median_max': [float(np.min(gaps)),float(np.median(gaps)),float(np.max(gaps))],
                       'sample_rows':[rows[0],rows[len(rows)//2],rows[-1]]}
c=meta['products']['IIRS']['corners']['System_Level_Coordinates']
ring=ogr.Geometry(ogr.wkbLinearRing)
for key in ['upper_left','upper_right','lower_right','lower_left','upper_left']:
    ring.AddPoint_2D(float(c[key+'_longitude']['value']),float(c[key+'_latitude']['value']))
p=ogr.Geometry(ogr.wkbPolygon);p.AddGeometry(ring)
xs=gt[0]+(np.arange(a.shape[1])+.5)*dx
area=total=0.0
for y,row in enumerate(a):
    lat=gt[3]+(y+.5)*dy
    line=ogr.Geometry(ogr.wkbLineString);line.AddPoint_2D(22,lat);line.AddPoint_2D(26,lat)
    g=p.Intersection(line)
    cell=1737400**2*math.radians(dx)*abs(math.sin(math.radians(gt[3]+y*dy))-math.sin(math.radians(gt[3]+(y+1)*dy)))/1e6
    total+=int(np.count_nonzero(row))*cell
    if g.IsEmpty():continue
    lo,hi,*_=g.GetEnvelope()
    area+=int(np.count_nonzero((row>0)&(xs>=lo)&(xs<=hi)))*cell
result['IIRS_TMC_support_proxy']={'intersection_km2':area,'tmc_nonzero_support_km2':total,
    'tmc_support_percent':area/total*100,
    'iirs_corner_area_percent':area/meta['products']['IIRS']['footprint_areas_km2']['System_Level_Coordinates']*100}
(base/'browse_overlap_proxy.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

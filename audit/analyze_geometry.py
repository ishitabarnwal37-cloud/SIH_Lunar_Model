"""Parse fixed-width OAT records and read OHRC geolocation grids."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from osgeo import ogr
from calculate_metadata import polygon, spherical_area

BASE = Path(__file__).resolve().parent
meta = json.loads((BASE / 'metadata_analysis.json').read_text())
widths = [8,6,4,28]+[20]*6+[12]*3+[56]*3+[14]*6+[12]*2+[1]+[9]*3+[10,5]+[9]*2+[16]*3+[41]
assert sum(widths) == 628
offsets = np.cumsum([0]+widths)

def parse(s):
    f = [s[a:b].strip() for a,b in zip(offsets[:-1], offsets[1:])]
    date = [int(s[18+i*4:22+i*4]) for i in range(7)]
    return {'time': datetime(*date[:6], date[6]*1000, tzinfo=timezone.utc).isoformat(),
            'record': int(f[1]), 'record_length': int(f[2]),
            'subsat_lat': float(f[16]), 'subsat_lon': float(f[17]),
            'sun_azimuth': float(f[18]), 'sun_elevation': float(f[19]),
            'payload_lat': float(f[20]), 'payload_lon': float(f[21]),
            'altitude_km': float(f[22]), 'emission_field': float(f[25]),
            'phase_field': float(f[26]), 'yaw_nadir_angle': float(f[27]),
            'solar_zenith': float(f[30]), 'yaw': float(f[32]),
            'roll': float(f[33]), 'pitch': float(f[34])}

def stats(rows):
    if not rows:
        return {'records': 0}
    out={'records':len(rows), 'time_start':rows[0]['time'], 'time_stop':rows[-1]['time']}
    for k in rows[0]:
        if k not in ['time','record','record_length']:
            x=np.array([r[k] for r in rows])
            out[k]={'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())}
    return out

result={}
for key,p in meta['products'].items():
    path=next((BASE/'snapshot'/p['id']).rglob('*.oat'))
    raw=path.read_bytes()
    assert len(raw)%628==0
    rows=[parse(raw[i:i+628].decode('ascii')) for i in range(0,len(raw),628)]
    rows=[r for r in rows if datetime.fromisoformat(p['start']) <= datetime.fromisoformat(r['time']) <= datetime.fromisoformat(p['stop'])]
    result[key]={'acquisition':stats(rows)}
    if key in ['IIRS','TMC']:
        result[key]['ohrc2024_latitude_window_centerline_proxy']=stats([r for r in rows if -0.444178<=r['payload_lat']<=0.372032])
    else:
        result[key]['ohrc_pair_latitude_window_centerline_proxy']=stats([r for r in rows if 0.2351974<=r['payload_lat']<=0.372032])
    if key.startswith('OHRC'):
        gpath=next((BASE/'snapshot'/p['id']).rglob('*.csv'))
        a=np.genfromtxt(gpath,delimiter=',',names=True)
        xs=np.unique(a['Pixel']);ys=np.unique(a['Scan'])
        rows_grid={(int(r['Pixel']),int(r['Scan'])):(r['Longitude'],r['Latitude']) for r in a}
        edge=[rows_grid[(int(x),int(ys[0]))] for x in xs]
        edge += [rows_grid[(int(xs[-1]),int(y))] for y in ys[1:]]
        edge += [rows_grid[(int(x),int(ys[-1]))] for x in xs[-2::-1]]
        edge += [rows_grid[(int(xs[0]),int(y))] for y in ys[-2::-1]]
        ring=ogr.Geometry(ogr.wkbLinearRing)
        for lon,lat in edge:ring.AddPoint_2D(lon,lat)
        poly=ogr.Geometry(ogr.wkbPolygon);poly.AddGeometry(ring)
        grid={'records':len(a),'grid_shape':[len(ys),len(xs)],'pixel_range':[float(xs[0]),float(xs[-1])],
              'scan_range':[float(ys[0]),float(ys[-1])], 'boundary_wkt':poly.ExportToWkt(), 'area_km2':spherical_area(poly)}
        def dist(c,d):
            lon,lat=c;lon2,lat2=d
            dlat=math.radians(lat2-lat);dlon=math.radians(lon2-lon)
            h=math.sin(dlat/2)**2+math.cos(math.radians(lat))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
            return 2*1737400*math.asin(math.sqrt(h))
        cross=[];along=[]
        for y in ys[::max(1,len(ys)//20)]:
            for x1,x2 in zip(xs[:-1],xs[1:]):
                cross.append(dist(rows_grid[(int(x1),int(y))],rows_grid[(int(x2),int(y))])/(x2-x1))
        for x in xs[::max(1,len(xs)//10)]:
            for y1,y2 in zip(ys[:-1],ys[1:]):
                along.append(dist(rows_grid[(int(x),int(y1))],rows_grid[(int(x),int(y2))])/(y2-y1))
        grid['sampled_cross_track_m_per_pixel']=[float(np.min(cross)),float(np.median(cross)),float(np.max(cross))]
        grid['sampled_along_track_m_per_pixel']=[float(np.min(along)),float(np.median(along)),float(np.max(along))]
        result[key]['grid']=grid

(BASE/'geometry_analysis.json').write_text(json.dumps(result,indent=2))
for k,r in result.items():
    print(k,json.dumps({s:{kk:vv for kk,vv in v.items() if kk!='boundary_wkt'} for s,v in r.items()}))

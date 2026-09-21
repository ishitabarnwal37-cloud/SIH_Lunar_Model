"""Read-only label audit; calculated areas assume straight edges in lon/lat.
No science raster is opened or modified by this script.
"""
import json
import math
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET
from osgeo import ogr

BASE = Path(__file__).resolve().parent
R = 1737400.0  # Explicit spherical Moon assumption, metres.
ogr.UseExceptions()

def tag(e):
    return e.tag.rsplit('}', 1)[-1]

def child(e, name):
    return next((x for x in e if tag(x) == name), None)

def value(e, name):
    x = next((x for x in e.iter() if tag(x) == name), None)
    return x.text.strip() if x is not None and x.text else None

def leaves(e):
    return {tag(x): {'value': (x.text or '').strip(), **x.attrib}
            for x in e.iter() if len(x) == 0}

def polygon(corners):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for name in ['upper_left', 'upper_right', 'lower_right', 'lower_left', 'upper_left']:
        ring.AddPoint_2D(float(corners[name + '_longitude']['value']),
                         float(corners[name + '_latitude']['value']))
    p = ogr.Geometry(ogr.wkbPolygon)
    p.AddGeometry(ring)
    assert p.IsValid()
    return p

def spherical_area(g):
    # Densify straight lon/lat boundaries before cylindrical equal-area mapping.
    p = g.Clone()
    p.Segmentize(0.005)
    def transform(h):
        if h.GetGeometryCount():
            for i in range(h.GetGeometryCount()):
                transform(h.GetGeometryRef(i))
        else:
            for i in range(h.GetPointCount()):
                lon, lat, _ = h.GetPoint(i)
                h.SetPoint_2D(i, R * math.radians(lon), R * math.sin(math.radians(lat)))
    transform(p)
    return p.GetArea() / 1e6

products = {}
for path in sorted((BASE / 'snapshot').glob('*/data/**/*.xml')):
    root = ET.parse(path).getroot()
    name = path.parts[-5]
    key = 'OHRC_2021' if '20210402' in name else 'OHRC_2024' if '20240330' in name else 'IIRS' if '_iir_' in name else 'TMC'
    params = next(x for x in root.iter() if tag(x) == 'Product_Parameters')
    geom = next(x for x in root.iter() if tag(x) == 'Geometry_Parameters')
    array = next(x for x in root.iter() if tag(x).startswith('Array_') and tag(x) != 'Array_Description')
    axes = {value(x, 'axis_name').lower(): int(value(x, 'elements')) for x in array if tag(x) == 'Axis_Array'}
    file = next(x for x in root.iter() if tag(x) == 'File')
    binary = path.parent / value(file, 'file_name')
    pointer = binary.read_text() if binary.exists() and binary.stat().st_size < 200 else None
    p = {'id': name, 'label': str(path.relative_to(BASE / 'snapshot')),
         'mission': value(root, 'title'), 'start': value(root, 'start_date_time'),
         'stop': value(root, 'stop_date_time'), 'level': value(root, 'processing_level'),
         'parameters': leaves(params), 'corners': {tag(x): leaves(x) for x in geom},
         'axes': axes, 'datatype': value(array, 'data_type'),
         'label_file_bytes': int(value(file, 'file_size')), 'md5': value(file, 'md5_checksum'),
         'lfs_pointer': pointer, 'bands': axes.get('band', 1)}
    p['footprint_areas_km2'] = {k: spherical_area(polygon(v)) for k, v in p['corners'].items()}
    if key == 'IIRS':
        p['selected_wavelengths_nm'] = {value(x, 'band_number'): value(x, 'center_wavelength')
            for x in array.iter() if child(x, 'band_number') is not None and value(x, 'band_number') in ['1','6','15','16','137','256']}
        p['expected_bytes_from_axes'] = math.prod(axes.values()) * 2
    products[key] = p

def chosen(p):
    for k in ['Refined_Corner_Coordinates', 'System_Level_Coordinates']:
        if k in p['corners']:
            return k, polygon(p['corners'][k])
    raise ValueError('Missing footprint')

pairs = []
for (a, pa), (b, pb) in combinations(products.items(), 2):
    ka, ga = chosen(pa)
    kb, gb = chosen(pb)
    inter = ga.Intersection(gb)
    area = spherical_area(inter) if not inter.IsEmpty() else 0
    item = {'a': a, 'b': b, 'footprint_basis': [ka, kb], 'area_km2': area,
            'fraction_a_pct': 100 * area / spherical_area(ga),
            'fraction_b_pct': 100 * area / spherical_area(gb),
            'intersection_wkt': inter.ExportToWkt(),
            'intersection_bounds_lonlat': inter.GetEnvelope() if area else None}
    # Compare alternate TMC boundary definitions. Corrected rectangle is an
    # outer raster extent, not proof that its pixels image terrain everywhere.
    if 'TMC' in [a, b]:
        other = ga if b == 'TMC' else gb
        item['tmc_boundary_sensitivity'] = {}
        for k, c in products['TMC']['corners'].items():
            g = polygon(c)
            i = other.Intersection(g)
            ar = spherical_area(i) if not i.IsEmpty() else 0
            item['tmc_boundary_sensitivity'][k] = {'area_km2': ar,
                'other_pct': ar / spherical_area(other) * 100,
                'tmc_pct': ar / spherical_area(g) * 100}
    for field in ['sun_elevation', 'sun_azimuth', 'solar_incidence']:
        va = float(pa['parameters'][field]['value'])
        vb = float(pb['parameters'][field]['value'])
        delta = abs(va-vb)
        item['delta_' + field] = min(delta, 360-delta) if field == 'sun_azimuth' else delta
    pairs.append(item)

out = {'method': 'Straight lon/lat corner edges clipped with OGR; spherical equal-area integration; not a valid-pixel overlap measurement',
       'moon_radius_assumed_m': R, 'products': products, 'pairs': pairs}
(BASE / 'metadata_analysis.json').write_text(json.dumps(out, indent=2))
for k, p in products.items():
    print(k, p['axes'], p['datatype'], 'bytes', p['label_file_bytes'], 'area', p['footprint_areas_km2'])
for p in pairs:
    print(json.dumps(p))

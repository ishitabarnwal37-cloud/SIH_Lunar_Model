"""Retrieve 600 bytes of TIFF geotags; create a metadata-only sparse probe."""
import json
import urllib.request
import struct
from pathlib import Path
base=Path(__file__).resolve().parent
tree=json.loads((base/'source_tree.json').read_text())
path=next(x['path'] for x in tree['tree'] if '/data/' in x['path'] and x['path'].endswith('.tif'))
url='https://media.githubusercontent.com/media/ishitabarnwal37-cloud/SIH_Lunar_Model/'+tree['sha']+'/'+path
start=2927900;length=600
req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}'})
with urllib.request.urlopen(req,timeout=30) as r:
    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
        raise RuntimeError('Server did not honor byte range')
    data=r.read(length)
result={'range_start':start,'bytes_read':len(data),
        'last_strip_offset':struct.unpack_from('<Q',data,2927916-start)[0],
        'pixel_scale':struct.unpack_from('<3d',data,2928198-start),
        'tiepoint':struct.unpack_from('<6d',data,2928222-start)}
p=base/'tmc_metadata_only_sparse.tif'
with p.open('wb') as f:
    f.write((base/'tmc_header_probe.bin').read_bytes());f.seek(start);f.write(data)
from osgeo import gdal
gdal.UseExceptions();gdal.SetConfigOption('GDAL_PAM_ENABLED','NO')
result['gdal']=gdal.Info(str(p),format='json')
(base/'tmc_geotags.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

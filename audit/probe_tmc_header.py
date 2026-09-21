"""Read at most 64 KiB of the public TMC LFS object, never the whole raster."""
import json
import urllib.request
from pathlib import Path
base=Path(__file__).resolve().parent
tree=json.loads((base/'source_tree.json').read_text())
path=next(x['path'] for x in tree['tree'] if '/data/' in x['path'] and x['path'].endswith('.tif'))
url='https://media.githubusercontent.com/media/ishitabarnwal37-cloud/SIH_Lunar_Model/'+tree['sha']+'/'+path
req=urllib.request.Request(url,headers={'Range':'bytes=0-65535'})
with urllib.request.urlopen(req,timeout=30) as r:
    data=r.read(65536)
    result={'status':r.status,'content_range':r.headers.get('Content-Range'),
            'content_length':r.headers.get('Content-Length'),'bytes_read':len(data),
            'first_16_hex':data[:16].hex()}
(base/'tmc_header_probe.bin').write_bytes(data)
(base/'tmc_header_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

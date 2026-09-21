"""Download only the four archived browse PNGs for diagnostic inspection."""
import json
import urllib.request
from pathlib import Path

base=Path(__file__).resolve().parent
tree=json.loads((base/'source_tree.json').read_text())
for x in tree['tree']:
    if '/browse/' not in x['path'] or not x['path'].endswith('.png'):
        continue
    if x['size'] > 10_000_000:
        raise RuntimeError('Browse exceeds 10 MB limit')
    p=base/'snapshot'/x['path']
    if p.exists() and p.stat().st_size == x['size']:
        continue
    url='https://raw.githubusercontent.com/ishitabarnwal37-cloud/SIH_Lunar_Model/'+tree['sha']+'/'+x['path']
    with urllib.request.urlopen(url, timeout=30) as r:
        data=r.read(x['size']+1)
    if len(data)!=x['size']:
        raise RuntimeError('Browse size mismatch')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(data)
    print(p.name, len(data), flush=True)

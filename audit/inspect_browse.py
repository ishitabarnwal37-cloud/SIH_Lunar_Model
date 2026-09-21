"""Browse-only diagnostics; no science-pixel statistics are implied."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base=Path(__file__).resolve().parent
out=base/'previews';out.mkdir(exist_ok=True)
paths=sorted((base/'snapshot').glob('*/browse/**/*.png'))
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',14)
sheet=Image.new('RGB',(1280,1050),'#f4f4f4');draw=ImageDraw.Draw(sheet)
draw.text((20,10),'Archived browse images — diagnostic views, not registration results',font=font,fill='black')
summary=[]
for i,p in enumerate(paths):
    im=Image.open(p);a=np.array(im)
    key='IIRS' if '_iir_' in p.name else 'TMC' if '_tmc_' in p.name else 'OHRC 2021' if '20210402' in p.name else 'OHRC 2024'
    entry={'product':key,'path':str(p.relative_to(base)), 'dimensions':list(im.size),'mode':im.mode,
        'scope':'all browse pixels, including borders; not science pixels',
        'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),
        'median':float(np.median(a)),'p2':float(np.percentile(a,2)),
        'p98':float(np.percentile(a,98)), 'zero_pct':float(np.mean(a==0)*100),
        '255_pct':float(np.mean(a==255)*100), 'nodata_pct':'not available',
        'valid_pixel_pct':'not available'}
    summary.append(entry)
    x=i*320
    draw.text((x+12,50),key,font=font,fill='black')
    draw.text((x+12,76),f'{im.width} × {im.height} browse pixels',font=small,fill='black')
    overview=im.convert('RGB');overview.thumbnail((285,560))
    sheet.paste(overview,(x+(320-overview.width)//2,105))
    draw.text((x+12,680),'Central browse crop (unregistered)',font=small,fill='black')
    # Take centre crop with original pixel aspect ratio; do not stretch.
    crop_w=min(im.width,480);crop_h=min(im.height,480)
    cx,cy=im.width//2,im.height//2
    crop=im.crop((cx-crop_w//2,cy-crop_h//2,cx+(crop_w+1)//2,cy+(crop_h+1)//2)).convert('RGB')
    crop.thumbnail((285,285))
    sheet.paste(crop,(x+(320-crop.width)//2,715))
    draw.text((x+12,1010),f'Zero: {entry["zero_pct"]:.1f}% (not nodata proof)',font=small,fill='black')
sheet.save(out/'browse_contact_sheet.png')
(base/'browse_statistics.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))

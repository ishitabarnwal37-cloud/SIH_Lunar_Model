"""Make truthful four-column before/after figures from archived browse PNGs.

Uses the repository's normalize_to_uint8 function without importing unrelated
OpenCV/skimage dependencies. This is radiometric preview preparation only:
no registration, image stretching, shadow removal or invented NAC reference.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
OUT = BASE.parent / 'outputs' / 'preprocessing_preview'
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the inspected function verbatim. Loading its single AST definition
# avoids running the module demo or requiring its unused processing packages.
source = BASE / 'snapshot' / 'lunar_preprocessing.py'
tree = ast.parse(source.read_text())
definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'normalize_to_uint8')
namespace = {'np': np}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
normalize = namespace['normalize_to_uint8']

font_path = '/System/Library/Fonts/Helvetica.ttc'
title_font = ImageFont.truetype(font_path, 26)
font = ImageFont.truetype(font_path, 20)
small = ImageFont.truetype(font_path, 16)

products = {}
records = []
for path in sorted((BASE / 'snapshot').glob('*/browse/**/*.png')):
    name = ('IIRS' if '_iir_' in path.name else 'TMC' if '_tmc_' in path.name
            else 'OHRC_2021' if '20210402' in path.name else 'OHRC_2024')
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    original = Image.open(path).convert('L')
    a = np.asarray(original)
    # Only remove contiguous zero runs at horizontal row ends for this browse
    # display. Interior zero-valued shadows are retained. Not a science mask.
    nonzero = a != 0
    has_data = nonzero.any(axis=1)
    first = nonzero.argmax(axis=1)
    last = a.shape[1] - 1 - nonzero[:, ::-1].argmax(axis=1)
    columns = np.arange(a.shape[1])[None, :]
    support = has_data[:, None] & (columns >= first[:, None]) & (columns <= last[:, None])
    if not support.any():
        raise ValueError(f'No nonzero browse support for {name}')
    values = a[support]
    low, high = np.percentile(values, [1, 99])
    processed = np.zeros_like(a)
    processed[support] = normalize(values, lower_percentile=1, upper_percentile=99)
    im = Image.fromarray(processed)
    destination = OUT / f'{name}_preprocessed_browse.png'
    im.save(destination)
    assert im.size == original.size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    products[name] = (original, im)
    records.append({'product': name, 'source': str(path.resolve()),
                    'source_sha256': original_hash, 'output': destination.name,
                    'dimensions': list(original.size), 'p1': float(low), 'p99': float(high),
                    'recipe': 'repository normalize_to_uint8, percentiles 1/99 on row-support pixels',
                    'mask': 'exclude only contiguous zero runs at each row end; preserve interior zeros',
                    'scope': 'archived browse-resolution radiometric preview, not science data or registration',
                    'geometry_changed': False})

def center_text(draw, box, text, chosen_font):
    bounds = draw.textbbox((0, 0), text, font=chosen_font)
    draw.text(((box[0]+box[2]-(bounds[2]-bounds[0]))/2, box[1]), text,
              font=chosen_font, fill='#141414')

def figure(pair, filename, title):
    width, height = 1200, 1200
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    center_text(draw, (0, 18, width, 60), title, title_font)
    x0, colw, top, image_top, image_bottom = 60, 270, 80, 164, 1050
    draw.rectangle((x0, top, x0+4*colw, image_bottom), outline='#242424', width=2)
    draw.line((x0, 122, x0+4*colw, 122), fill='#242424', width=2)
    draw.line((x0, image_top, x0+4*colw, image_top), fill='#242424', width=2)
    draw.line((x0+2*colw, top, x0+2*colw, image_bottom), fill='#242424', width=2)
    for j in (1, 3):
        draw.line((x0+j*colw, 122, x0+j*colw, image_bottom), fill='#242424', width=1)
    center_text(draw, (x0, 90, x0+2*colw, 120), 'Original browse images', font)
    center_text(draw, (x0+2*colw, 90, x0+4*colw, 120), 'Preprocessed browse images', font)
    for i, key in enumerate(pair + pair):
        x = x0+i*colw
        center_text(draw, (x, 132, x+colw, 162), key.replace('_', ' '), font)
        im = products[key][0 if i < 2 else 1].convert('RGB')
        im.thumbnail((colw-20, image_bottom-image_top-20), Image.Resampling.LANCZOS)
        canvas.paste(im, (x+(colw-im.width)//2, image_top+10))
    center_text(draw, (0, 1072, width, 1095), 'Processing: 1st–99th percentile contrast stretch; image geometry preserved.', small)
    center_text(draw, (0, 1100, width, 1125), 'Full browse extents shown independently. Columns are not geometrically aligned.', small)
    center_text(draw, (0, 1128, width, 1155), 'No SIFT/ASIFT result is shown. Full-resolution science-data processing is pending.', small)
    canvas.save(OUT / filename)

figure(['OHRC_2021', 'OHRC_2024'], 'OHRC_preprocessing_comparison.png', 'OHRC preprocessing comparison')
figure(['TMC', 'IIRS'], 'TMC_IIRS_preprocessing_comparison.png', 'TMC / IIRS preprocessing comparison')
(OUT / 'processing_manifest.json').write_text(json.dumps(records, indent=2))
(OUT / 'README.md').write_text('''# Preprocessing preview outputs

These figures use the four authentic archived browse PNGs from SIH_Lunar_Model.
They reproduce the original-pair / preprocessed-pair layout requested by the user.
They do not contain LRO NAC imagery or registration/matching results.

Processing reuses the inspected repository function `normalize_to_uint8` with
1st/99th percentiles. Contiguous zero runs at row ends are excluded as a browse
display-border heuristic; interior zero pixels are kept. This mask must not be
used as an authoritative science-data validity mask.

Each derivative has exactly the input browse dimensions. Figure thumbnails
preserve aspect ratio; different columns are independently fitted to the page,
not resampled to a common GSD. This is a conservative radiometric baseline,
not a reproduction of an unspecified paper's filters or the repository's entire
OHRC_NAC CLAHE/inversion/dilation recipe. Source images are unchanged.

`processing_manifest.json` records provenance, percentiles, dimensions and hashes.
Full-resolution processing requires complete source IMG/TIF/QUB files; the current
repository science objects have the integrity problems documented in
`audit/DATASET_AUDIT.md`.
''')
print(json.dumps({'output_directory': str(OUT), 'products': records}, indent=2))

#!/usr/bin/env python3
"""
Generate a full HTML gallery for all images in test_results/visualized_images
"""
import os
from pathlib import Path

IMAGES_DIR = Path('test_results/visualized_images')
OUTPUT_HTML = Path('test_results/gallery_full.html')

html_head = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visualization Gallery</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f3f4f6;margin:0;padding:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.card{background:#fff;border-radius:6px;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,.08)}
.card img{width:100%;height:180px;object-fit:cover;display:block}
.card .meta{padding:8px;font-size:13px;color:#111}
.header{margin-bottom:16px}
.header a{color:#1f6feb;text-decoration:none}
</style>
</head>
<body>
<div class="header">
  <h1>Test Results - Visualized Images</h1>
  <p>Click any image to open the full-size file in a new tab.</p>
  <p>Images directory: <code>test_results/visualized_images/</code></p>
</div>
<div class="grid">
'''

html_tail = '''</div>
</body>
</html>'''

images = []
if IMAGES_DIR.exists():
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            images.append(p.name)
else:
    print(f"Images folder not found: {IMAGES_DIR}")

items = []
for name in images:
    src = f"visualized_images/{name}"
    item = f'''<div class="card"><a href="{src}" target="_blank"><img src="{src}" loading="lazy" alt="{name}"></a><div class="meta">{name}</div></div>'''
    items.append(item)

OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_head)
    f.write('\n'.join(items))
    f.write(html_tail)

print(f"Gallery written to: {OUTPUT_HTML}")
print(f"Images count: {len(images)}")

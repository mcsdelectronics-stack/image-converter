# Image Vector Converter

Low quality bitmap → clean, editable **vector** (Illustrator **EPS** / SVG).

A logo arrives as a 150 px JPEG with blocking, noise and speckles. This tool
repairs the bitmap, flattens it into a handful of colours, traces every colour
region into smooth cubic Bezier paths, and writes an EPS that opens in
Illustrator, CorelDRAW, Inkscape or Affinity Designer.

The output is real vector geometry: infinitely scalable, editable paths, no
raster image embedded.

---

## Quick start (Windows)

Double click **`run.bat`**. It creates the environment on the first run and
opens the app.

Then: **Open image → adjust the sliders → Convert → Save EPS**.

## Manual start

```bat
python -m pip install -r requirements.txt
python main.py
```

## Command line

```bat
python main.py -i low_quality_logo.jpg -o logo.eps
python main.py -i logo.png -o logo.svg -c 6 -d 0.4 --upscale 3
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-c, --colors` | 8 | number of flat colours (2–64) |
| `-d, --detail` | 0.6 | curve accuracy in source pixels, lower = more points |
| `--corner` | 75 | angle (degrees) above which a vertex stays a corner |
| `--min-area` | 6 | drop shapes smaller than this (source px²) |
| `--denoise` | 6 | JPEG denoise, 0–30 |
| `--upscale` | 1.5 | working-resolution upscale, 1–4 (a *minimum*, see below) |
| `--contrast` | 2 | local contrast boost, 0–10 |
| `--sharpen` | 40 | unsharp amount, 0–200 |
| `--despeckle` | 1 | speck and pinhole removal, 0–10 |
| `--keep-background` | off | keep the background instead of making it transparent |

---

## কীভাবে কাজ করে (বাংলা)

1. **Open image** চাপুন — PNG / JPG / WEBP / BMP / TIFF।
2. App আগে ছবিটা ঠিক করে (**Denoise → Upscale → Contrast → Sharpen → Despeckle**),
   তাই JPEG-এর ব্লকিং আর নয়েজ তাড়ায়।
3. **Colours** স্লাইডার দিয়ে ঠিক করুন — লোগোর জন্য ৪–১২ যথেষ্ট।
4. **Convert** — ছবিটা ভেক্টরে রূপান্তর হয়, preview দেখুন।
5. **Save EPS** — Illustrator-এ `File → Open` করলেই editable path পাবেন
   (Object → Ungroup দিলে আলাদা আলাদা path ভাঙবে)।

Sliders (সব source pixel-এ হিসাব করা, তাই `Upscale` বদলালেও result একই থাকে):

- **Detail accuracy** — ছোট মান = বেশি anchor point, মসৃণ curve, বড় ফাইল
- **Remove specks** — ছোট ছোট দাগ, ছিদ্র ও noise বাদ দেয়
- **Corner angle** — কমালে ধোকা corner ধরে রাখে
- **Background** — default: background সরিয়ে transparent EPS
- **Upscale** — এটা একটা *সর্বনিম্ন* মান। ছোট ছবি (যেমন ৬০px আইকন) নিজে থেকেই
  বেশি বড় করে trace হয়, কারণ এত কম পিক্সেলে নিজের curve বোঝা যায় না।

---

## Trace quality, measured

Tuning by eye does not scale, so the tracer is scored by `bench_quality.py` on
four degraded images: mean distance between the traced outline and the edge the
*original* photo really has (lower is better), colour error against the source,
and how many anchors the file needs.

| Change | mean edge error | logo anchors | logo EPS |
| --- | --- | --- | --- |
| starting point | 0.712 px | 1360 | 64 KB |
| palette trained on flat interiors, boundary pixels decided by their neighbourhood | 0.286 px | 1171 | 49 KB |
| working resolution 1.5× instead of 2× | 0.229 px | 685 | 29 KB |
| clustering on colour + local context | **0.221 px** | **563** | **24 KB** |

3.2× more accurate outlines from 2.4× fewer anchors — the file is smaller *and*
closer to the original. The four things that actually mattered:

- **Palette from flat interiors.** k-means trained only on pixels away from an
  edge, so JPEG speckle never earns a palette entry of its own.
- **Boundary pixels decided by their neighbourhood.** Anti-aliased edge pixels
  sit exactly between two colours; colour distance alone hands them to whichever
  side the noise favoured, and that is what makes a traced edge wobble.
- **Clustering on colour *and* local context.** A thin stroke's dark core and
  its lighter flank are the same drawing, but plain k-means spends a dozen
  palette entries on that ramp and shreds one line into a dozen slivers.
- **1.5× working resolution.** 2× magnifies compression artefacts along with the
  image; 1.5× traces closer to the truth *and* emits fewer anchors.

Plus, in the geometry stage: every corner is snapped onto the intersection of the
two straight edges meeting there (a corner found on a pixel outline is a pixel
*centre*, up to half a pixel off the real corner), and each fitted segment gets a
Gauss-Newton pass that slides its two control points along their own tangents —
accepted only when it genuinely reduces the error.

Reproduce the table with `python bench_quality.py`, or sweep individual settings
with `python sweep.py`.

---

## How the conversion works

| Stage | Module | What it does |
| --- | --- | --- |
| Repair | `ivc/enhance.py` | edge-preserving denoise, supersample, CLAHE contrast, unsharp, morphology |
| Quantise | `ivc/quantize.py` | spatially-aware k-means in CIE Lab, boundary settling, near-duplicate palette merge, optional background removal |
| Trace | `ivc/trace.py` | one mask per colour → OpenCV contours (outer shapes *and* holes) |
| Fit | `ivc/geometry.py` | Douglas-Peucker, corner detection + snapping, least-squares cubic fitting with Newton-Raphson reparameterisation, control-point refinement, recursive subdivision |
| Write | `ivc/document.py` | EPS (PostScript, even-odd fills) and SVG |
| Orchestrate | `ivc/engine.py` | the pipeline plus the vector preview renderer |
| UI | `ivc/gui.py` | Tkinter app, threaded conversion, live preview |

Design notes worth knowing:

- **Holes stay holes.** Every outline of a colour (outer shape and hole) goes
  into one path filled with the even-odd rule, so the counter of an "o" or the
  eye of a letter survives without any depth bookkeeping.
- **Corners are snapped, not approximated.** Each corner is moved onto the
  intersection of the two straight edges leading into and out of it, and the
  neighbouring arcs leave and arrive along those edges. Traced square corners
  land within 0.000 px of the true corner.
- **Smoothing happens before simplification.** Averaging a dense pixel outline
  barely moves it; averaging already-thinned points visibly shrinks every
  curve, so the order matters.
- **Coarse outlines are never smoothed.** A four-point rectangle would lose a
  quarter of a side length at each corner, so smoothing is skipped unless the
  outline is a fine pixel staircase.
- **A smooth closed outline still gets two cut points**, otherwise a single fit
  has to wrap onto its own start point, which is degenerate.
- **Pixel controls are converted into working-image units**, which is why the
  "Detail" and "Remove specks" sliders keep their meaning when you change
  "Upscale" — the file size does not explode either.
- **Empty palette entries are dropped.** A colour that loses every pixel once the
  boundaries settle would otherwise become an empty layer in the output.

## Tests

```bat
python test_geometry.py    # curve fitting: circles keep their radius, corners stay sharp
python test_fidelity.py    # the vector output reproduces the quantised bitmap
python test_eps.py         # EPS parses as PostScript, SVG parses as XML
python test_pipeline.py    # end-to-end on a deliberately bad JPEG
python test_gui.py         # GUI flow: convert, re-convert, cancel
python test_stress.py      # a 1200x900 image, grayscale, alpha, a 24x18 icon
python test_compare.py     # writes testdata/compare.png for eyeballing
python bench_quality.py    # edge accuracy / colour error / anchor count per image
python sweep.py            # what each slider actually buys, in numbers
```

`test_fidelity.py` is the interesting one: it renders the traced document and
compares it with the flat-colour bitmap the tracer was asked to reproduce, and
only counts mismatches that sit further than the fit tolerance away from a
colour boundary. Those would be real structural errors (wrong nesting, missing
shape, filled hole). It currently reports 0.00 %.

`bench_quality.py` is the one that keeps the tracer honest between releases: it
measures where the traced outline actually lands relative to the real edge in
the source, which no eyeball test can do.

## Requirements

Python 3.10+, `pillow`, `numpy`, `opencv-python-headless` (see
`requirements.txt`).

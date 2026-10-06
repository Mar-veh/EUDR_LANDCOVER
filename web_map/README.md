# Ghana EUDR Tree-Crop Map: web portal

**Live:** https://eudr-ghana.vercel.app

A web map of where Ghana's 2025 tree crops (the merged Tree Crop Plantation class from TabPFN-3.5) overlap land that was forest on 31 December 2020, the EUDR cut-off. The 2020 forest baseline is the EU's JRC Global Forest Cover 2020 V4. The overlap is split by the model's calibrated confidence.

## Run it

```
cd web_map
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m app
```

Then open http://127.0.0.1:8000.

The server reads `PRIVATE_KEY`, `SERVICE_EMAIL`, `PROJECT_ID`, `PREDICTED_CLASSCODE` and `PREDICTED_CONFIDENCE` from the repository's `.env`. The key never reaches the browser. Opening `public/index.html` straight from disk only shows a "map server not reachable" notice.

## Deploy on Vercel

1. In Vercel, import the GitHub repository and choose **Import single project** next to `web_map` (FastAPI). The Root Directory becomes `web_map`.
2. Under Environment Variables, add `PROJECT_ID`, `SERVICE_EMAIL`, `PRIVATE_KEY`, `PREDICTED_CLASSCODE` and `PREDICTED_CONFIDENCE`, copied from `.env`. The private key can be pasted as it appears there, quotes and `\n` included.
3. Deploy. Vercel serves `public/` from its CDN and runs the FastAPI app in `app/main.py` for `/api/...`.

Every visitor's map tiles and pixel clicks are computed by Earth Engine under the project in `PROJECT_ID`.

## What the portal does

A full-screen map with one small panel. The panel switches between two layers and shows a one-line explanation and a legend:

- **EUDR overlap (default).** Tree crop on 2020 forest, in red where the model is confident (0.6 or more) and in amber where it is less sure. It is drawn over a green tint of the 2020 forest.
- **Land cover.** The 17-class 2025 map from `new_class.xlsx` and `color_code.txt`, with no majority filter. The switch "Fade areas where the model is unsure" fades pixels as the model's confidence drops, which shows where the map can be trusted most. A transparency slider (0–100%) lets the map or satellite image show through the land cover.

Click the map to see what is at that spot: the EUDR status, the 2025 class (with the detailed class, such as cocoa), whether the spot is in the 2020 forest baseline, and the model's confidence. A Map/Satellite switch changes the background. Light and dark mode follow the system setting.

**Show me around.** First-time visitors are offered a nine-step guided tour of the controls. It can be replayed from the "Show me around" button in the top bar.

Map tiles are rendered live by Earth Engine from the assets. The class asset uses MODE pyramids. When zoomed out, a map pixel therefore shows the most common class in its area, so small overlaps appear as you zoom in.

The overlap is a screening indicator, not a legal determination.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/config` | default threshold and the forest dataset |
| `GET /api/layers/{eudr,forest2020,landcover,confidence}` | an Earth Engine XYZ tile URL; parameters `threshold`, `outside`, `fade`, `scope` |
| `GET /api/point?lon=&lat=&threshold=` | the pixel inspector values |

The page talks to the API only through `public/js/source.js`. Set `window.WM_API_BASE` there to point a separately hosted frontend at the API.

## National and district totals (not computed yet)

A national figure for the EUDR panel ("X ha of tree crop in Ghana is on 2020 forest") needs one reduction over all 2.4 billion pixels, which is too large for a live request. `scripts/district_stats.py export` runs it as an Earth Engine batch export to a table asset. `scripts/district_stats.py fetch` then writes `public/data/stats.json`, and the page picks the file up automatically. **This writes an asset to the Earth Engine project, so it only runs with the owner's approval.**

## Layout

```
app/                    FastAPI backend: Earth Engine login, layers, pixel inspector
public/                 the page (index.html, css/, js/, data/tables.js, vendor/leaflet/)
scripts/build_tables.py rebuilds app/data/ and public/data/tables.js from new_class.xlsx, color_code.txt and the CSVs
scripts/district_stats.py   national, regional and district totals (batch export, needs approval)
```

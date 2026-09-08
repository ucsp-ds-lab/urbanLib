# urbanLib

GPU-parallel urban analysis: crime hotspots, amenity coverage, and safest routes over a city's road network.

The three operators (`hotspot`, `amenity`, `saferoute`) use custom CUDA kernels (via `cudf`/`cupy`) to parallelize snapping points to the road network and route computation — this is a **GPU-only** library: you need an NVIDIA GPU with CUDA to actually use it.

## Requirements

- Python >= 3.9
- NVIDIA GPU + compatible CUDA toolkit (for `run`/`visualize` with real data)

## Installation

```bash
pip install "urbanlib[gpu] @ git+https://github.com/ucsp-ds-lab/urbanLib.git"
```

This installs the `urbanlib` CLI along with `cudf`/`cupy` (RAPIDS), resolved directly from PyPI. Without the `[gpu]` extra the install doesn't fail, but only `urbanlib fetch` stays available (no GPU needed) — `run` and `visualize` with real data need the GPU operators.

## Usage

```bash
# Downloads the road network from OSM and caches it under data/<city>/
# (fetched automatically if missing, this step is optional)
urbanlib fetch --city "Manhattan, New York, USA"

# Run one or more operators
urbanlib run --city "Manhattan, New York, USA" --operator hotspot --crimes crimes.csv
urbanlib run --city "Manhattan, New York, USA" --operator amenity --amenities amenities.csv

# --modes already defaults to "fast,balanced,safe" -- computes all 3 modes
# at once, no need to pass the flag
urbanlib run --city "Manhattan, New York, USA" --operator saferoute \
  --start 40.758,-73.985 --end 40.700,-74.016

# You can also pass your own alpha (0 = distance only, 1 = avoid crime only)
# instead of -- or mixed with -- the presets
urbanlib run --city "Manhattan, New York, USA" --operator saferoute \
  --start 40.758,-73.985 --end 40.700,-74.016 --modes 0.2

urbanlib run --city "Manhattan, New York, USA" --operator all --crimes crimes.csv --amenities amenities.csv

# The "routes" map compares all 3 saferoute modes side by side
urbanlib visualize --city "Manhattan, New York, USA" --operator hotspot,amenity,routes
```

`--crimes`/`--amenities` expect a CSV with `Latitude`,`Longitude` columns (`amenities.csv` also needs an `amenity` column with the category). Results land in `output/<city>/` (CSV, GeoJSON, HTML maps).

`run` doesn't save performance metrics (timings, RAM/VRAM usage) by default; add `--save-metrics` if you need them.

## Operators

| Operator | What it does |
|---|---|
| `hotspot` | Snaps each crime to the nearest road network node (point-segment snapping) and aggregates the count per node. |
| `amenity` | Same as `hotspot`, but for OSM points of interest, aggregated per node and category. |
| `saferoute` | Multi-objective Dijkstra routing, weighing distance against accumulated crime risk. `--modes fast,balanced,safe`, or any float alpha in `[0, 1]`. |

## Development

```bash
pip install -e ".[gpu,dev]"
pytest
```

Most tests run without a GPU (they use a synthetic road network). The ones that exercise the real operators (`test_hotspot.py`, `test_amenity.py`, `test_saferoute.py`, `test_citygraph.py`) require `cudf`/`cupy` to be installed and skip themselves automatically when unavailable.

## 📚 Citation

If you use UrbanLib in your research, please cite:

> VILCA-QUISPE, E.; GOMEZ-NIETO, E. **UrbanLib: A GPU-based library for accelerating urban data exploration**. In: Conference on Graphics, Patterns and Images, 39. (SIBGRAPI), 2026, Goiânia, GO. Proceedings... 2026. On-line. URI: <upn:EEAFFE:8JMKD2USNRW34M/4GBJQNB>. Available from: <http://urlib.net/upn:EEAFFE:8JMKD2USNRW34M/4GBJQNB>.

```bibtex
@inproceedings{vilcaquispe2026urbanlib,
  author    = {Vilca-Quispe, E. and Gomez-Nieto, E.},
  title     = {UrbanLib: A GPU-based library for accelerating urban data exploration},
  booktitle = {Proceedings of the 39th Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year      = {2026},
  address   = {Goi{\^a}nia, GO, Brazil},
  note      = {On-line},
  url       = {http://urlib.net/upn:EEAFFE:8JMKD2USNRW34M/4GBJQNB}
}
```

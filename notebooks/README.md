# notebooks/

Placeholder. This will hold the `.ipynb` versions of the pipeline scripts
that `run-notebook.yml` executes on schedule:

- `1_append_daily.ipynb`  (from `../1_append_daily.py`)
- `2_append_weekly.ipynb` (from `../2_append_weekly.py`)
- `3_screen.ipynb`        (from `../3_screen.py`)

Not converted yet -- the `.py` scripts in the parent folder are the current
source of truth. Convert with `jupytext --to notebook <script>.py` (or paste
into a fresh notebook) once the repo/workflow is ready to go live.

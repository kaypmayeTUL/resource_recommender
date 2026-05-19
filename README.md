# Acquisition Recommendation Scorer — Standalone App

Self-contained Streamlit app extracted from the larger Library Collection Dashboard. Score candidate book lists against your checkout history to prioritize purchases.

## Run locally

```bash
pip install -r requirements.txt
streamlit run recommender_app.py
```

## requirements.txt

```
streamlit>=1.28
pandas>=2.0
numpy>=1.24
plotly>=5.15
nltk>=3.8
```

NLTK data (`punkt`, `wordnet`, `omw-1.4`) is downloaded automatically on first run via `_ensure_nltk()`.

## Deploy to Streamlit Community Cloud

1. Push `recommender_app.py` and `requirements.txt` to a GitHub repo.
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing to the repo.
3. Set the main file to `recommender_app.py`.
4. Deploy. First load takes ~30 seconds while NLTK data downloads.

## Inputs

| File | Required | Columns |
|---|---|---|
| Checkouts | Yes | `title`, `checkouts` (required); `author`, `subjects`, `lc_classification` (recommended) |
| Recommendations | Yes | `title` (required); `author`, `subjects`, `lc_classification` (recommended) |
| Faculty research interests | No | `name`, `department`, `research_interests` |
| Custom synonym groups | No | `term`, `group_label` |

## Output

A scored, sortable recommendations list with per-component breakdowns (subject similarity, LC fit, author popularity, faculty match), downloadable as CSV with analysis notes.

## Relationship to the dashboard

This is a literal extract of the recommender from the larger `library_dashboard.py`. The two share zero runtime dependencies — bug fixes or feature additions must be ported manually in either direction.

If you want the full dashboard (Collection Profiler, COUNTER Analyzer, Zero-Use Identifier, and this recommender all in one app), use `library_dashboard.py` instead.

## Tulane styling

Hardcoded Tulane green (`#285C4D`) and blue (`#71C5E8`). Edit the `<style>` block near the top of the file to change colors.

## Contact

Kay P Maye (kmaye@tulane.edu) — Howard-Tilton Memorial Library, Tulane University

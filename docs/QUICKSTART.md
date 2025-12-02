# Quick Start Guide

This guide walks you through running your first dance comparison analysis.

## Step 1: Prepare Your Data

Your motion capture CSV files should have:
- Semicolon (`;`) delimiters
- 5 header rows (name, fps, date, columns, units)
- Position data in millimeters
- Column names like `LWristPositions X`, `RShoulderPositions Y`, etc.

## Step 2: Run the Pipeline

### Single Dance Analysis

```bash
python run_pipeline.py your_dance.csv
```

This produces:
- `your_dance_standardized.csv` - Cleaned data
- `your_dance_visualization.html` - 3D animation

### Two-Dance Comparison

```bash
python run_pipeline.py dance1.csv dance2.csv
```

Additional outputs:
- `comparison_dance1_vs_dance2.txt` - Similarity analysis
- `multi_dance_report.txt` - Summary report

### Multiple Dances

```bash
python run_pipeline.py dance1.csv dance2.csv dance3.csv dance4.csv
```

The pipeline will compute all pairwise comparisons and identify clusters.

## Step 3: View Results

### 3D Visualizations

Open any `*_visualization.html` file in your web browser:

```bash
# On Mac
open results/your_dance_visualization.html

# On Windows
start results/your_dance_visualization.html

# On Linux
xdg-open results/your_dance_visualization.html
```

### Reading the Comparison Report

The comparison report shows:
1. **Overall similarity score** (0-100%)
2. **Joint-pair breakdown** - Which body part relationships are similar/different
3. **Key observations** - Plain-language summary

## Step 4: Host for StoryMap

### Option A: GitHub Pages (Free)

1. Create a `docs/` folder in your repo
2. Copy HTML files there
3. Enable GitHub Pages in repository settings
4. Your visualizations will be at: `https://username.github.io/repo-name/filename.html`

### Option B: Direct Embed

If your StoryMap platform allows file uploads, you can upload the HTML directly.

## Troubleshooting

### "No joints detected"

Your CSV column names don't match expected patterns. Check that position columns contain keywords like:
- `Wrist`, `Elbow`, `Shoulder`, `Hip`, `Knee`, `Ankle`
- `Positions` or `Position`
- `X`, `Y`, `Z` suffixes

### Very low similarity scores

Low scores (< 10%) usually mean:
- Dances are genuinely different traditions
- Or: QTC thresholds need tuning for your data

Try adjusting thresholds:
```python
# In qtc_encoding.py, increase thresholds for noisier data
encode_dance_qtc(df, distance_threshold=10.0, movement_threshold=5.0)
```

### Visualization not animating

- Check browser console for JavaScript errors
- Ensure the HTML file loaded completely
- Try a different browser (Chrome/Firefox recommended)

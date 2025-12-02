# Global Movement Research (GMR) Pipeline

A Python pipeline for cross-cultural dance comparison using motion capture data. This tool enables researchers to analyze, visualize, and compare dance movements from different traditions using Qualitative Trajectory Calculus (QTC) encoding and sequence alignment methods.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

The GMR Pipeline transforms raw motion capture data into:
- **Standardized datasets** with consistent coordinate systems and frame rates
- **3D visualizations** that can be embedded in websites or StoryMaps
- **QTC encodings** that capture relational movement patterns between body parts
- **Similarity scores** comparing dances across multiple dimensions
- **Diffusion analysis** identifying potential movement pattern relationships

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gmr-pipeline.git
cd gmr-pipeline

# Install dependencies
pip install -r requirements.txt

# Run analysis on your motion capture files
python run_pipeline.py dance1.csv dance2.csv --output-dir results/
```

## Pipeline Modules

| Module | Description |
|--------|-------------|
| `data_cleaning.py` | Load, clean, and standardize motion capture data |
| `visualization.py` | Generate interactive 3D skeleton animations |
| `qtc_encoding.py` | Encode movements using Qualitative Trajectory Calculus |
| `sequence_alignment.py` | Compare QTC sequences using alignment algorithms |
| `multi_dance_analysis.py` | Analyze multiple dances, identify clusters |
| `run_pipeline.py` | Main script that orchestrates the full analysis |

## Input Data Format

The pipeline accepts CSV files from the **Captury motion capture system** with the following structure:

```
Row 1: Recording name
Row 2: Frame rate (e.g., "59.9988")
Row 3: Recording date
Row 4: Column headers (joint names with X/Y/Z suffixes)
Row 5: Units (mm for positions, degrees for rotations)
Row 6+: Frame data
```

### Supported Joints

The pipeline recognizes 24 standard joints:
- Head, Neck, Spine (1-4)
- Shoulders, Elbows, Wrists, Hands (L/R)
- Hips, Knees, Ankles, Toes (L/R)
- Center of Gravity

## Output Files

After running the pipeline, you'll find:

```
output_dir/
├── dance1_standardized.csv      # Cleaned, normalized data
├── dance1_visualization.html    # Interactive 3D animation
├── dance2_standardized.csv
├── dance2_visualization.html
├── comparison_dance1_vs_dance2.txt  # Detailed comparison
├── multi_dance_report.txt       # Summary report
└── analysis_results.json        # Machine-readable results
```

## 3D Visualizations

The generated HTML files are self-contained and can be:
- Opened directly in any web browser
- Embedded in ArcGIS StoryMaps using `<iframe>`
- Hosted on GitHub Pages
- Shared as standalone files

### Controls
- **Play/Pause**: Space bar or button
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Speed**: Adjust with slider (0.1x - 3x)

## Understanding QTC Encoding

Qualitative Trajectory Calculus captures how body parts move *relative to each other*:

| Symbol | Distance | Vertical | Horizontal |
|--------|----------|----------|------------|
| `-` | Approaching | Moving down | Moving left |
| `0` | Stable | Stable | Stable |
| `+` | Receding | Moving up | Moving right |

Each frame produces a 3-character code (e.g., `+0-` means "receding, vertically stable, moving left").

### Default Joint Pairs Analyzed

1. Hand L ↔ Hand R (bilateral coordination)
2. Hand L ↔ Hip R (cross-body reach)
3. Hand R ↔ Hip L (cross-body reach)
4. Head ↔ Center of Gravity (torso inclination)
5. Shoulder L ↔ Shoulder R (torso rotation)
6. Elbow L ↔ Knee R (cross-lateral coordination)
7. Elbow R ↔ Knee L (cross-lateral coordination)
8. Wrist L ↔ Ankle L (ipsilateral coordination)
9. Wrist R ↔ Ankle R (ipsilateral coordination)
10. Toe L ↔ Toe R (footwork patterns)

## Interpreting Results

### Similarity Scores

| Score | Interpretation |
|-------|----------------|
| 70-100% | High similarity - shared movement vocabulary |
| 40-70% | Moderate similarity - some common patterns |
| 0-40% | Low similarity - distinct movement traditions |

### Example Output

```
COMPARISON: Bata vs Esapkaide
==================================================
Overall Similarity: 5.2%

These dances show relatively DISTINCT movement patterns.

KEY OBSERVATIONS:
  • Different hand/arm patterns (5.1% avg)
  • Different footwork patterns (5.2% avg)
```

## Embedding in StoryMaps

To embed visualizations in ArcGIS StoryMap:

1. Host the HTML file (GitHub Pages, your server, etc.)
2. In StoryMap, add an "Embed" block
3. Use the URL to your hosted HTML file
4. Adjust dimensions as needed

Example GitHub Pages URL:
```
https://YOUR_USERNAME.github.io/gmr-pipeline/visualizations/Bata_dance_visualization.html
```

## Extending the Pipeline

### Adding New Joint Pairs

Edit `qtc_encoding.py`:

```python
DEFAULT_JOINT_PAIRS = [
    ('hand_L', 'hand_R'),
    ('your_joint_1', 'your_joint_2'),
    # Add more pairs...
]
```

### Adjusting Thresholds

```python
# In qtc_encoding.py
encode_dance_qtc(df, 
    distance_threshold=10.0,  # mm (default: 5.0)
    movement_threshold=5.0    # mm (default: 3.0)
)
```

### Custom Alignment Scoring

```python
# In sequence_alignment.py
needleman_wunsch(seq1, seq2,
    match_score=5,      # Reward for matches
    mismatch_score=-2,  # Penalty for mismatches
    gap_penalty=-3      # Penalty for gaps
)
```

## Requirements

- Python 3.8+
- pandas
- numpy

Install with:
```bash
pip install -r requirements.txt
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gmr_pipeline,
  title = {Global Movement Research Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/gmr-pipeline}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Motion capture data from Captury system
- QTC methodology adapted from movement science literature
- Sequence alignment based on Needleman-Wunsch and Smith-Waterman algorithms

# Methodology: QTC-Based Dance Comparison

This document explains the theoretical foundation and implementation of the GMR Pipeline's movement analysis approach.

## Why Qualitative Trajectory Calculus?

Traditional motion capture analysis often focuses on **absolute positions** - where body parts are located in 3D space. However, dance traditions are better characterized by **relational patterns** - how body parts move relative to each other.

Qualitative Trajectory Calculus (QTC) captures these relational dynamics by encoding:
1. Whether two points are approaching, stable, or receding
2. Their relative vertical movement
3. Their relative horizontal movement

This abstraction allows comparison across:
- Different performers (varying body sizes)
- Different speeds (tempo variations)
- Different spatial orientations (facing different directions)

## The Encoding Process

### Step 1: Data Standardization

Before encoding, motion capture data is normalized:

1. **Frame rate normalization**: Resample to 30 FPS
2. **Coordinate centering**: Translate so center of gravity is at origin
3. **Orientation alignment**: Rotate so dancer faces +Z axis
4. **Gap interpolation**: Fill missing frames (up to 5 consecutive)

### Step 2: Joint Pair Selection

We analyze 10 joint pairs that capture distinct movement dimensions:

| Pair | Movement Dimension |
|------|-------------------|
| Hand L ↔ Hand R | Bilateral arm coordination |
| Hand ↔ Opposite Hip | Cross-body reaching |
| Head ↔ CoG | Torso inclination |
| Shoulder L ↔ Shoulder R | Torso rotation |
| Elbow ↔ Opposite Knee | Cross-lateral patterns |
| Wrist ↔ Same-side Ankle | Ipsilateral coordination |
| Toe L ↔ Toe R | Footwork width/crossing |

### Step 3: QTC Symbol Generation

For each frame transition (frame n → frame n+1), we compute a 3-character symbol:

```
Symbol = [Distance][Vertical][Horizontal]

Distance:
  '-' : Points approaching (Δdistance < -threshold)
  '0' : Distance stable (|Δdistance| ≤ threshold)
  '+' : Points receding (Δdistance > +threshold)

Vertical (relative to joint B):
  '-' : Joint A moving down relative to B
  '0' : Vertically stable
  '+' : Joint A moving up relative to B

Horizontal (relative to joint B, in facing plane):
  '-' : Joint A moving left relative to B
  '0' : Horizontally stable
  '+' : Joint A moving right relative to B
```

### Step 4: Sequence Alignment

QTC sequences are compared using two algorithms:

**Needleman-Wunsch (Global Alignment)**
- Aligns entire sequences end-to-end
- Identifies overall structural similarity
- Useful for dances of similar length/structure

**Smith-Waterman (Local Alignment)**
- Finds best matching subsequences
- Identifies shared motifs within different contexts
- Useful for finding common phrases in longer dances

### Scoring

Alignment scoring uses partial matches:

| Match Type | Score |
|------------|-------|
| Perfect match (3/3 chars) | +3 |
| Partial match (2/3 chars) | +1.5 |
| Weak match (1/3 chars) | -0.5 |
| No match (0/3 chars) | -1 |
| Gap | -2 |

Normalized similarity = raw_score / max_possible_score

## Interpreting Results

### High Similarity (70%+)

Indicates shared movement vocabulary:
- Common dance tradition
- Direct transmission/learning
- Functional similarities (e.g., agricultural movements)

### Moderate Similarity (40-70%)

Suggests partial overlap:
- Related but distinct traditions
- Common ancestral influences
- Parallel evolution of some patterns

### Low Similarity (<40%)

Indicates distinct traditions:
- Independent development
- Different cultural contexts
- Different functional purposes

## Limitations

1. **Temporal abstraction**: QTC loses exact timing information
2. **Threshold sensitivity**: Results depend on threshold settings
3. **Joint pair selection**: Different pairs may reveal different patterns
4. **Data quality**: Noise in motion capture affects encoding

## References

- Van de Weghe, N., et al. (2005). "Qualitative Trajectory Calculus and the composition of movements." 
- Needleman, S.B. & Wunsch, C.D. (1970). "A general method applicable to the search for similarities in the amino acid sequence of two proteins."
- Smith, T.F. & Waterman, M.S. (1981). "Identification of common molecular subsequences."

"""
GMR Pipeline - Part C: QTC Encoding of Movement
================================================
This module encodes motion capture data into Qualitative Trajectory Calculus (QTC)
representations for cross-dance comparison.

QTC captures qualitative relationships between joint pairs:
- Distance change (approaching/stable/receding)
- Relative vertical movement (up/stable/down)
- Relative horizontal movement (left/stable/right)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Default joint pairs for QTC analysis
DEFAULT_JOINT_PAIRS = [
    ('hand_L', 'hand_R'),       # Hand relationship (clapping, spreading)
    ('hand_L', 'hip_R'),        # Cross-body reaching
    ('hand_R', 'hip_L'),        # Cross-body reaching
    ('head', 'center_of_gravity'),  # Torso bend/extension
    ('elbow_L', 'knee_R'),      # Cross-body coordination
    ('elbow_R', 'knee_L'),      # Cross-body coordination
    ('toe_L', 'toe_R'),         # Foot spacing
    ('shoulder_L', 'shoulder_R'),  # Shoulder rotation/tilt
    ('wrist_L', 'ankle_L'),     # Same-side coordination
    ('wrist_R', 'ankle_R'),     # Same-side coordination
]


@dataclass
class QTCSymbol:
    """
    A single QTC symbol encoding the relationship between two joints.
    
    Symbols:
    - distance: '-' (approaching), '0' (stable), '+' (receding)
    - vertical: '-' (A moving down relative to B), '0' (stable), '+' (A moving up relative to B)
    - horizontal: '-' (A moving left relative to B), '0' (stable), '+' (A moving right relative to B)
    """
    distance: str  # -, 0, +
    vertical: str  # -, 0, +
    horizontal: str  # -, 0, +
    
    def __str__(self):
        return f"{self.distance}{self.vertical}{self.horizontal}"
    
    def __eq__(self, other):
        if isinstance(other, QTCSymbol):
            return (self.distance == other.distance and 
                    self.vertical == other.vertical and 
                    self.horizontal == other.horizontal)
        return False
    
    def __hash__(self):
        return hash(str(self))
    
    @classmethod
    def from_string(cls, s: str) -> 'QTCSymbol':
        """Parse a QTC symbol from its string representation."""
        if len(s) != 3:
            raise ValueError(f"Invalid QTC symbol: {s}")
        return cls(s[0], s[1], s[2])


def compute_qtc_symbol(
    pos_a_curr: np.ndarray,  # Current position of joint A [x, y, z]
    pos_b_curr: np.ndarray,  # Current position of joint B [x, y, z]
    pos_a_prev: np.ndarray,  # Previous position of joint A
    pos_b_prev: np.ndarray,  # Previous position of joint B
    distance_threshold: float = 5.0,  # mm threshold for "stable"
    movement_threshold: float = 3.0,   # mm threshold for movement
) -> QTCSymbol:
    """
    Compute the QTC symbol for a joint pair at a single frame.
    
    Args:
        pos_a_curr, pos_b_curr: Current 3D positions
        pos_a_prev, pos_b_prev: Previous 3D positions
        distance_threshold: Threshold for distance change to be considered significant
        movement_threshold: Threshold for relative movement to be considered significant
        
    Returns:
        QTCSymbol encoding the qualitative relationship
    """
    # Distance change
    dist_curr = np.linalg.norm(pos_a_curr - pos_b_curr)
    dist_prev = np.linalg.norm(pos_a_prev - pos_b_prev)
    dist_delta = dist_curr - dist_prev
    
    if dist_delta < -distance_threshold:
        distance = '-'  # Approaching
    elif dist_delta > distance_threshold:
        distance = '+'  # Receding
    else:
        distance = '0'  # Stable
    
    # Relative vertical movement (Y axis)
    # Positive Y is up in our coordinate system
    rel_y_curr = pos_a_curr[1] - pos_b_curr[1]
    rel_y_prev = pos_a_prev[1] - pos_b_prev[1]
    rel_y_delta = rel_y_curr - rel_y_prev
    
    if rel_y_delta < -movement_threshold:
        vertical = '-'  # A moving down relative to B
    elif rel_y_delta > movement_threshold:
        vertical = '+'  # A moving up relative to B
    else:
        vertical = '0'  # Stable
    
    # Relative horizontal movement (X axis, left/right)
    rel_x_curr = pos_a_curr[0] - pos_b_curr[0]
    rel_x_prev = pos_a_prev[0] - pos_b_prev[0]
    rel_x_delta = rel_x_curr - rel_x_prev
    
    if rel_x_delta < -movement_threshold:
        horizontal = '-'  # A moving left relative to B
    elif rel_x_delta > movement_threshold:
        horizontal = '+'  # A moving right relative to B
    else:
        horizontal = '0'  # Stable
    
    return QTCSymbol(distance, vertical, horizontal)


def encode_joint_pair_qtc(
    df: pd.DataFrame,
    joint_a: str,
    joint_b: str,
    distance_threshold: float = 5.0,
    movement_threshold: float = 3.0,
) -> List[QTCSymbol]:
    """
    Encode a joint pair relationship as a sequence of QTC symbols.
    
    Args:
        df: Standardized DataFrame with joint positions
        joint_a, joint_b: Names of the two joints to compare
        distance_threshold: Threshold for distance change
        movement_threshold: Threshold for relative movement
        
    Returns:
        List of QTCSymbol for each frame transition
    """
    # Get position columns
    cols_a = [f"{joint_a}_x", f"{joint_a}_y", f"{joint_a}_z"]
    cols_b = [f"{joint_b}_x", f"{joint_b}_y", f"{joint_b}_z"]
    
    # Check if joints exist
    if not all(c in df.columns for c in cols_a + cols_b):
        return []
    
    symbols = []
    
    for i in range(1, len(df)):
        pos_a_curr = df.iloc[i][cols_a].values.astype(float)
        pos_b_curr = df.iloc[i][cols_b].values.astype(float)
        pos_a_prev = df.iloc[i-1][cols_a].values.astype(float)
        pos_b_prev = df.iloc[i-1][cols_b].values.astype(float)
        
        symbol = compute_qtc_symbol(
            pos_a_curr, pos_b_curr,
            pos_a_prev, pos_b_prev,
            distance_threshold, movement_threshold
        )
        symbols.append(symbol)
    
    return symbols


def encode_dance_qtc(
    df: pd.DataFrame,
    joint_pairs: List[Tuple[str, str]] = None,
    distance_threshold: float = 5.0,
    movement_threshold: float = 3.0,
) -> Dict[Tuple[str, str], List[QTCSymbol]]:
    """
    Encode a complete dance as QTC sequences for multiple joint pairs.
    
    Args:
        df: Standardized DataFrame with joint positions
        joint_pairs: List of joint pairs to encode (defaults to DEFAULT_JOINT_PAIRS)
        distance_threshold: Threshold for distance change
        movement_threshold: Threshold for relative movement
        
    Returns:
        Dictionary mapping joint pairs to their QTC sequences
    """
    if joint_pairs is None:
        joint_pairs = DEFAULT_JOINT_PAIRS
    
    result = {}
    
    for joint_a, joint_b in joint_pairs:
        symbols = encode_joint_pair_qtc(
            df, joint_a, joint_b,
            distance_threshold, movement_threshold
        )
        if symbols:
            result[(joint_a, joint_b)] = symbols
    
    return result


def qtc_to_string(symbols: List[QTCSymbol]) -> str:
    """Convert a QTC sequence to a string representation."""
    return ''.join(str(s) for s in symbols)


def string_to_qtc(s: str) -> List[QTCSymbol]:
    """Parse a string back into QTC symbols."""
    if len(s) % 3 != 0:
        raise ValueError(f"Invalid QTC string length: {len(s)}")
    return [QTCSymbol.from_string(s[i:i+3]) for i in range(0, len(s), 3)]


def compute_qtc_statistics(symbols: List[QTCSymbol]) -> Dict:
    """
    Compute statistics about a QTC sequence.
    
    Returns:
        Dictionary with counts and percentages of different symbol types
    """
    if not symbols:
        return {}
    
    n = len(symbols)
    
    # Count distance symbols
    dist_approaching = sum(1 for s in symbols if s.distance == '-')
    dist_stable = sum(1 for s in symbols if s.distance == '0')
    dist_receding = sum(1 for s in symbols if s.distance == '+')
    
    # Count vertical symbols
    vert_down = sum(1 for s in symbols if s.vertical == '-')
    vert_stable = sum(1 for s in symbols if s.vertical == '0')
    vert_up = sum(1 for s in symbols if s.vertical == '+')
    
    # Count horizontal symbols
    horz_left = sum(1 for s in symbols if s.horizontal == '-')
    horz_stable = sum(1 for s in symbols if s.horizontal == '0')
    horz_right = sum(1 for s in symbols if s.horizontal == '+')
    
    return {
        'total_frames': n,
        'distance': {
            'approaching': dist_approaching, 'approaching_pct': dist_approaching / n,
            'stable': dist_stable, 'stable_pct': dist_stable / n,
            'receding': dist_receding, 'receding_pct': dist_receding / n,
        },
        'vertical': {
            'down': vert_down, 'down_pct': vert_down / n,
            'stable': vert_stable, 'stable_pct': vert_stable / n,
            'up': vert_up, 'up_pct': vert_up / n,
        },
        'horizontal': {
            'left': horz_left, 'left_pct': horz_left / n,
            'stable': horz_stable, 'stable_pct': horz_stable / n,
            'right': horz_right, 'right_pct': horz_right / n,
        }
    }


def find_recurring_patterns(symbols: List[QTCSymbol], min_length: int = 3, min_occurrences: int = 2) -> Dict[str, int]:
    """
    Find recurring patterns (motifs) in a QTC sequence.
    
    Args:
        symbols: QTC sequence
        min_length: Minimum pattern length (in symbols)
        min_occurrences: Minimum number of occurrences to report
        
    Returns:
        Dictionary mapping pattern strings to their occurrence counts
    """
    s = qtc_to_string(symbols)
    patterns = {}
    
    # Look for patterns of various lengths
    for length in range(min_length * 3, len(s) // 2, 3):  # *3 because each symbol is 3 chars
        for start in range(0, len(s) - length, 3):
            pattern = s[start:start + length]
            count = s.count(pattern)
            if count >= min_occurrences and pattern not in patterns:
                patterns[pattern] = count
    
    # Sort by occurrence count
    return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))


# QTC Rule Documentation
QTC_RULES = """
QTC ENCODING RULES
==================

Each frame transition is encoded as a 3-character symbol: [distance][vertical][horizontal]

DISTANCE (joint A relative to joint B):
  '-' : A is approaching B (distance decreasing)
  '0' : Distance is stable (change below threshold)
  '+' : A is receding from B (distance increasing)

VERTICAL (A's Y position relative to B's Y position):
  '-' : A is moving DOWN relative to B
  '0' : Vertical relationship is stable
  '+' : A is moving UP relative to B

HORIZONTAL (A's X position relative to B's X position):
  '-' : A is moving LEFT relative to B
  '0' : Horizontal relationship is stable
  '+' : A is moving RIGHT relative to B

THRESHOLDS:
  - Distance threshold: Changes < threshold are considered stable
  - Movement threshold: Relative position changes < threshold are stable

EXAMPLES:
  '---' : Approaching + moving down + moving left
  '000' : All relationships stable (minimal movement)
  '++-' : Receding + moving up + moving left
  '-00' : Approaching while maintaining relative position
"""


if __name__ == "__main__":
    # Test with sample data
    from data_cleaning import process_dance_csv
    
    # Process a dance
    print("Loading and processing dance data...")
    df, summary = process_dance_csv("/mnt/user-data/uploads/Bata_dance_Sinclair.csv")
    
    print(f"\nEncoding QTC for {summary['processed_frames']} frames...")
    qtc_sequences = encode_dance_qtc(df)
    
    print(f"\nEncoded {len(qtc_sequences)} joint pairs:")
    for pair, symbols in qtc_sequences.items():
        print(f"\n  {pair[0]} <-> {pair[1]}: {len(symbols)} symbols")
        stats = compute_qtc_statistics(symbols)
        print(f"    Distance: {stats['distance']['approaching_pct']:.1%} approaching, "
              f"{stats['distance']['stable_pct']:.1%} stable, "
              f"{stats['distance']['receding_pct']:.1%} receding")
        
        # Find patterns
        patterns = find_recurring_patterns(symbols, min_length=4, min_occurrences=3)
        if patterns:
            print(f"    Top patterns: {list(patterns.items())[:3]}")
    
    print("\n" + QTC_RULES)

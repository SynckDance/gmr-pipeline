"""
GMR Pipeline - Part A: Data Cleaning & Standardization
=======================================================
This module handles loading, cleaning, and standardizing motion capture CSV data.

Key operations:
- Load Captury mocap CSV files
- Clean duplicates and handle missing values
- Normalize frame rate to consistent FPS
- Recenter coordinates around pelvis/center of gravity
- Align facing direction to consistent axis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Standard joint mapping for our skeleton
JOINT_MAPPING = {
    # Core body
    'center_of_gravity': 'CenterOfGravity',
    'spine': 'SpinePosition',
    'spine1': 'Spine1Position',
    'spine2': 'Spine2Position',
    'spine3': 'Spine3Position',
    'spine4': 'Spine4Position',
    'neck': 'NeckPosition',
    'head': 'HeadPosition',
    'head_end': 'HeadEndPosition',
    
    # Left arm
    'shoulder_L': 'LShoulderPositions',
    'elbow_L': 'LElbowPositions',
    'wrist_L': 'LWristPositions',
    'hand_L': 'LHandEndPosition',
    
    # Right arm
    'shoulder_R': 'RShoulderPositions',
    'elbow_R': 'RElbowPositions',
    'wrist_R': 'RWristPositions',
    'hand_R': 'RHandEndPosition',
    
    # Left leg
    'hip_L': 'LHipPositions',
    'knee_L': 'LKneePositions',
    'ankle_L': 'LAnklePositions',
    'toe_L': 'LToePositions',
    
    # Right leg
    'hip_R': 'RHipPositions',
    'knee_R': 'RKneePositions',
    'ankle_R': 'RAnklePositions',
    'toe_R': 'RToePositions',
}


def load_captury_csv(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a Captury motion capture CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Tuple of (dataframe with position data, metadata dict)
    """
    filepath = Path(filepath)
    
    # Read header lines for metadata
    with open(filepath, 'r') as f:
        line1 = f.readline().strip().split(';')
        line2 = f.readline().strip().split(';')
    
    metadata = {
        'recorded_at': line1[1] if len(line1) > 1 else None,
        'recorded_with': line1[3] if len(line1) > 3 else None,
        'frames': int(line2[1]) if len(line2) > 1 else None,
        'frame_rate': float(line2[3]) if len(line2) > 3 else None,
    }
    
    # Read the main data (skip first 5 header rows)
    df = pd.read_csv(filepath, sep=';', skiprows=5, header=None)
    
    # Read column names from row 3 (0-indexed: row 2)
    col_df = pd.read_csv(filepath, sep=';', skiprows=2, nrows=1, header=None)
    columns = col_df.iloc[0].tolist()
    
    # Build proper column names
    new_columns = []
    current_joint = None
    coord_idx = 0
    coords = ['X', 'Y', 'Z']
    
    for i, col in enumerate(columns):
        if i < 3:  # Frame, Sub Frame, Annotations
            if i == 0:
                new_columns.append('frame')
            elif i == 1:
                new_columns.append('sub_frame')
            else:
                new_columns.append('annotations')
        else:
            # Joint coordinate columns
            if pd.notna(col) and col.strip():
                current_joint = col.strip()
                coord_idx = 0
            
            if current_joint:
                new_columns.append(f"{current_joint}_{coords[coord_idx % 3]}")
                coord_idx += 1
    
    # Truncate or pad columns to match data
    if len(new_columns) < len(df.columns):
        new_columns.extend([f'col_{i}' for i in range(len(new_columns), len(df.columns))])
    elif len(new_columns) > len(df.columns):
        new_columns = new_columns[:len(df.columns)]
    
    df.columns = new_columns
    
    return df, metadata


def extract_joint_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized joint positions from the raw dataframe.
    
    Args:
        df: Raw dataframe from load_captury_csv
        
    Returns:
        DataFrame with standardized joint position columns
    """
    result = {'frame': df['frame'].values}
    
    for std_name, captury_name in JOINT_MAPPING.items():
        for coord in ['X', 'Y', 'Z']:
            col_name = f"{captury_name}_{coord}"
            if col_name in df.columns:
                result[f"{std_name}_{coord.lower()}"] = pd.to_numeric(df[col_name], errors='coerce').values
    
    return pd.DataFrame(result)


def clean_data(df: pd.DataFrame, max_gap_frames: int = 5) -> pd.DataFrame:
    """
    Clean the motion capture data.
    
    Operations:
    - Drop duplicate rows
    - Interpolate short gaps in position data
    - Drop rows with excessive missing data
    
    Args:
        df: DataFrame with joint positions
        max_gap_frames: Maximum gap size to interpolate
        
    Returns:
        Cleaned DataFrame
    """
    df = df.drop_duplicates()
    
    # Get position columns (exclude 'frame')
    pos_cols = [c for c in df.columns if c != 'frame']
    
    # Interpolate short gaps
    for col in pos_cols:
        df[col] = df[col].interpolate(method='linear', limit=max_gap_frames)
    
    # Drop rows where more than 50% of position data is missing
    threshold = len(pos_cols) * 0.5
    df = df.dropna(thresh=int(threshold), subset=pos_cols)
    
    return df.reset_index(drop=True)


def normalize_frame_rate(df: pd.DataFrame, current_fps: float, target_fps: float = 30.0) -> pd.DataFrame:
    """
    Resample data to a target frame rate.
    
    Args:
        df: DataFrame with joint positions
        current_fps: Current frame rate of the data
        target_fps: Target frame rate
        
    Returns:
        Resampled DataFrame
    """
    if abs(current_fps - target_fps) < 0.1:
        return df
    
    # Calculate resampling ratio
    ratio = current_fps / target_fps
    
    # Create new frame indices
    n_new_frames = int(len(df) / ratio)
    new_indices = np.linspace(0, len(df) - 1, n_new_frames)
    
    # Interpolate all columns
    result = {'frame': np.arange(n_new_frames)}
    pos_cols = [c for c in df.columns if c != 'frame']
    
    for col in pos_cols:
        result[col] = np.interp(new_indices, np.arange(len(df)), df[col].values)
    
    return pd.DataFrame(result)


def recenter_around_cog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recenter all joint positions around the Center of Gravity.
    This keeps the figure near the origin.
    
    Args:
        df: DataFrame with joint positions
        
    Returns:
        Recentered DataFrame
    """
    df = df.copy()
    
    # Get COG columns
    cog_x = df.get('center_of_gravity_x', None)
    cog_y = df.get('center_of_gravity_y', None)  
    cog_z = df.get('center_of_gravity_z', None)
    
    if cog_x is None:
        print("Warning: No center_of_gravity found, skipping recentering")
        return df
    
    # Subtract COG from all position columns
    for col in df.columns:
        if col.endswith('_x') and col != 'center_of_gravity_x':
            df[col] = df[col] - cog_x
        elif col.endswith('_y') and col != 'center_of_gravity_y':
            df[col] = df[col] - cog_y
        elif col.endswith('_z') and col != 'center_of_gravity_z':
            df[col] = df[col] - cog_z
    
    # Set COG to origin
    df['center_of_gravity_x'] = 0
    df['center_of_gravity_y'] = 0
    df['center_of_gravity_z'] = 0
    
    return df


def align_facing_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rotate the skeleton so the dancer faces a consistent direction (+Z axis).
    Uses the vector from spine to head to determine facing.
    
    Args:
        df: DataFrame with joint positions
        
    Returns:
        Rotated DataFrame
    """
    df = df.copy()
    
    # Calculate average facing direction using shoulders
    shoulder_l_x = df.get('shoulder_L_x', df.get('spine_x', None))
    shoulder_l_z = df.get('shoulder_L_z', df.get('spine_z', None))
    shoulder_r_x = df.get('shoulder_R_x', df.get('spine_x', None))
    shoulder_r_z = df.get('shoulder_R_z', df.get('spine_z', None))
    
    if shoulder_l_x is None:
        print("Warning: Cannot determine facing direction, skipping alignment")
        return df
    
    # Get average shoulder midpoint direction
    mid_x = (shoulder_l_x.mean() + shoulder_r_x.mean()) / 2
    mid_z = (shoulder_l_z.mean() + shoulder_r_z.mean()) / 2
    
    # Calculate rotation angle to face +Z
    current_angle = np.arctan2(mid_x, mid_z)
    
    # Rotate all X-Z positions
    cos_a = np.cos(-current_angle)
    sin_a = np.sin(-current_angle)
    
    x_cols = [c for c in df.columns if c.endswith('_x')]
    z_cols = [c.replace('_x', '_z') for c in x_cols]
    
    for x_col, z_col in zip(x_cols, z_cols):
        if x_col in df.columns and z_col in df.columns:
            x_vals = df[x_col].values
            z_vals = df[z_col].values
            df[x_col] = x_vals * cos_a - z_vals * sin_a
            df[z_col] = x_vals * sin_a + z_vals * cos_a
    
    return df


def process_dance_csv(filepath: str, target_fps: float = 30.0) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to process a dance CSV file through the full cleaning pipeline.
    
    Args:
        filepath: Path to the Captury CSV file
        target_fps: Target frame rate for normalization
        
    Returns:
        Tuple of (processed DataFrame, summary dict)
    """
    # Load raw data
    raw_df, metadata = load_captury_csv(filepath)
    
    # Extract standardized positions
    df = extract_joint_positions(raw_df)
    
    original_frames = len(df)
    joints_detected = [c.rsplit('_', 1)[0] for c in df.columns if c != 'frame']
    joints_detected = list(set(joints_detected))
    
    # Clean data
    df = clean_data(df)
    
    # Normalize frame rate
    if metadata['frame_rate']:
        df = normalize_frame_rate(df, metadata['frame_rate'], target_fps)
    
    # Recenter around COG
    df = recenter_around_cog(df)
    
    # Align facing direction
    df = align_facing_direction(df)
    
    summary = {
        'source_file': str(filepath),
        'original_frames': original_frames,
        'processed_frames': len(df),
        'original_fps': metadata['frame_rate'],
        'target_fps': target_fps,
        'joints_detected': sorted(joints_detected),
        'recorded_at': metadata['recorded_at'],
        'cleaning_notes': []
    }
    
    if original_frames != len(df):
        summary['cleaning_notes'].append(f"Resampled from {original_frames} to {len(df)} frames")
    
    return df, summary


def save_standardized_csv(df: pd.DataFrame, output_path: str):
    """Save the standardized DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved standardized data to {output_path}")


# Demo/test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/mnt/user-data/uploads/Bata_dance_Sinclair.csv"
    
    print(f"\nProcessing: {filepath}")
    print("=" * 60)
    
    df, summary = process_dance_csv(filepath)
    
    print(f"\nSummary:")
    print(f"  Original frames: {summary['original_frames']}")
    print(f"  Processed frames: {summary['processed_frames']}")
    print(f"  Original FPS: {summary['original_fps']}")
    print(f"  Target FPS: {summary['target_fps']}")
    print(f"  Joints detected ({len(summary['joints_detected'])}): {summary['joints_detected'][:10]}...")
    print(f"  Recorded at: {summary['recorded_at']}")
    
    if summary['cleaning_notes']:
        print(f"  Notes: {summary['cleaning_notes']}")
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")

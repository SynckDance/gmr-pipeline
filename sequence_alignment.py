"""
GMR Pipeline - Part D: Sequence Alignment Methods (SAMs)
=========================================================
This module implements sequence alignment algorithms for comparing QTC-encoded dances.

Uses dynamic programming (Needleman-Wunsch for global, Smith-Waterman for local)
to find optimal alignments and compute similarity scores.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from qtc_encoding import QTCSymbol, qtc_to_string, string_to_qtc

# Default scoring parameters
DEFAULT_MATCH_SCORE = 3
DEFAULT_MISMATCH_SCORE = -1
DEFAULT_GAP_PENALTY = -2


@dataclass
class AlignmentResult:
    """Result of a sequence alignment."""
    seq1_aligned: str
    seq2_aligned: str
    score: float
    normalized_score: float  # 0-1 similarity
    alignment_length: int
    matches: int
    mismatches: int
    gaps: int
    
    def __str__(self):
        return (f"Score: {self.score:.1f} | Normalized: {self.normalized_score:.3f}\n"
                f"Length: {self.alignment_length} | Matches: {self.matches} | "
                f"Mismatches: {self.mismatches} | Gaps: {self.gaps}")


def qtc_match_score(s1: str, s2: str, match: float = DEFAULT_MATCH_SCORE, 
                    mismatch: float = DEFAULT_MISMATCH_SCORE) -> float:
    """
    Compute match score between two QTC symbols (as 3-char strings).
    
    Partial matches (e.g., same distance but different vertical) get intermediate scores.
    """
    if len(s1) != 3 or len(s2) != 3:
        return mismatch
    
    # Count matching dimensions
    matches = sum(1 for a, b in zip(s1, s2) if a == b)
    
    if matches == 3:
        return match  # Perfect match
    elif matches == 2:
        return match * 0.5  # 2/3 match
    elif matches == 1:
        return mismatch * 0.5  # 1/3 match
    else:
        return mismatch  # No match


def needleman_wunsch(
    seq1: str, 
    seq2: str,
    match_score: float = DEFAULT_MATCH_SCORE,
    mismatch_score: float = DEFAULT_MISMATCH_SCORE,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> AlignmentResult:
    """
    Global sequence alignment using Needleman-Wunsch algorithm.
    
    Args:
        seq1, seq2: QTC sequences as strings (each symbol is 3 chars)
        match_score: Score for matching symbols
        mismatch_score: Score for mismatching symbols
        gap_penalty: Penalty for gaps
        
    Returns:
        AlignmentResult with aligned sequences and scores
    """
    # Work with 3-char symbols
    n1 = len(seq1) // 3
    n2 = len(seq2) // 3
    
    # Initialize scoring matrix
    score_matrix = np.zeros((n1 + 1, n2 + 1))
    
    # Initialize first row and column with gap penalties
    for i in range(n1 + 1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(n2 + 1):
        score_matrix[0, j] = j * gap_penalty
    
    # Fill the matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            s1 = seq1[(i-1)*3:i*3]
            s2 = seq2[(j-1)*3:j*3]
            
            match = score_matrix[i-1, j-1] + qtc_match_score(s1, s2, match_score, mismatch_score)
            delete = score_matrix[i-1, j] + gap_penalty
            insert = score_matrix[i, j-1] + gap_penalty
            
            score_matrix[i, j] = max(match, delete, insert)
    
    # Traceback
    aligned1, aligned2 = [], []
    i, j = n1, n2
    matches, mismatches, gaps = 0, 0, 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s1 = seq1[(i-1)*3:i*3]
            s2 = seq2[(j-1)*3:j*3]
            current = score_matrix[i, j]
            diagonal = score_matrix[i-1, j-1]
            
            if current == diagonal + qtc_match_score(s1, s2, match_score, mismatch_score):
                aligned1.append(s1)
                aligned2.append(s2)
                if s1 == s2:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
                continue
        
        if i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_penalty:
            aligned1.append(seq1[(i-1)*3:i*3])
            aligned2.append('---')
            gaps += 1
            i -= 1
        else:
            aligned1.append('---')
            aligned2.append(seq2[(j-1)*3:j*3])
            gaps += 1
            j -= 1
    
    # Reverse alignments
    aligned1.reverse()
    aligned2.reverse()
    
    final_score = score_matrix[n1, n2]
    alignment_length = len(aligned1)
    
    # Normalize score: 0-1 scale
    max_possible = max(n1, n2) * match_score
    min_possible = max(n1, n2) * min(mismatch_score, gap_penalty)
    normalized = (final_score - min_possible) / (max_possible - min_possible) if max_possible != min_possible else 0
    normalized = max(0, min(1, normalized))
    
    return AlignmentResult(
        seq1_aligned=''.join(aligned1),
        seq2_aligned=''.join(aligned2),
        score=final_score,
        normalized_score=normalized,
        alignment_length=alignment_length,
        matches=matches,
        mismatches=mismatches,
        gaps=gaps
    )


def smith_waterman(
    seq1: str, 
    seq2: str,
    match_score: float = DEFAULT_MATCH_SCORE,
    mismatch_score: float = DEFAULT_MISMATCH_SCORE,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> Tuple[AlignmentResult, int, int]:
    """
    Local sequence alignment using Smith-Waterman algorithm.
    Finds the best matching subsequence.
    
    Args:
        seq1, seq2: QTC sequences as strings
        match_score, mismatch_score, gap_penalty: Scoring parameters
        
    Returns:
        Tuple of (AlignmentResult, start_pos_seq1, start_pos_seq2)
    """
    n1 = len(seq1) // 3
    n2 = len(seq2) // 3
    
    # Initialize scoring matrix
    score_matrix = np.zeros((n1 + 1, n2 + 1))
    
    max_score = 0
    max_pos = (0, 0)
    
    # Fill the matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            s1 = seq1[(i-1)*3:i*3]
            s2 = seq2[(j-1)*3:j*3]
            
            match = score_matrix[i-1, j-1] + qtc_match_score(s1, s2, match_score, mismatch_score)
            delete = score_matrix[i-1, j] + gap_penalty
            insert = score_matrix[i, j-1] + gap_penalty
            
            score_matrix[i, j] = max(0, match, delete, insert)
            
            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                max_pos = (i, j)
    
    # Traceback from maximum
    aligned1, aligned2 = [], []
    i, j = max_pos
    matches, mismatches, gaps = 0, 0, 0
    start_i, start_j = i, j
    
    while i > 0 and j > 0 and score_matrix[i, j] > 0:
        s1 = seq1[(i-1)*3:i*3]
        s2 = seq2[(j-1)*3:j*3]
        current = score_matrix[i, j]
        diagonal = score_matrix[i-1, j-1]
        
        if current == diagonal + qtc_match_score(s1, s2, match_score, mismatch_score):
            aligned1.append(s1)
            aligned2.append(s2)
            if s1 == s2:
                matches += 1
            else:
                mismatches += 1
            start_i, start_j = i-1, j-1
            i -= 1
            j -= 1
        elif i > 0 and current == score_matrix[i-1, j] + gap_penalty:
            aligned1.append(seq1[(i-1)*3:i*3])
            aligned2.append('---')
            gaps += 1
            start_i = i-1
            i -= 1
        else:
            aligned1.append('---')
            aligned2.append(seq2[(j-1)*3:j*3])
            gaps += 1
            start_j = j-1
            j -= 1
    
    aligned1.reverse()
    aligned2.reverse()
    
    alignment_length = len(aligned1)
    
    # Normalize
    max_possible = alignment_length * match_score if alignment_length > 0 else 1
    normalized = max_score / max_possible if max_possible > 0 else 0
    
    result = AlignmentResult(
        seq1_aligned=''.join(aligned1),
        seq2_aligned=''.join(aligned2),
        score=max_score,
        normalized_score=normalized,
        alignment_length=alignment_length,
        matches=matches,
        mismatches=mismatches,
        gaps=gaps
    )
    
    return result, start_i, start_j


def find_high_similarity_regions(
    seq1: str,
    seq2: str,
    window_size: int = 30,  # In symbols (3 chars each)
    threshold: float = 0.6,
    match_score: float = DEFAULT_MATCH_SCORE,
    mismatch_score: float = DEFAULT_MISMATCH_SCORE,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> List[Dict]:
    """
    Find regions of high similarity between two sequences using a sliding window.
    
    Args:
        seq1, seq2: QTC sequences
        window_size: Size of sliding window in symbols
        threshold: Minimum normalized score to report
        
    Returns:
        List of dictionaries describing high-similarity regions
    """
    regions = []
    n1 = len(seq1) // 3
    n2 = len(seq2) // 3
    
    window_chars = window_size * 3
    
    for i in range(0, len(seq1) - window_chars + 1, 3):  # Step by 1 symbol
        window1 = seq1[i:i + window_chars]
        
        best_score = 0
        best_j = 0
        
        for j in range(0, len(seq2) - window_chars + 1, 3):
            window2 = seq2[j:j + window_chars]
            
            result = needleman_wunsch(window1, window2, match_score, mismatch_score, gap_penalty)
            
            if result.normalized_score > best_score:
                best_score = result.normalized_score
                best_j = j
        
        if best_score >= threshold:
            regions.append({
                'seq1_start': i // 3,
                'seq1_end': (i + window_chars) // 3,
                'seq2_start': best_j // 3,
                'seq2_end': (best_j + window_chars) // 3,
                'similarity': best_score,
            })
    
    # Merge overlapping regions
    if not regions:
        return regions
    
    regions.sort(key=lambda x: x['seq1_start'])
    merged = [regions[0]]
    
    for region in regions[1:]:
        last = merged[-1]
        if region['seq1_start'] <= last['seq1_end']:
            # Overlapping - extend
            last['seq1_end'] = max(last['seq1_end'], region['seq1_end'])
            last['seq2_end'] = max(last['seq2_end'], region['seq2_end'])
            last['similarity'] = max(last['similarity'], region['similarity'])
        else:
            merged.append(region)
    
    return merged


def compare_joint_pair_sequences(
    qtc1: Dict[Tuple[str, str], List[QTCSymbol]],
    qtc2: Dict[Tuple[str, str], List[QTCSymbol]],
    match_score: float = DEFAULT_MATCH_SCORE,
    mismatch_score: float = DEFAULT_MISMATCH_SCORE,
    gap_penalty: float = DEFAULT_GAP_PENALTY,
) -> Dict:
    """
    Compare two dances by aligning their QTC sequences for each joint pair.
    
    Args:
        qtc1, qtc2: QTC encodings from encode_dance_qtc
        
    Returns:
        Dictionary with comparison results for each joint pair
    """
    common_pairs = set(qtc1.keys()) & set(qtc2.keys())
    
    results = {}
    
    for pair in common_pairs:
        seq1 = qtc_to_string(qtc1[pair])
        seq2 = qtc_to_string(qtc2[pair])
        
        # Global alignment
        global_result = needleman_wunsch(seq1, seq2, match_score, mismatch_score, gap_penalty)
        
        # Local alignment
        local_result, local_start1, local_start2 = smith_waterman(seq1, seq2, match_score, mismatch_score, gap_penalty)
        
        results[pair] = {
            'global_alignment': global_result,
            'local_alignment': local_result,
            'local_position': (local_start1, local_start2),
            'seq1_length': len(seq1) // 3,
            'seq2_length': len(seq2) // 3,
        }
    
    return results


def compute_overall_similarity(comparison_results: Dict) -> float:
    """
    Compute overall similarity score from all joint pair comparisons.
    
    Uses weighted average of global alignment scores.
    """
    if not comparison_results:
        return 0.0
    
    total_score = 0
    total_weight = 0
    
    for pair, result in comparison_results.items():
        score = result['global_alignment'].normalized_score
        # Weight by average sequence length
        weight = (result['seq1_length'] + result['seq2_length']) / 2
        total_score += score * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0


def generate_comparison_narrative(
    comparison_results: Dict,
    dance1_name: str = "Dance 1",
    dance2_name: str = "Dance 2"
) -> str:
    """
    Generate a human-readable narrative describing the comparison results.
    
    Args:
        comparison_results: Output from compare_joint_pair_sequences
        dance1_name, dance2_name: Names for the dances
        
    Returns:
        Plain-language description of similarities and differences
    """
    if not comparison_results:
        return "No common joint pairs found for comparison."
    
    overall = compute_overall_similarity(comparison_results)
    
    # Categorize joint pairs by similarity
    high_sim = []  # > 0.7
    med_sim = []   # 0.4-0.7
    low_sim = []   # < 0.4
    
    for pair, result in comparison_results.items():
        score = result['global_alignment'].normalized_score
        pair_name = f"{pair[0]} ↔ {pair[1]}"
        if score > 0.7:
            high_sim.append((pair_name, score))
        elif score > 0.4:
            med_sim.append((pair_name, score))
        else:
            low_sim.append((pair_name, score))
    
    # Build narrative
    lines = [
        f"COMPARISON: {dance1_name} vs {dance2_name}",
        "=" * 50,
        f"\nOverall Similarity: {overall:.1%}",
        ""
    ]
    
    if overall > 0.7:
        lines.append(f"These dances show STRONG overall similarity in movement patterns.")
    elif overall > 0.4:
        lines.append(f"These dances show MODERATE similarity with some shared and distinct patterns.")
    else:
        lines.append(f"These dances show relatively DISTINCT movement patterns.")
    
    lines.append("")
    
    if high_sim:
        lines.append("SHARED MOVEMENT PATTERNS (high similarity):")
        for name, score in sorted(high_sim, key=lambda x: -x[1]):
            lines.append(f"  • {name}: {score:.1%} match")
        lines.append("")
    
    if med_sim:
        lines.append("RELATED PATTERNS (moderate similarity):")
        for name, score in sorted(med_sim, key=lambda x: -x[1]):
            lines.append(f"  • {name}: {score:.1%} match")
        lines.append("")
    
    if low_sim:
        lines.append("DISTINCT PATTERNS (low similarity):")
        for name, score in sorted(low_sim, key=lambda x: -x[1]):
            lines.append(f"  • {name}: {score:.1%} match")
    
    # Add specific observations
    lines.append("\nKEY OBSERVATIONS:")
    
    # Check for hand coordination
    hand_pairs = [p for p in comparison_results if 'hand' in str(p).lower()]
    if hand_pairs:
        hand_avg = np.mean([comparison_results[p]['global_alignment'].normalized_score for p in hand_pairs])
        if hand_avg > 0.6:
            lines.append(f"  • Similar hand/arm coordination patterns ({hand_avg:.1%} avg)")
        else:
            lines.append(f"  • Different hand/arm patterns ({hand_avg:.1%} avg)")
    
    # Check for leg coordination
    leg_pairs = [p for p in comparison_results if any(x in str(p).lower() for x in ['toe', 'knee', 'ankle'])]
    if leg_pairs:
        leg_avg = np.mean([comparison_results[p]['global_alignment'].normalized_score for p in leg_pairs])
        if leg_avg > 0.6:
            lines.append(f"  • Similar footwork/leg patterns ({leg_avg:.1%} avg)")
        else:
            lines.append(f"  • Different footwork patterns ({leg_avg:.1%} avg)")
    
    # Check for cross-body coordination
    cross_pairs = [p for p in comparison_results if 
                   ('_L' in str(p) and '_R' in str(p)) or ('_R' in str(p) and '_L' in str(p))]
    if cross_pairs:
        cross_avg = np.mean([comparison_results[p]['global_alignment'].normalized_score for p in cross_pairs])
        if cross_avg > 0.6:
            lines.append(f"  • Similar cross-body coordination ({cross_avg:.1%} avg)")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    # Test with two dances
    from data_cleaning import process_dance_csv
    from qtc_encoding import encode_dance_qtc
    
    print("Loading dances...")
    df1, summary1 = process_dance_csv("/mnt/user-data/uploads/Bata_dance_Sinclair.csv")
    df2, summary2 = process_dance_csv("/mnt/user-data/uploads/Esapkaide_Sinclair__2_.csv")
    
    print("Encoding QTC...")
    qtc1 = encode_dance_qtc(df1)
    qtc2 = encode_dance_qtc(df2)
    
    print("Comparing sequences...")
    results = compare_joint_pair_sequences(qtc1, qtc2)
    
    print("\n" + generate_comparison_narrative(results, "Bata Dance", "Esapkaide"))

"""
GMR Pipeline - Part E: Multi-Dance Comparison & Diffusion Analysis
===================================================================
This module enables comparison across multiple dances to identify:
- Clusters of similar dances
- Potential diffusion pathways
- Shared movement signatures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

from data_cleaning import process_dance_csv
from qtc_encoding import encode_dance_qtc, compute_qtc_statistics, find_recurring_patterns
from sequence_alignment import (
    compare_joint_pair_sequences, 
    compute_overall_similarity,
    generate_comparison_narrative
)


def load_multiple_dances(file_paths: List[str], target_fps: float = 30.0) -> Dict[str, Dict]:
    """
    Load and process multiple dance files.
    
    Args:
        file_paths: List of paths to CSV files
        target_fps: Target frame rate for normalization
        
    Returns:
        Dictionary mapping dance names to their data
    """
    dances = {}
    
    for path in file_paths:
        name = Path(path).stem
        print(f"Processing: {name}")
        
        try:
            df, summary = process_dance_csv(path, target_fps)
            qtc = encode_dance_qtc(df)
            
            dances[name] = {
                'df': df,
                'summary': summary,
                'qtc': qtc,
                'path': path
            }
        except Exception as e:
            print(f"  Error processing {name}: {e}")
    
    return dances


def compute_pairwise_similarity_matrix(dances: Dict[str, Dict]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise similarity scores between all dances.
    
    Args:
        dances: Dictionary from load_multiple_dances
        
    Returns:
        Tuple of (similarity matrix, list of dance names)
    """
    names = list(dances.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        matrix[i, i] = 1.0  # Self-similarity
        for j in range(i + 1, n):
            qtc1 = dances[names[i]]['qtc']
            qtc2 = dances[names[j]]['qtc']
            
            results = compare_joint_pair_sequences(qtc1, qtc2)
            similarity = compute_overall_similarity(results)
            
            matrix[i, j] = similarity
            matrix[j, i] = similarity
    
    return matrix, names


def identify_clusters(similarity_matrix: np.ndarray, names: List[str], 
                      threshold: float = 0.5) -> List[List[str]]:
    """
    Identify clusters of similar dances using simple threshold-based clustering.
    
    Args:
        similarity_matrix: Pairwise similarity scores
        names: Dance names
        threshold: Minimum similarity to be in same cluster
        
    Returns:
        List of clusters (each cluster is a list of dance names)
    """
    n = len(names)
    assigned = [False] * n
    clusters = []
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start new cluster
        cluster = [names[i]]
        assigned[i] = True
        
        for j in range(i + 1, n):
            if not assigned[j] and similarity_matrix[i, j] >= threshold:
                cluster.append(names[j])
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def analyze_cluster_signatures(dances: Dict[str, Dict], cluster: List[str]) -> Dict:
    """
    Analyze shared movement signatures within a cluster.
    
    Args:
        dances: Dictionary from load_multiple_dances
        cluster: List of dance names in the cluster
        
    Returns:
        Dictionary describing shared signatures
    """
    if len(cluster) < 2:
        return {'cluster': cluster, 'shared_patterns': [], 'notes': 'Single dance cluster'}
    
    # Collect QTC statistics for each dance
    all_stats = {}
    for name in cluster:
        qtc = dances[name]['qtc']
        stats = {}
        for pair, symbols in qtc.items():
            stats[pair] = compute_qtc_statistics(symbols)
        all_stats[name] = stats
    
    # Find common patterns across joint pairs
    shared_patterns = []
    
    # Get common joint pairs
    common_pairs = set.intersection(*[set(dances[n]['qtc'].keys()) for n in cluster])
    
    for pair in common_pairs:
        pair_data = {
            'joint_pair': f"{pair[0]} ↔ {pair[1]}",
            'analysis': {}
        }
        
        # Compare distance patterns
        dist_approaching = [all_stats[n][pair]['distance']['approaching_pct'] for n in cluster]
        dist_stable = [all_stats[n][pair]['distance']['stable_pct'] for n in cluster]
        dist_receding = [all_stats[n][pair]['distance']['receding_pct'] for n in cluster]
        
        # Check for consistent patterns (low variance)
        if np.std(dist_approaching) < 0.15:
            avg = np.mean(dist_approaching)
            if avg > 0.3:
                pair_data['analysis']['distance'] = f"Consistent approaching tendency ({avg:.0%})"
            elif avg < 0.15:
                pair_data['analysis']['distance'] = f"Consistently stable/receding"
        
        # Compare vertical patterns
        vert_up = [all_stats[n][pair]['vertical']['up_pct'] for n in cluster]
        vert_down = [all_stats[n][pair]['vertical']['down_pct'] for n in cluster]
        
        if np.std(vert_up) < 0.15 and np.mean(vert_up) > 0.3:
            pair_data['analysis']['vertical'] = f"Shared upward movement tendency"
        elif np.std(vert_down) < 0.15 and np.mean(vert_down) > 0.3:
            pair_data['analysis']['vertical'] = f"Shared downward movement tendency"
        
        if pair_data['analysis']:
            shared_patterns.append(pair_data)
    
    return {
        'cluster': cluster,
        'cluster_size': len(cluster),
        'shared_patterns': shared_patterns,
        'common_joint_pairs': len(common_pairs)
    }


def suggest_diffusion_pathways(similarity_matrix: np.ndarray, names: List[str],
                               threshold: float = 0.4) -> List[Dict]:
    """
    Suggest potential diffusion pathways based on similarity patterns.
    
    Identifies chains of related dances that might indicate historical or
    geographical diffusion of movement patterns.
    
    Args:
        similarity_matrix: Pairwise similarity scores
        names: Dance names
        threshold: Minimum similarity for connection
        
    Returns:
        List of potential pathway descriptions
    """
    n = len(names)
    pathways = []
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= threshold:
                adj[i].append((j, similarity_matrix[i, j]))
                adj[j].append((i, similarity_matrix[i, j]))
    
    # Find connected components with strong links
    visited = set()
    
    def dfs_path(start, path, min_score):
        results = []
        for next_node, score in adj[start]:
            if next_node not in path:
                new_path = path + [next_node]
                new_min = min(min_score, score)
                if len(new_path) >= 2:
                    results.append((new_path, new_min))
                results.extend(dfs_path(next_node, new_path, new_min))
        return results
    
    for i in range(n):
        paths = dfs_path(i, [i], 1.0)
        for path, score in paths:
            if len(path) >= 2:
                pathway = {
                    'dances': [names[p] for p in path],
                    'strength': score,
                    'path_length': len(path),
                    'description': f"Connection chain: {' → '.join(names[p] for p in path)}"
                }
                pathways.append(pathway)
    
    # Remove duplicates and sort by strength
    unique_pathways = []
    seen = set()
    for p in sorted(pathways, key=lambda x: (-x['path_length'], -x['strength'])):
        key = tuple(sorted(p['dances']))
        if key not in seen:
            seen.add(key)
            unique_pathways.append(p)
    
    return unique_pathways[:10]  # Top 10 pathways


def generate_multi_dance_report(dances: Dict[str, Dict], 
                                similarity_matrix: np.ndarray,
                                names: List[str]) -> str:
    """
    Generate a comprehensive report comparing multiple dances.
    
    Args:
        dances: Dictionary from load_multiple_dances
        similarity_matrix: Pairwise similarity matrix
        names: Dance names
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "GLOBAL MOVEMENT RESEARCH - MULTI-DANCE COMPARISON REPORT",
        "=" * 70,
        "",
        f"Number of dances analyzed: {len(names)}",
        f"Dances: {', '.join(names)}",
        "",
    ]
    
    # Similarity matrix summary
    lines.extend([
        "PAIRWISE SIMILARITY MATRIX",
        "-" * 40,
    ])
    
    # Create formatted matrix
    header = "             " + "  ".join(f"{n[:8]:>8}" for n in names)
    lines.append(header)
    
    for i, name in enumerate(names):
        row = f"{name[:12]:12} " + "  ".join(f"{similarity_matrix[i,j]:8.2f}" for j in range(len(names)))
        lines.append(row)
    
    lines.append("")
    
    # Clusters
    clusters = identify_clusters(similarity_matrix, names, threshold=0.5)
    lines.extend([
        "IDENTIFIED CLUSTERS (threshold: 50% similarity)",
        "-" * 40,
    ])
    
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"Cluster {i}: {', '.join(cluster)}")
        
        if len(cluster) > 1:
            sig = analyze_cluster_signatures(dances, cluster)
            if sig['shared_patterns']:
                lines.append("  Shared signatures:")
                for pattern in sig['shared_patterns'][:3]:
                    lines.append(f"    • {pattern['joint_pair']}")
                    for key, val in pattern['analysis'].items():
                        lines.append(f"      - {val}")
    
    lines.append("")
    
    # Diffusion pathways
    pathways = suggest_diffusion_pathways(similarity_matrix, names)
    if pathways:
        lines.extend([
            "POTENTIAL DIFFUSION PATHWAYS",
            "-" * 40,
            "Note: These are speculative suggestions based on movement similarity.",
            "Further research is needed to establish actual historical connections.",
            "",
        ])
        
        for pathway in pathways[:5]:
            lines.append(f"• {pathway['description']} (strength: {pathway['strength']:.0%})")
    
    lines.append("")
    
    # Summary statistics
    avg_sim = np.mean(similarity_matrix[np.triu_indices(len(names), k=1)])
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix * (1 - np.eye(len(names)))), similarity_matrix.shape)
    min_sim_idx = np.unravel_index(np.argmin(similarity_matrix + np.eye(len(names)) * 999), similarity_matrix.shape)
    
    lines.extend([
        "SUMMARY STATISTICS",
        "-" * 40,
        f"Average pairwise similarity: {avg_sim:.1%}",
        f"Most similar pair: {names[max_sim_idx[0]]} & {names[max_sim_idx[1]]} ({similarity_matrix[max_sim_idx]:.1%})",
        f"Least similar pair: {names[min_sim_idx[0]]} & {names[min_sim_idx[1]]} ({similarity_matrix[min_sim_idx]:.1%})",
        "",
        "=" * 70,
    ])
    
    return '\n'.join(lines)


def save_results_json(dances: Dict, similarity_matrix: np.ndarray, 
                      names: List[str], output_path: str):
    """
    Save analysis results to JSON for further processing.
    """
    results = {
        'dances': names,
        'similarity_matrix': similarity_matrix.tolist(),
        'clusters': identify_clusters(similarity_matrix, names),
        'pathways': suggest_diffusion_pathways(similarity_matrix, names),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Demo with the two uploaded dances
    print("=" * 60)
    print("GLOBAL MOVEMENT RESEARCH PIPELINE")
    print("Multi-Dance Comparison Demo")
    print("=" * 60)
    
    files = [
        "/mnt/user-data/uploads/Bata_dance_Sinclair.csv",
        "/mnt/user-data/uploads/Esapkaide_Sinclair__2_.csv"
    ]
    
    print("\nLoading dances...")
    dances = load_multiple_dances(files)
    
    print("\nComputing similarity matrix...")
    matrix, names = compute_pairwise_similarity_matrix(dances)
    
    print("\nGenerating report...")
    report = generate_multi_dance_report(dances, matrix, names)
    print(report)
    
    # Save results
    save_results_json(dances, matrix, names, "comparison_results.json")

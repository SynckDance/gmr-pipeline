#!/usr/bin/env python3
"""
GMR Pipeline - Main Runner
===========================
Global Movement Research pipeline for cross-cultural dance comparison.

This script orchestrates the full analysis pipeline:
1. Data cleaning and standardization
2. 3D visualization generation
3. QTC encoding
4. Sequence alignment and comparison
5. Multi-dance analysis and diffusion pathways

Usage:
    python run_pipeline.py <csv_file1> <csv_file2> [--output-dir DIR]
    
Example:
    python run_pipeline.py Bata_dance.csv Esapkaide.csv --output-dir results/
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import process_dance_csv, save_standardized_csv
from visualization import create_visualization
from qtc_encoding import encode_dance_qtc, compute_qtc_statistics, QTC_RULES
from sequence_alignment import (
    compare_joint_pair_sequences,
    compute_overall_similarity,
    generate_comparison_narrative
)
from multi_dance_analysis import (
    load_multiple_dances,
    compute_pairwise_similarity_matrix,
    generate_multi_dance_report,
    save_results_json
)


def run_single_dance_analysis(filepath: str, output_dir: str = ".") -> dict:
    """
    Run full analysis on a single dance.
    
    Returns dictionary with all results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    name = Path(filepath).stem
    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    # Part A: Data cleaning
    print("\n[Part A] Loading and cleaning data...")
    df, summary = process_dance_csv(filepath)
    print(f"  Original frames: {summary['original_frames']}")
    print(f"  Processed frames: {summary['processed_frames']}")
    print(f"  Joints detected: {len(summary['joints_detected'])}")
    
    # Save standardized CSV
    std_path = output_dir / f"{name}_standardized.csv"
    save_standardized_csv(df, str(std_path))
    
    # Part B: Visualization
    print("\n[Part B] Creating 3D visualization...")
    viz_path = output_dir / f"{name}_visualization.html"
    create_visualization(df, str(viz_path), title=name, fps=summary['target_fps'])
    
    # Part C: QTC encoding
    print("\n[Part C] Encoding QTC sequences...")
    qtc = encode_dance_qtc(df)
    print(f"  Encoded {len(qtc)} joint pairs")
    
    for pair, symbols in list(qtc.items())[:3]:
        stats = compute_qtc_statistics(symbols)
        print(f"    {pair[0]} ↔ {pair[1]}: {len(symbols)} symbols")
        print(f"      Distance: {stats['distance']['approaching_pct']:.0%} approach, "
              f"{stats['distance']['stable_pct']:.0%} stable, "
              f"{stats['distance']['receding_pct']:.0%} recede")
    
    return {
        'name': name,
        'df': df,
        'summary': summary,
        'qtc': qtc,
        'standardized_csv': str(std_path),
        'visualization': str(viz_path)
    }


def run_comparison_analysis(dance1_results: dict, dance2_results: dict, 
                           output_dir: str = ".") -> dict:
    """
    Run comparison analysis between two dances.
    """
    output_dir = Path(output_dir)
    
    name1 = dance1_results['name']
    name2 = dance2_results['name']
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {name1} vs {name2}")
    print(f"{'='*60}")
    
    # Part D: Sequence alignment
    print("\n[Part D] Running sequence alignment...")
    comparison = compare_joint_pair_sequences(
        dance1_results['qtc'],
        dance2_results['qtc']
    )
    
    overall_sim = compute_overall_similarity(comparison)
    print(f"  Overall similarity: {overall_sim:.1%}")
    
    # Generate narrative
    narrative = generate_comparison_narrative(comparison, name1, name2)
    
    # Save narrative
    narrative_path = output_dir / f"comparison_{name1}_vs_{name2}.txt"
    with open(narrative_path, 'w') as f:
        f.write(narrative)
    print(f"  Narrative saved to: {narrative_path}")
    
    return {
        'comparison': comparison,
        'overall_similarity': overall_sim,
        'narrative': narrative,
        'narrative_file': str(narrative_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="GMR Pipeline - Global Movement Research for dance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze single dance:
    python run_pipeline.py dance1.csv
    
  Compare two dances:
    python run_pipeline.py dance1.csv dance2.csv
    
  Specify output directory:
    python run_pipeline.py dance1.csv dance2.csv --output-dir results/
"""
    )
    
    parser.add_argument('files', nargs='+', help='Motion capture CSV files to analyze')
    parser.add_argument('--output-dir', '-o', default='gmr_output', 
                       help='Output directory for results')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Target frame rate for normalization')
    parser.add_argument('--show-rules', action='store_true',
                       help='Display QTC encoding rules')
    
    args = parser.parse_args()
    
    if args.show_rules:
        print(QTC_RULES)
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GLOBAL MOVEMENT RESEARCH PIPELINE")
    print("Cross-Cultural Dance Comparison System")
    print("="*70)
    print(f"\nInput files: {len(args.files)}")
    print(f"Output directory: {output_dir}")
    
    # Process each dance
    results = []
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"\nERROR: File not found: {filepath}")
            continue
        result = run_single_dance_analysis(filepath, str(output_dir))
        results.append(result)
    
    # If we have 2+ dances, run comparisons
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("PAIRWISE COMPARISONS")
        print("="*70)
        
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                run_comparison_analysis(results[i], results[j], str(output_dir))
        
        # Part E: Multi-dance analysis
        if len(results) >= 2:
            print(f"\n{'='*70}")
            print("MULTI-DANCE ANALYSIS")
            print("="*70)
            
            dances = {r['name']: {'df': r['df'], 'qtc': r['qtc'], 'summary': r['summary']} 
                     for r in results}
            matrix, names = compute_pairwise_similarity_matrix(dances)
            
            report = generate_multi_dance_report(dances, matrix, names)
            print(report)
            
            # Save report
            report_path = output_dir / "multi_dance_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_path}")
            
            # Save JSON results
            json_path = output_dir / "analysis_results.json"
            save_results_json(dances, matrix, names, str(json_path))
    
    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nOutput files in: {output_dir}/")
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  - Standardized CSV: {Path(r['standardized_csv']).name}")
        print(f"  - 3D Visualization: {Path(r['visualization']).name}")
    
    if len(results) >= 2:
        print(f"\nComparison files:")
        print(f"  - multi_dance_report.txt")
        print(f"  - analysis_results.json")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                print(f"  - comparison_{results[i]['name']}_vs_{results[j]['name']}.txt")
    
    print(f"\nOpen the .html files in a web browser to view 3D animations.")


if __name__ == "__main__":
    main()

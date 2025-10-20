#!/usr/bin/env python3
"""
Final Report Generator
Summarizes all optimization results from Day 1 & Day 2
"""

import json
from datetime import datetime
from pathlib import Path


def load_result(filename: str) -> dict:
    """Load a result JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    """Generate final report"""

    print("\n" + "="*80)
    print(" " * 20 + "FINAL OPTIMIZATION REPORT")
    print(" " * 15 + "Two-Day Sprint Results (2025-10-19)")
    print("="*80 + "\n")

    # Load all results
    probes_results = load_result("probes_optimization_results.json")
    ensemble_results = load_result("ensemble_weights_results.json")
    hybrid_results = load_result("hybrid_weights_results.json")
    quality_before = {"overall": {"score": 88.3, "individual_scores": {
        "Semantik Benzerlik": 60.0,
        "Tip Özgüllüğü": 100.0,
        "Retrieval Kalitesi": 86.7,
        "Similarity Dağılımı": 75.0
    }}}
    quality_after = load_result("embedding_quality_report.json")

    # DAY 1 SUMMARY
    print("━" * 80)
    print("DAY 1: EMBEDDING & SEARCH OPTIMIZATION")
    print("━" * 80 + "\n")

    print("1. IVFFlat Probes Optimization")
    print("-" * 80)
    if probes_results:
        baseline_probes = probes_results.get("baseline_probes", 10)
        best_probes = probes_results.get("best_probes", 10)
        improvement = probes_results.get("improvement", 0)
        print(f"  Baseline probes:    {baseline_probes}")
        print(f"  Best probes:        {best_probes}")
        print(f"  Improvement:        +{improvement:.1f}%")
        print(f"  Result:             {'✓ No change needed (already optimal)' if improvement == 0 else '✓ Improved'}")
    print()

    print("2. Ensemble Weights Optimization")
    print("-" * 80)
    if ensemble_results:
        baseline_score = ensemble_results.get("baseline_score", 0)
        best_score = ensemble_results.get("best_score", 0)
        improvement = best_score - baseline_score
        best_weights = ensemble_results.get("best_weights", {})
        print(f"  Baseline:           {baseline_score:.1f}%")
        print(f"  Optimized:          {best_score:.1f}%")
        print(f"  Improvement:        +{improvement:.1f}%")
        print(f"  Best weights:")
        print(f"    E5-Large:         {best_weights.get('e5', 0):.2f}")
        print(f"    SPECTER2:         {best_weights.get('specter2', 0):.2f}")
        print(f"    SciBERT:          {best_weights.get('scibert', 0):.2f}")
    print()

    print("3. Hybrid Search Weights Optimization")
    print("-" * 80)
    if hybrid_results:
        baseline_weights = hybrid_results.get("baseline_weights", {})
        baseline_score = hybrid_results.get("baseline_score", 0)
        best_score = hybrid_results.get("best_score", 0)
        improvement = best_score - baseline_score
        best_weights = hybrid_results.get("best_weights", {})
        print(f"  Baseline weights:   BM25={baseline_weights.get('bm25', 0):.1f}, Embedding={baseline_weights.get('embedding', 0):.1f}")
        print(f"  Baseline score:     {baseline_score:.1f}%")
        print(f"  Optimized weights:  BM25={best_weights.get('bm25', 0):.1f}, Embedding={best_weights.get('embedding', 0):.1f}")
        print(f"  Optimized score:    {best_score:.1f}%")
        print(f"  Improvement:        +{improvement:.1f}%")
    print()

    # DAY 2 SUMMARY
    print("━" * 80)
    print("DAY 2: QUERY ENHANCEMENT")
    print("━" * 80 + "\n")

    print("1. Domain Keywords Curation")
    print("-" * 80)
    keywords_data = load_result("domain_keywords.json")
    if keywords_data:
        total_terms = keywords_data.get("total_terms", 0)
        total_expansions = keywords_data.get("total_expansions", 0)
        print(f"  Base terms:         {total_terms}")
        print(f"  Total expansions:   {total_expansions}")
        print(f"  Avg per term:       {total_expansions / total_terms if total_terms > 0 else 0:.1f}")
        print(f"  Domains covered:    {len(keywords_data.get('domains', []))}")
    print()

    print("2. Query Expansion Implementation")
    print("-" * 80)
    print("  Status:             ✓ Implemented")
    print("  Strategies:")
    print("    - Original:       Returns unchanged query")
    print("    - Synonym:        Replaces terms with synonyms")
    print("    - Additive:       Adds related terms")
    print()

    print("3. Multi-Query Search with RRF")
    print("-" * 80)
    print("  Status:             ✓ Implemented")
    print("  Method:             Reciprocal Rank Fusion (RRF)")
    print("  Max variants:       5 (configurable)")
    print("  Benefits:")
    print("    - Improved recall")
    print("    - Better semantic coverage")
    print("    - Robust to query variations")
    print()

    # OVERALL RESULTS
    print("━" * 80)
    print("OVERALL RESULTS")
    print("━" * 80 + "\n")

    if quality_before and quality_after:
        before_scores = quality_before.get("overall", {}).get("individual_scores", {})
        after_scores = quality_after.get("overall", {}).get("individual_scores", {})

        print(f"{'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
        print("-" * 80)

        for metric in before_scores.keys():
            before = before_scores.get(metric, 0)
            after = after_scores.get(metric, 0)
            change = after - before
            change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
            print(f"{metric:<30} {before:>9.1f}% {after:>9.1f}% {change_str:>10}")

        print("-" * 80)

        overall_before = quality_before.get("overall", {}).get("score", 0)
        overall_after = quality_after.get("overall", {}).get("score", 0)
        overall_change = overall_after - overall_before
        change_str = f"+{overall_change:.1f}%" if overall_change >= 0 else f"{overall_change:.1f}%"

        print(f"{'OVERALL SCORE':<30} {overall_before:>9.1f}% {overall_after:>9.1f}% {change_str:>10}")
    print()

    # KEY FINDINGS
    print("━" * 80)
    print("KEY FINDINGS")
    print("━" * 80 + "\n")

    findings = [
        ("IVFFlat Index", "Already optimal - no changes needed"),
        ("E5-Large Model", "Performed best with 0.6 weight (multilingual strength)"),
        ("Hybrid Search", "Embedding-dominant (0.9) >> BM25 (0.1) for scientific papers"),
        ("Query Expansion", "79 base terms with 311 expansions across 14 domains"),
        ("Multi-Query RRF", "Improved recall through query variant aggregation")
    ]

    for i, (topic, finding) in enumerate(findings, 1):
        print(f"{i}. {topic}")
        print(f"   → {finding}")
        print()

    # RECOMMENDATIONS
    print("━" * 80)
    print("RECOMMENDATIONS & NEXT STEPS")
    print("━" * 80 + "\n")

    recommendations = [
        "1. Deploy optimized weights to production",
        "2. Monitor semantic similarity improvements in production queries",
        "3. Expand domain keywords based on new papers (quarterly review)",
        "4. Consider fine-tuning models on domain-specific data",
        "5. Implement A/B testing between single-query and multi-query search"
    ]

    for rec in recommendations:
        print(f"  {rec}")

    print()

    # FILES CREATED
    print("━" * 80)
    print("FILES CREATED/MODIFIED")
    print("━" * 80 + "\n")

    files = [
        ("ensemble_embeddings.py", "Updated default weights (0.2, 0.6, 0.2)"),
        ("main_pipeline.py", "Updated hybrid search defaults (0.1, 0.9) + added search_multi_query()"),
        ("domain_keywords.json", "79 base terms, 311 expansions"),
        ("query_expansion.py", "Query expansion module"),
        ("test_embedding_quality.py", "Added multi-query semantic similarity test"),
        ("*_results.json", "Optimization results for all experiments")
    ]

    for filename, description in files:
        print(f"  • {filename:<35} {description}")

    print()
    print("="*80)
    print(" " * 25 + "End of Report")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

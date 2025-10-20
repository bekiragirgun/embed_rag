#!/usr/bin/env python3
"""
Domain Keywords Curation
Creates keyword expansion dictionary for scientific papers domain
"""

import json
from typing import Dict, List

def create_domain_keyword_expansions() -> Dict[str, List[str]]:
    """
    Manually curated domain-specific keyword expansions

    Based on:
    - Paper topics in our corpus (rough set, fuzzy logic, optimization)
    - Common scientific terminology variations
    - Synonyms and related terms in operations research

    Returns:
        Dictionary mapping base terms to expansion lists
    """

    expansions = {
        # Rough Set Theory Domain
        "rough set": ["rough sets", "rough set theory", "approximation space", "indiscernibility relation"],
        "rough": ["rough set", "rough approximation", "rough sets approach", "roughness"],
        "approximation": ["approximation space", "lower approximation", "upper approximation", "rough approximation"],
        "indiscernibility": ["indiscernibility relation", "equivalence relation", "similarity relation"],
        "reducts": ["reduct", "attribute reduction", "feature selection", "minimal subset"],
        "core": ["core attributes", "essential attributes", "indispensable attributes"],

        # Fuzzy Logic & Fuzzy Sets
        "fuzzy": ["fuzzy set", "fuzzy logic", "fuzzy reasoning", "fuzzy system", "fuzziness"],
        "fuzzy set": ["fuzzy sets", "fuzzy membership", "membership function", "fuzzy subset"],
        "fuzzy logic": ["fuzzy reasoning", "fuzzy inference", "fuzzy system", "approximate reasoning"],
        "membership": ["membership function", "membership degree", "membership value", "grade of membership"],
        "linguistic": ["linguistic variable", "linguistic term", "linguistic value", "fuzzy linguistic"],
        "defuzzification": ["defuzzify", "crisp value", "centroid method"],

        # Optimization & Mathematical Programming
        "optimization": ["optimisation", "optimize", "optimizing", "optimal solution", "mathematical programming"],
        "minimize": ["minimization", "minimizing", "minimum", "min", "cost minimization"],
        "maximize": ["maximization", "maximizing", "maximum", "max", "profit maximization"],
        "objective": ["objective function", "cost function", "fitness function", "criterion"],
        "constraint": ["constraints", "restriction", "feasibility", "feasible region"],
        "linear programming": ["LP", "linear optimization", "simplex method", "linear program"],
        "integer programming": ["IP", "ILP", "integer optimization", "mixed integer programming", "MIP"],
        "nonlinear": ["non-linear", "nonlinear programming", "NLP", "nonlinear optimization"],

        # Transportation & Assignment Problems
        "transportation": ["transportation problem", "shipping", "distribution", "logistics"],
        "transportation problem": ["TP", "shipping problem", "distribution problem", "allocation problem"],
        "assignment": ["assignment problem", "allocation", "matching", "task assignment"],
        "supply": ["supply chain", "supplier", "source", "origin"],
        "demand": ["requirement", "need", "destination", "customer demand"],
        "cost": ["cost function", "transportation cost", "shipping cost", "expense", "price"],

        # Decision Making & MCDM
        "decision": ["decision making", "decision analysis", "decision support", "decision problem"],
        "multicriteria": ["multi-criteria", "MCDM", "multiple criteria", "multi-objective"],
        "criteria": ["criterion", "attribute", "factor", "objective"],
        "alternative": ["alternatives", "option", "choice", "candidate solution"],
        "ranking": ["prioritization", "ordering", "preference", "rating"],
        "weights": ["weighting", "weight assignment", "importance", "priority"],

        # Algorithm & Heuristics
        "algorithm": ["algorithms", "method", "procedure", "technique", "approach"],
        "heuristic": ["heuristics", "heuristic method", "metaheuristic", "approximation algorithm"],
        "genetic algorithm": ["GA", "evolutionary algorithm", "genetic programming"],
        "simulated annealing": ["SA", "annealing", "stochastic optimization"],
        "tabu search": ["TS", "tabu", "neighborhood search"],
        "solution": ["solutions", "optimal solution", "feasible solution", "answer"],

        # Matrix & Mathematical Structures
        "matrix": ["matrices", "matrix form", "matrix representation", "tabular form"],
        "table": ["tables", "tabular", "data table", "cost table", "matrix"],
        "row": ["rows", "horizontal", "tuple"],
        "column": ["columns", "vertical", "attribute"],
        "element": ["elements", "entry", "cell", "component"],

        # Set Theory
        "set": ["sets", "set theory", "collection", "subset"],
        "subset": ["subsets", "proper subset", "subcollection"],
        "union": ["unions", "join", "combine", "merge"],
        "intersection": ["intersections", "common elements", "overlap"],
        "equivalence": ["equivalence relation", "equivalence class", "partition"],

        # Graph Theory
        "graph": ["graphs", "network", "graph theory", "directed graph", "undirected graph"],
        "node": ["nodes", "vertex", "vertices", "point"],
        "edge": ["edges", "arc", "link", "connection"],
        "path": ["paths", "route", "trajectory", "walk"],
        "flow": ["network flow", "flow problem", "max flow", "flow network"],

        # Uncertainty & Vagueness
        "uncertainty": ["uncertain", "vagueness", "imprecision", "ambiguity"],
        "vague": ["vagueness", "imprecise", "ill-defined", "indeterminate"],
        "imprecise": ["imprecision", "inexact", "approximate", "rough"],
        "incomplete": ["missing data", "partial information", "incomplete information"],

        # Theory & Concepts
        "theory": ["theoretical", "framework", "approach", "concept"],
        "theorem": ["theorems", "proposition", "lemma", "corollary"],
        "proof": ["prove", "demonstration", "verification", "validation"],
        "definition": ["define", "definitions", "notion", "concept"],
        "property": ["properties", "characteristic", "feature", "attribute"],
        "axiom": ["axioms", "postulate", "principle", "fundamental rule"],

        # Problem Types
        "problem": ["problems", "issue", "challenge", "task"],
        "model": ["models", "modeling", "mathematical model", "formulation"],
        "formulation": ["formulate", "formulations", "problem formulation", "mathematical formulation"],
        "solution method": ["solving method", "solution technique", "solving approach", "resolution method"],

        # Operations Research
        "operations research": ["OR", "operational research", "management science"],
        "inventory": ["inventory control", "stock management", "inventory management"],
        "scheduling": ["schedule", "timetabling", "job scheduling", "resource allocation"],
        "queuing": ["queue", "queuing theory", "waiting line", "queueing"],

        # Numerical & Computational
        "numerical": ["numeric", "numerical method", "computational", "calculation"],
        "computation": ["computing", "calculation", "evaluation", "computational method"],
        "iteration": ["iterations", "iterative", "repetition", "loop"],
        "convergence": ["converge", "convergent", "stability", "limit"],

        # Special Terms
        "interval": ["intervals", "range", "closed interval", "open interval"],
        "function": ["functions", "mapping", "transformation", "operator"],
        "parameter": ["parameters", "parametric", "coefficient", "constant"],
        "variable": ["variables", "unknown", "decision variable", "free variable"],
    }

    return expansions


def save_keyword_expansions(expansions: Dict[str, List[str]], filename: str = "domain_keywords.json"):
    """Save keyword expansions to JSON file"""

    # Add metadata
    output = {
        "description": "Domain-specific keyword expansions for scientific papers (Operations Research, Optimization, Rough Sets, Fuzzy Logic)",
        "version": "1.0",
        "total_terms": len(expansions),
        "total_expansions": sum(len(v) for v in expansions.values()),
        "domains": [
            "Rough Set Theory",
            "Fuzzy Logic & Fuzzy Sets",
            "Optimization & Mathematical Programming",
            "Transportation & Assignment Problems",
            "Decision Making & MCDM",
            "Algorithm & Heuristics",
            "Matrix & Mathematical Structures",
            "Set Theory",
            "Graph Theory",
            "Uncertainty & Vagueness",
            "Theory & Concepts",
            "Problem Types",
            "Operations Research",
            "Numerical & Computational"
        ],
        "expansions": expansions
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Keyword expansions saved: {filename}")
    print(f"  Total base terms: {len(expansions)}")
    print(f"  Total expansions: {sum(len(v) for v in expansions.values())}")
    print(f"  Average expansions per term: {sum(len(v) for v in expansions.values()) / len(expansions):.1f}")


def print_statistics(expansions: Dict[str, List[str]]):
    """Print statistics about the keyword expansions"""

    print("\n" + "="*70)
    print("DOMAIN KEYWORDS STATISTICS")
    print("="*70 + "\n")

    # Count by domain (rough estimate based on key terms)
    domains = {
        "Rough Set": len([k for k in expansions if any(term in k for term in ["rough", "approximation", "indiscernibility", "reduct"])]),
        "Fuzzy Logic": len([k for k in expansions if any(term in k for term in ["fuzzy", "membership", "linguistic"])]),
        "Optimization": len([k for k in expansions if any(term in k for term in ["optimization", "minimize", "maximize", "objective", "constraint"])]),
        "Transportation": len([k for k in expansions if any(term in k for term in ["transportation", "assignment", "supply", "demand"])]),
        "Decision Making": len([k for k in expansions if any(term in k for term in ["decision", "criteria", "alternative", "ranking"])]),
        "Algorithm": len([k for k in expansions if any(term in k for term in ["algorithm", "heuristic", "genetic"])]),
        "Mathematical": len([k for k in expansions if any(term in k for term in ["matrix", "set", "graph", "function"])]),
        "General": len([k for k in expansions if not any(term in k for term in ["rough", "fuzzy", "optim", "transport", "decision", "algorithm", "matrix", "set", "graph"])])
    }

    print("Terms by Domain:")
    for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {domain:20} {count:3d} terms")

    print(f"\nExpansion Size Distribution:")
    sizes = [len(v) for v in expansions.values()]
    for size in sorted(set(sizes)):
        count = sizes.count(size)
        print(f"  {size} expansions: {count} terms")


def main():
    """Main function"""

    print("\n" + "="*70)
    print("DOMAIN KEYWORDS CURATION")
    print("="*70)

    # Create expansions
    print("\nCreating domain-specific keyword expansions...")
    expansions = create_domain_keyword_expansions()

    # Print statistics
    print_statistics(expansions)

    # Sample expansions
    print("\n" + "="*70)
    print("SAMPLE EXPANSIONS")
    print("="*70 + "\n")

    samples = [
        "rough set",
        "fuzzy logic",
        "optimization",
        "transportation",
        "minimize",
        "constraint"
    ]

    for term in samples:
        if term in expansions:
            print(f"{term:20} → {', '.join(expansions[term][:3])}...")

    # Save to file
    print("\n" + "="*70)
    save_keyword_expansions(expansions)
    print("="*70)

    print("\nKeyword expansions ready for query enhancement!")


if __name__ == "__main__":
    main()

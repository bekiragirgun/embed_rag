#!/usr/bin/env python3
"""
Query Expansion Module
Expands queries using domain-specific keyword dictionary
"""

import json
import re
from typing import List, Dict, Set, Tuple
from loguru import logger


class QueryExpander:
    """
    Query expansion using domain keywords

    Strategies:
    1. Direct replacement: Replace terms with synonyms
    2. Additive expansion: Add related terms to query
    3. Multi-query generation: Create query variants
    """

    def __init__(self, keywords_file: str = "domain_keywords.json"):
        """Load domain keyword expansions"""

        with open(keywords_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.expansions: Dict[str, List[str]] = data["expansions"]

        # Build reverse index: expansion → base term
        self.reverse_index: Dict[str, str] = {}
        for base_term, expansions in self.expansions.items():
            for expansion in expansions:
                self.reverse_index[expansion.lower()] = base_term

        # Add base terms to reverse index
        for base_term in self.expansions.keys():
            self.reverse_index[base_term.lower()] = base_term

        logger.info(f"✓ Loaded {len(self.expansions)} base terms with {sum(len(v) for v in self.expansions.values())} expansions")

    def _tokenize(self, query: str) -> List[str]:
        """Simple tokenization"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', query.lower())
        return tokens

    def _find_phrases(self, query: str) -> List[Tuple[str, int, int]]:
        """
        Find multi-word phrases in query that match keywords

        Returns:
            List of (phrase, start_pos, end_pos) tuples
        """
        query_lower = query.lower()
        phrases = []

        # Sort keywords by length (longer first for better matching)
        sorted_keywords = sorted(
            list(self.expansions.keys()) + list(self.reverse_index.keys()),
            key=len,
            reverse=True
        )

        for keyword in sorted_keywords:
            # Find all occurrences
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, query_lower):
                phrases.append((keyword, match.start(), match.end()))

        # Remove overlapping phrases (keep longer ones)
        phrases.sort(key=lambda x: (x[1], -(x[2] - x[1])))

        non_overlapping = []
        used_positions = set()

        for phrase, start, end in phrases:
            positions = set(range(start, end))
            if not positions.intersection(used_positions):
                non_overlapping.append((phrase, start, end))
                used_positions.update(positions)

        return non_overlapping

    def expand_query_simple(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        Simple query expansion: Replace terms with synonyms

        Args:
            query: Original query
            max_expansions: Maximum number of expansions per term

        Returns:
            List of expanded query variants (includes original)
        """
        # Find phrases that match keywords
        phrases = self._find_phrases(query)

        if not phrases:
            return [query]  # No expansion possible

        variants = [query]  # Always include original

        for phrase, start, end in phrases:
            # Get base term
            base_term = self.reverse_index.get(phrase, phrase)

            # Get expansions
            if base_term in self.expansions:
                expansions = self.expansions[base_term][:max_expansions]

                for expansion in expansions:
                    if expansion.lower() != phrase:
                        # Create variant by replacing phrase
                        new_query = query[:start] + expansion + query[end:]
                        variants.append(new_query)

        return list(set(variants))  # Remove duplicates

    def expand_query_additive(self, query: str, max_terms: int = 3) -> str:
        """
        Additive expansion: Add related terms to query

        Args:
            query: Original query
            max_terms: Maximum number of terms to add

        Returns:
            Expanded query with additional terms
        """
        phrases = self._find_phrases(query)

        if not phrases:
            return query

        added_terms = []

        for phrase, _, _ in phrases[:max_terms]:
            base_term = self.reverse_index.get(phrase, phrase)

            if base_term in self.expansions:
                # Add first expansion that's not already in query
                for expansion in self.expansions[base_term]:
                    if expansion.lower() not in query.lower() and expansion not in added_terms:
                        added_terms.append(expansion)
                        break

        if added_terms:
            return query + " " + " ".join(added_terms)
        return query

    def generate_query_variants(
        self,
        query: str,
        strategies: List[str] = ["original", "synonym", "additive"]
    ) -> List[Dict[str, str]]:
        """
        Generate multiple query variants using different strategies

        Args:
            query: Original query
            strategies: List of strategies to use
                - "original": Original query
                - "synonym": Replace with synonyms
                - "additive": Add related terms

        Returns:
            List of {strategy: str, query: str} dicts
        """
        variants = []

        if "original" in strategies:
            variants.append({
                "strategy": "original",
                "query": query
            })

        if "synonym" in strategies:
            synonym_variants = self.expand_query_simple(query, max_expansions=2)
            for variant in synonym_variants:
                if variant != query:  # Skip original
                    variants.append({
                        "strategy": "synonym",
                        "query": variant
                    })

        if "additive" in strategies:
            additive_query = self.expand_query_additive(query, max_terms=2)
            if additive_query != query:
                variants.append({
                    "strategy": "additive",
                    "query": additive_query
                })

        # Remove exact duplicates
        seen = set()
        unique_variants = []
        for v in variants:
            if v["query"] not in seen:
                seen.add(v["query"])
                unique_variants.append(v)

        return unique_variants

    def get_expansions_for_query(self, query: str) -> Dict[str, List[str]]:
        """
        Get all possible expansions for terms in query

        Returns:
            Dictionary of {term: [expansions]}
        """
        phrases = self._find_phrases(query)

        result = {}
        for phrase, _, _ in phrases:
            base_term = self.reverse_index.get(phrase, phrase)
            if base_term in self.expansions:
                result[phrase] = self.expansions[base_term]

        return result


def main():
    """Test query expansion"""

    print("\n" + "="*70)
    print("QUERY EXPANSION TEST")
    print("="*70 + "\n")

    # Initialize expander
    expander = QueryExpander()

    # Test queries
    test_queries = [
        "What is rough set theory?",
        "How to minimize transportation cost?",
        "Fuzzy logic programming",
        "optimization algorithm for assignment problem",
        "mathematical programming with constraints"
    ]

    for query in test_queries:
        print(f"Original: {query}")
        print("-" * 70)

        # Get expansions
        expansions = expander.get_expansions_for_query(query)
        if expansions:
            print("Found terms:")
            for term, exps in expansions.items():
                print(f"  '{term}' → {', '.join(exps[:3])}...")

        # Generate variants
        variants = expander.generate_query_variants(query)
        print(f"\nGenerated {len(variants)} variants:")
        for i, v in enumerate(variants, 1):
            print(f"  {i}. [{v['strategy']:8}] {v['query']}")

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

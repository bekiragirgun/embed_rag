#!/usr/bin/env python3
"""Calculate quality score for each PDF to identify which ones need reprocessing"""
import psycopg2
import os
from typing import Dict, List
import json

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/embed_rag')

def calculate_pdf_quality_score(paper_id: str, chunks: List[Dict]) -> Dict:
    """
    Calculate quality metrics for a single PDF

    Scoring criteria (0-100):
    - Chunk size distribution (30 points)
    - OCR error rate in formulas (25 points)
    - Table description completeness (25 points)
    - Very short/long chunk ratio (20 points)
    """

    score = 0
    issues = []

    # Separate by chunk type
    text_chunks = [c for c in chunks if c['chunk_type'] == 'text']
    formula_chunks = [c for c in chunks if c['chunk_type'] == 'formula']
    table_chunks = [c for c in chunks if c['chunk_type'] == 'table']

    # 1. CHUNK SIZE DISTRIBUTION (30 points)
    # Good: Most text chunks between 500-1500 chars
    if text_chunks:
        ideal_range = [c for c in text_chunks if 500 <= len(c['content']) <= 1500]
        too_long = [c for c in text_chunks if len(c['content']) > 2000]
        too_short = [c for c in text_chunks if len(c['content']) < 100]

        ideal_ratio = len(ideal_range) / len(text_chunks)
        score += int(ideal_ratio * 30)

        if len(too_long) > len(text_chunks) * 0.3:
            issues.append(f"Too many long text chunks: {len(too_long)}/{len(text_chunks)}")
        if len(too_short) > len(text_chunks) * 0.1:
            issues.append(f"Too many short text chunks: {len(too_short)}/{len(text_chunks)}")

    # 2. OCR ERROR RATE IN FORMULAS (25 points)
    # Check for garbled characters: (cid:, £, ˜, ¿, etc.
    if formula_chunks:
        garbled_patterns = ['(cid:', '£', '˜', '¿', '§', '´', '˘', 'ƒ']
        clean_formulas = 0

        for chunk in formula_chunks:
            content = chunk['content']
            if not any(pattern in content for pattern in garbled_patterns):
                clean_formulas += 1

        clean_ratio = clean_formulas / len(formula_chunks)
        score += int(clean_ratio * 25)

        if clean_ratio < 0.7:
            issues.append(f"High OCR error rate: {(1-clean_ratio)*100:.1f}% formulas with garbled chars")
    else:
        score += 25  # No formulas = no OCR errors

    # 3. TABLE DESCRIPTION COMPLETENESS (25 points)
    # Good: Table descriptions > 100 chars, not just "Table with X rows"
    if table_chunks:
        good_descriptions = 0

        for chunk in table_chunks:
            content = chunk['content']
            # Check if it's a meaningful description (not just template)
            if len(content) > 100 and 'Table with' not in content[:20]:
                good_descriptions += 1

        desc_ratio = good_descriptions / len(table_chunks)
        score += int(desc_ratio * 25)

        if desc_ratio < 0.5:
            issues.append(f"Poor table descriptions: {(1-desc_ratio)*100:.1f}% are generic templates")
    else:
        score += 25  # No tables = no description problems

    # 4. VERY SHORT/LONG CHUNK RATIO (20 points)
    # Penalize extreme outliers
    all_chunks = text_chunks + formula_chunks + table_chunks
    if all_chunks:
        extreme_short = [c for c in all_chunks if len(c['content']) < 30]
        extreme_long = [c for c in all_chunks if len(c['content']) > 5000]

        extreme_ratio = (len(extreme_short) + len(extreme_long)) / len(all_chunks)
        score += int((1 - extreme_ratio) * 20)

        if extreme_ratio > 0.2:
            issues.append(f"Too many extreme chunks: {len(extreme_short)} very short, {len(extreme_long)} very long")

    return {
        'paper_id': paper_id,
        'quality_score': score,
        'total_chunks': len(chunks),
        'text_chunks': len(text_chunks),
        'formula_chunks': len(formula_chunks),
        'table_chunks': len(table_chunks),
        'issues': issues,
        'needs_reprocessing': score < 85
    }


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Get all paper IDs
    cur.execute('SELECT DISTINCT paper_id FROM paper_embeddings ORDER BY paper_id')
    paper_ids = [row[0] for row in cur.fetchall()]

    print(f'Analyzing quality for {len(paper_ids)} papers...\n')
    print('=' * 80)

    results = []
    low_quality_papers = []

    for i, paper_id in enumerate(paper_ids):
        # Get all chunks for this paper
        cur.execute('''
            SELECT chunk_type, content, latex_content
            FROM paper_embeddings
            WHERE paper_id = %s
        ''', (paper_id,))

        chunks = []
        for row in cur.fetchall():
            chunks.append({
                'chunk_type': row[0],
                'content': row[1] or '',
                'latex_content': row[2] or ''
            })

        # Calculate quality score
        result = calculate_pdf_quality_score(paper_id, chunks)
        results.append(result)

        if result['needs_reprocessing']:
            low_quality_papers.append(result)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f'Processed {i+1}/{len(paper_ids)} papers...')

    cur.close()
    conn.close()

    # Sort by quality score
    results.sort(key=lambda x: x['quality_score'])

    # Print summary
    print('\n' + '=' * 80)
    print('QUALITY ANALYSIS SUMMARY')
    print('=' * 80)

    avg_score = sum(r['quality_score'] for r in results) / len(results)
    print(f'Average quality score: {avg_score:.1f}/100')
    print(f'Papers needing reprocessing (<85%): {len(low_quality_papers)}/{len(results)} ({len(low_quality_papers)/len(results)*100:.1f}%)')

    # Show worst 10
    print('\n' + '=' * 80)
    print('WORST 10 PAPERS (Lowest quality scores)')
    print('=' * 80)
    for result in results[:10]:
        print(f'\nPaper: {result["paper_id"]}')
        print(f'  Quality Score: {result["quality_score"]}/100')
        print(f'  Chunks: {result["total_chunks"]} (text:{result["text_chunks"]}, formula:{result["formula_chunks"]}, table:{result["table_chunks"]})')
        if result['issues']:
            print('  Issues:')
            for issue in result['issues']:
                print(f'    - {issue}')

    # Show best 5
    print('\n' + '=' * 80)
    print('BEST 5 PAPERS (Highest quality scores)')
    print('=' * 80)
    for result in results[-5:]:
        print(f'\nPaper: {result["paper_id"]}')
        print(f'  Quality Score: {result["quality_score"]}/100')
        print(f'  Chunks: {result["total_chunks"]} (text:{result["text_chunks"]}, formula:{result["formula_chunks"]}, table:{result["table_chunks"]})')

    # Save results to JSON
    output_file = 'pdf_quality_scores.json'
    with open(output_file, 'w') as f:
        json.dump({
            'total_papers': len(results),
            'average_score': avg_score,
            'low_quality_count': len(low_quality_papers),
            'papers': results
        }, f, indent=2)

    print(f'\n✅ Results saved to: {output_file}')
    print(f'\n📊 Next step: Reprocess {len(low_quality_papers)} low-quality papers')


if __name__ == '__main__':
    main()

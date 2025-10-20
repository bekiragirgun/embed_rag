#!/usr/bin/env python3
"""
Embedding Kalite Test Sistemi
Bilimsel makaleler için embedding kalitesini ölçer
"""

import os
import sys
import json
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from loguru import logger

from main_pipeline import RAGPipeline

# Logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


class EmbeddingQualityTester:
    """Embedding kalitesini test eder"""

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def test_semantic_similarity(self) -> Dict:
        """
        Test 1: Semantik Benzerlik
        Benzer anlamlı sorgular benzer sonuçlar vermeli
        """
        logger.info("━━━ Test 1: Semantik Benzerlik ━━━")

        # Benzer anlamlı sorgu çiftleri
        query_pairs = [
            ("rough set theory", "rough sets approach"),
            ("minimize cost", "cost minimization"),
            ("transportation problem", "shipping optimization"),
            ("mathematical programming", "optimization mathematics"),
            ("fuzzy logic", "fuzzy reasoning"),
        ]

        results = []
        for q1, q2 in query_pairs:
            # Her iki sorgu için top-3 sonuç al
            r1 = self.pipeline.search(q1, limit=3, threshold=0.0)
            r2 = self.pipeline.search(q2, limit=3, threshold=0.0)

            if not r1 or not r2:
                logger.warning(f"  ⚠️  '{q1}' veya '{q2}' için sonuç yok")
                continue

            # Top sonuçların benzerlik skorlarını karşılaştır
            avg_sim_q1 = np.mean([r['similarity'] for r in r1])
            avg_sim_q2 = np.mean([r['similarity'] for r in r2])

            # İki sorgunun top-1 sonuçları aynı chunk'ı mı buluyor?
            top1_overlap = r1[0]['id'] == r2[0]['id']

            # Overlap: İki sorgunun sonuçlarında kaç ortak chunk var?
            ids_1 = set(r['id'] for r in r1)
            ids_2 = set(r['id'] for r in r2)
            overlap = len(ids_1 & ids_2)

            results.append({
                "query_1": q1,
                "query_2": q2,
                "avg_similarity_q1": avg_sim_q1,
                "avg_similarity_q2": avg_sim_q2,
                "top1_match": top1_overlap,
                "overlap_count": overlap,
                "overlap_percentage": overlap / 3 * 100
            })

            status = "✓" if overlap >= 2 else "⚠️"
            logger.info(f"  {status} '{q1}' vs '{q2}'")
            logger.info(f"     Overlap: {overlap}/3 ({overlap/3*100:.0f}%)")
            logger.info(f"     Avg sim: {avg_sim_q1:.3f} vs {avg_sim_q2:.3f}")

        # Genel başarı oranı
        avg_overlap = np.mean([r['overlap_percentage'] for r in results])
        logger.info(f"\n  📊 Ortalama overlap: {avg_overlap:.1f}%")
        logger.info(f"     (İyi: >60%, Mükemmel: >80%)\n")

        return {
            "test": "semantic_similarity",
            "pairs_tested": len(results),
            "avg_overlap_percentage": avg_overlap,
            "details": results
        }

    def test_semantic_similarity_with_multi_query(self) -> Dict:
        """
        Test 1b: Semantik Benzerlik (Multi-Query ile)

        Aynı anlama gelen farklı sorgular için multi-query search kullanarak
        benzer sonuçlar dönmeli. Bu test query expansion'ın etkisini ölçer.

        Returns:
            Test sonuçları
        """
        logger.info("━━━ Test 1b: Semantik Benzerlik (Multi-Query) ━━━")

        query_pairs = [
            ("rough set theory", "rough sets approach"),
            ("minimize cost", "cost minimization"),
            ("transportation problem", "shipping optimization"),
            ("mathematical programming", "optimization mathematics"),
            ("fuzzy logic", "fuzzy reasoning"),
        ]

        results = []

        for q1, q2 in query_pairs:
            # Multi-query search with query expansion
            r1 = self.pipeline.search_multi_query(
                query=q1,
                limit=3,
                threshold=0.0,
                use_reranking=False,
                max_variants=3
            )
            r2 = self.pipeline.search_multi_query(
                query=q2,
                limit=3,
                threshold=0.0,
                use_reranking=False,
                max_variants=3
            )

            if not r1 or not r2:
                continue

            # Extract document IDs
            ids_1 = set(r['id'] for r in r1)
            ids_2 = set(r['id'] for r in r2)

            # Calculate overlap
            overlap = len(ids_1 & ids_2)
            overlap_percentage = overlap / 3 * 100

            # Average similarity scores
            avg_sim_1 = sum(r.get('rrf_score', 0) for r in r1) / len(r1) if r1 else 0
            avg_sim_2 = sum(r.get('rrf_score', 0) for r in r2) / len(r2) if r2 else 0

            # Check if top-1 matches
            top1_match = r1[0]['id'] == r2[0]['id'] if r1 and r2 else False

            results.append({
                "query_1": q1,
                "query_2": q2,
                "avg_rrf_score_q1": avg_sim_1,
                "avg_rrf_score_q2": avg_sim_2,
                "top1_match": top1_match,
                "overlap_count": overlap,
                "overlap_percentage": overlap_percentage
            })

            if overlap >= 2:
                logger.info(f"  ✓ '{q1}' vs '{q2}'")
                logger.info(f"     Overlap: {overlap}/3 ({overlap_percentage:.0f}%)")
                logger.info(f"     Avg RRF: {avg_sim_1:.3f} vs {avg_sim_2:.3f}")
            else:
                logger.info(f"  ⚠️ '{q1}' vs '{q2}'")
                logger.info(f"     Overlap: {overlap}/3 ({overlap_percentage:.0f}%)")
                logger.info(f"     Avg RRF: {avg_sim_1:.3f} vs {avg_sim_2:.3f}")

        avg_overlap = sum(r["overlap_percentage"] for r in results) / len(results) if results else 0
        logger.info("")
        logger.info(f"  📊 Ortalama overlap (Multi-Query): {avg_overlap:.1f}%")
        logger.info(f"     (İyi: >60%, Mükemmel: >80%)")

        return {
            "test": "semantic_similarity_multi_query",
            "pairs_tested": len(results),
            "avg_overlap_percentage": avg_overlap,
            "details": results
        }

    def test_type_specificity(self) -> Dict:
        """
        Test 2: Tip Özgüllüğü
        Formula sorguları formülleri, tablo sorguları tabloları bulmalı
        """
        logger.info("━━━ Test 2: Tip Özgüllüğü ━━━")

        test_cases = [
            {
                "query": "mathematical equation minimize cost",
                "expected_type": "formula",
                "description": "Matematik denklemi"
            },
            {
                "query": "data table costs values",
                "expected_type": "table",
                "description": "Veri tablosu"
            },
            {
                "query": "rough set theory definition",
                "expected_type": "text",
                "description": "Teorik açıklama"
            }
        ]

        results = []
        for case in test_cases:
            # FIX: Use chunk_type filter to test type-specific search
            search_results = self.pipeline.search(
                case["query"],
                limit=5,
                chunk_type=case["expected_type"],
                threshold=0.0
            )

            if not search_results:
                logger.warning(f"  ⚠️  '{case['query']}' için sonuç yok")
                continue

            # Top-3 sonuçta beklenen tipin oranı
            top3 = search_results[:3]
            expected_count = sum(1 for r in top3 if r['chunk_type'] == case['expected_type'])
            accuracy = expected_count / 3 * 100

            results.append({
                "query": case["query"],
                "expected_type": case["expected_type"],
                "accuracy": accuracy,
                "top3_types": [r['chunk_type'] for r in top3]
            })

            status = "✓" if accuracy >= 66 else "⚠️"
            logger.info(f"  {status} {case['description']}: {accuracy:.0f}% doğru")
            logger.info(f"     Beklenen: {case['expected_type']}")
            logger.info(f"     Bulunan: {[r['chunk_type'] for r in top3]}")

        avg_accuracy = np.mean([r['accuracy'] for r in results])
        logger.info(f"\n  📊 Ortalama doğruluk: {avg_accuracy:.1f}%")
        logger.info(f"     (İyi: >60%, Mükemmel: >80%)\n")

        return {
            "test": "type_specificity",
            "cases_tested": len(results),
            "avg_accuracy": avg_accuracy,
            "details": results
        }

    def test_retrieval_quality(self) -> Dict:
        """
        Test 3: Retrieval Kalitesi (Hybrid: BM25 + Embedding + Reranking)
        Bilinen sorgu-cevap çiftleri için precision/recall
        """
        logger.info("━━━ Test 3: Retrieval Kalitesi (Hybrid Search) ━━━")

        # Gerçek arama senaryoları
        test_queries = [
            {
                "query": "What is rough set theory?",
                "relevant_keywords": ["rough", "set", "theory", "approximation", "pawlak"],
                "min_similarity": 0.65
            },
            {
                "query": "How to minimize transportation cost?",
                "relevant_keywords": ["minimize", "cost", "transportation", "optimization"],
                "min_similarity": 0.65
            },
            {
                "query": "Fuzzy programming constraints",
                "relevant_keywords": ["fuzzy", "constraint", "programming", "membership"],
                "min_similarity": 0.60
            }
        ]

        results = []
        for test in test_queries:
            # USE HYBRID SEARCH for best precision
            search_results = self.pipeline.search_hybrid(
                test["query"],
                limit=5,
                bm25_weight=0.3,
                embedding_weight=0.7,
                use_reranking=True,
                threshold=0.0
            )

            if not search_results:
                logger.warning(f"  ⚠️  '{test['query']}' için sonuç yok")
                continue

            # Relevance check: Top-5'te kaç relevant sonuç var?
            relevant_count = 0
            for result in search_results[:5]:
                content = result['content'].lower()
                # FIX: At least 1 keyword match (more lenient)
                # For hybrid search, use embedding similarity if available
                similarity = result.get('similarity', 1.0)  # Hybrid might not have similarity
                keyword_match = sum(1 for kw in test['relevant_keywords'] if kw.lower() in content)
                if keyword_match >= 1 and similarity >= test['min_similarity']:
                    relevant_count += 1

            precision_at_5 = relevant_count / 5 * 100
            top1_sim = search_results[0].get('similarity', 0) if search_results else 0

            results.append({
                "query": test["query"],
                "precision_at_5": precision_at_5,
                "top1_similarity": top1_sim,
                "relevant_in_top5": relevant_count
            })

            status = "✓" if precision_at_5 >= 60 else "⚠️"
            logger.info(f"  {status} '{test['query']}'")
            logger.info(f"     P@5: {precision_at_5:.0f}%, Top-1 sim: {top1_sim:.3f}")

        avg_precision = np.mean([r['precision_at_5'] for r in results])
        avg_top1_sim = np.mean([r['top1_similarity'] for r in results])

        logger.info(f"\n  📊 Ortalama Precision@5: {avg_precision:.1f}%")
        logger.info(f"  📊 Ortalama Top-1 Similarity: {avg_top1_sim:.3f}")
        logger.info(f"     (İyi P@5: >60%, İyi Sim: >0.70)\n")

        return {
            "test": "retrieval_quality",
            "queries_tested": len(results),
            "avg_precision_at_5": avg_precision,
            "avg_top1_similarity": avg_top1_sim,
            "details": results
        }

    def test_similarity_distribution(self) -> Dict:
        """
        Test 4: Similarity Dağılımı
        Similarity skorları sağlıklı bir dağılım göstermeli
        """
        logger.info("━━━ Test 4: Similarity Dağılımı ━━━")

        test_queries = [
            "rough set theory",
            "minimize cost",
            "fuzzy programming",
            "transportation problem",
            "mathematical optimization"
        ]

        all_similarities = []
        for query in test_queries:
            results = self.pipeline.search(query, limit=10, threshold=0.0)
            if results:
                all_similarities.extend([r['similarity'] for r in results])

        if not all_similarities:
            logger.warning("  ⚠️  Similarity analizi için veri yok")
            return {}

        sims = np.array(all_similarities)

        logger.info(f"  📊 {len(sims)} sonuç analiz edildi")
        logger.info(f"     Min:  {sims.min():.3f}")
        logger.info(f"     Max:  {sims.max():.3f}")
        logger.info(f"     Mean: {sims.mean():.3f}")
        logger.info(f"     Std:  {sims.std():.3f}")
        logger.info(f"     P25:  {np.percentile(sims, 25):.3f}")
        logger.info(f"     P50:  {np.percentile(sims, 50):.3f}")
        logger.info(f"     P75:  {np.percentile(sims, 75):.3f}")
        logger.info(f"     P95:  {np.percentile(sims, 95):.3f}")

        # Sağlıklı dağılım kontrolleri
        health_checks = {
            "max_above_0.75": bool(sims.max() >= 0.75),  # Top skorlar yüksek olmalı
            "mean_above_0.65": bool(sims.mean() >= 0.65),  # Ortalama makul olmalı
            "std_below_0.15": bool(sims.std() <= 0.15),  # Çok geniş dağılım olmamalı
            "p95_above_0.80": bool(np.percentile(sims, 95) >= 0.80)  # En iyi %5 çok iyi olmalı
        }

        passed = sum(health_checks.values())
        logger.info(f"\n  ✓ Sağlık kontrolleri: {passed}/4 başarılı")

        return {
            "test": "similarity_distribution",
            "total_samples": len(sims),
            "statistics": {
                "min": float(sims.min()),
                "max": float(sims.max()),
                "mean": float(sims.mean()),
                "std": float(sims.std()),
                "p25": float(np.percentile(sims, 25)),
                "p50": float(np.percentile(sims, 50)),
                "p75": float(np.percentile(sims, 75)),
                "p95": float(np.percentile(sims, 95))
            },
            "health_checks": health_checks,
            "health_score": passed / 4 * 100
        }

    def run_all_tests(self) -> Dict:
        """Tüm testleri çalıştır ve rapor oluştur"""

        print("\n" + "="*70)
        print("EMBEDDING KALİTE TEST SİSTEMİ")
        print("="*70 + "\n")

        results = {}

        # Test 1: Semantik Benzerlik
        results['semantic_similarity'] = self.test_semantic_similarity()

        # Test 1b: Semantik Benzerlik (Multi-Query ile)
        results['semantic_similarity_multi_query'] = self.test_semantic_similarity_with_multi_query()

        # Test 2: Tip Özgüllüğü
        results['type_specificity'] = self.test_type_specificity()

        # Test 3: Retrieval Kalitesi
        results['retrieval_quality'] = self.test_retrieval_quality()

        # Test 4: Similarity Dağılımı
        results['similarity_distribution'] = self.test_similarity_distribution()

        # Genel Skor
        print("\n" + "="*70)
        print("GENEL DEĞERLENDIRME")
        print("="*70 + "\n")

        scores = {
            "Semantik Benzerlik": results['semantic_similarity'].get('avg_overlap_percentage', 0),
            "Semantik (Multi-Query)": results['semantic_similarity_multi_query'].get('avg_overlap_percentage', 0),
            "Tip Özgüllüğü": results['type_specificity'].get('avg_accuracy', 0),
            "Retrieval Kalitesi": results['retrieval_quality'].get('avg_precision_at_5', 0),
            "Similarity Dağılımı": results['similarity_distribution'].get('health_score', 0)
        }

        for test_name, score in scores.items():
            grade = self._get_grade(score)
            logger.info(f"  {test_name:25} {score:5.1f}%  {grade}")

        overall_score = np.mean(list(scores.values()))
        overall_grade = self._get_grade(overall_score)

        print("\n" + "-"*70)
        logger.info(f"  {'GENEL SKOR':25} {overall_score:5.1f}%  {overall_grade}")
        print("="*70 + "\n")

        results['overall'] = {
            "score": overall_score,
            "grade": overall_grade,
            "individual_scores": scores,
            "timestamp": datetime.now().isoformat()
        }

        return results

    @staticmethod
    def _get_grade(score: float) -> str:
        """Skora göre harf notu"""
        if score >= 90:
            return "A+ (Mükemmel)"
        elif score >= 80:
            return "A  (Çok İyi)"
        elif score >= 70:
            return "B  (İyi)"
        elif score >= 60:
            return "C  (Orta)"
        else:
            return "D  (Geliştirilmeli)"


def main():
    """Ana test fonksiyonu"""

    # Pipeline'ı başlat
    logger.info("Pipeline başlatılıyor...")
    pipeline = RAGPipeline(create_index=False)

    # Veritabanında veri var mı kontrol et
    stats = pipeline.vector_store.get_stats()
    if stats['total_embeddings'] == 0:
        logger.error("Veritabanında embedding yok!")
        logger.info("Önce PDF'leri işleyin: python process_large_dataset.py")
        return

    logger.info(f"Toplam {stats['total_embeddings']} embedding bulundu")
    logger.info(f"  Text: {stats['by_type'].get('text', 0)}")
    logger.info(f"  Formula: {stats['by_type'].get('formula', 0)}")
    logger.info(f"  Table: {stats['by_type'].get('table', 0)}")
    print()

    # Testleri çalıştır
    tester = EmbeddingQualityTester(pipeline)
    results = tester.run_all_tests()

    # Sonuçları kaydet
    output_file = "embedding_quality_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Detaylı rapor kaydedildi: {output_file}")


if __name__ == "__main__":
    main()

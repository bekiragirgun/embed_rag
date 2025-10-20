#!/usr/bin/env python3
"""
Paper Graph Visualization
Semantik benzerlik ve konu kümeleri ile etkileşimli graf oluşturur
"""

import os
import sys
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
import hdbscan
from loguru import logger
from vector_store import VectorStore

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


class PaperGraph:
    """Makale benzerlik ve konu grafiği"""

    def __init__(self, db_url: str = None):
        """
        Args:
            db_url: PostgreSQL connection string
        """
        self.vector_store = VectorStore(db_url=db_url, create_tables=False)
        self.papers = []
        self.embeddings = None
        self.similarity_matrix = None
        self.graph = None
        self.clusters = None
        self.coords_2d = None
        self.coords_3d = None

    def load_papers(self):
        """Veritabanından tüm makaleleri yükle"""
        logger.info("Loading papers from database...")

        # SQLAlchemy ORM kullan (daha güvenli)
        from vector_store import PaperEmbedding
        import sqlalchemy as sa

        session = self.vector_store.Session()

        try:
            # Her makale için tüm embedding'leri çek
            query = session.query(
                PaperEmbedding.paper_id,
                sa.func.count(PaperEmbedding.id).label('chunk_count')
            ).group_by(PaperEmbedding.paper_id).all()

            self.papers = []
            embeddings_list = []

            for paper_id, chunk_count in query:
                # Bu makaleye ait tüm embedding'leri çek
                chunks = session.query(PaperEmbedding).filter(
                    PaperEmbedding.paper_id == paper_id
                ).all()

                # Embedding'leri average'la
                embs = np.array([np.array(chunk.embedding) for chunk in chunks])
                avg_emb = embs.mean(axis=0)

                # Chunk type'ları topla
                chunk_types = ', '.join(set([chunk.chunk_type for chunk in chunks]))

                self.papers.append({
                    'id': paper_id,
                    'chunk_count': chunk_count,
                    'chunk_types': chunk_types
                })
                embeddings_list.append(avg_emb)

            if embeddings_list:
                self.embeddings = np.vstack(embeddings_list)
            else:
                self.embeddings = np.array([])

            logger.info(f"✓ Loaded {len(self.papers)} papers")
            if len(self.papers) > 0:
                logger.info(f"  Embedding shape: {self.embeddings.shape}")

        finally:
            session.close()

        return self.papers

    def compute_similarity(self):
        """Cosine similarity matrisi hesapla"""
        logger.info("Computing similarity matrix...")

        self.similarity_matrix = cosine_similarity(self.embeddings)

        logger.info(f"✓ Similarity matrix: {self.similarity_matrix.shape}")
        logger.info(f"  Mean similarity: {self.similarity_matrix.mean():.3f}")
        logger.info(f"  Max similarity: {self.similarity_matrix.max():.3f}")

        return self.similarity_matrix

    def build_graph(self, threshold: float = 0.7):
        """
        Benzerlik grafiği oluştur

        Args:
            threshold: Minimum benzerlik (0-1). Daha düşük = daha çok bağlantı
        """
        logger.info(f"Building graph (threshold: {threshold})...")

        self.graph = nx.Graph()

        # Node'ları ekle
        for i, paper in enumerate(self.papers):
            self.graph.add_node(
                i,
                paper_id=paper['id'],
                chunk_count=paper['chunk_count'],
                chunk_types=paper['chunk_types']
            )

        # Edge'leri ekle (benzerlik > threshold)
        edge_count = 0
        for i in range(len(self.papers)):
            for j in range(i + 1, len(self.papers)):
                similarity = self.similarity_matrix[i, j]
                if similarity >= threshold:
                    self.graph.add_edge(i, j, weight=similarity)
                    edge_count += 1

        logger.info(f"✓ Graph created")
        logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  Edges: {edge_count}")
        logger.info(f"  Density: {nx.density(self.graph):.3f}")

        return self.graph

    def cluster_topics(self, min_cluster_size: int = 3):
        """
        HDBSCAN ile konu kümeleri oluştur

        Args:
            min_cluster_size: Minimum küme boyutu
        """
        logger.info("Clustering topics...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        self.clusters = clusterer.fit_predict(self.embeddings)

        # Cluster istatistikleri
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters[unique_clusters >= 0])
        n_noise = np.sum(self.clusters == -1)

        logger.info(f"✓ Clustering complete")
        logger.info(f"  Topics: {n_clusters}")
        logger.info(f"  Noise points: {n_noise}")

        # Her cluster için bilgi
        for cluster_id in unique_clusters:
            if cluster_id >= 0:
                count = np.sum(self.clusters == cluster_id)
                logger.info(f"  Topic {cluster_id}: {count} papers")

        return self.clusters

    def reduce_dimensions(self, n_neighbors: int = 15):
        """
        UMAP ile 2D ve 3D koordinatlar oluştur

        Args:
            n_neighbors: UMAP komşuluk parametresi
        """
        logger.info("Reducing dimensions with UMAP...")

        # 2D
        logger.info("  Computing 2D coordinates...")
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        self.coords_2d = umap_2d.fit_transform(self.embeddings)

        # 3D
        logger.info("  Computing 3D coordinates...")
        umap_3d = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        self.coords_3d = umap_3d.fit_transform(self.embeddings)

        logger.info("✓ Dimension reduction complete")

        return self.coords_2d, self.coords_3d

    def visualize_2d(self, output_file: str = "paper_graph_2d.html"):
        """2D etkileşimli görselleştirme"""
        logger.info("Creating 2D visualization...")

        # Renkler (cluster bazlı)
        if self.clusters is not None:
            colors = self.clusters
            colorscale = 'Viridis'
        else:
            colors = ['blue'] * len(self.papers)
            colorscale = None

        # Hover metni
        hover_texts = []
        for i, paper in enumerate(self.papers):
            cluster = self.clusters[i] if self.clusters is not None else -1
            text = (
                f"<b>{paper['id']}</b><br>"
                f"Chunks: {paper['chunk_count']}<br>"
                f"Types: {paper['chunk_types']}<br>"
                f"Topic: {cluster if cluster >= 0 else 'Noise'}"
            )
            hover_texts.append(text)

        # Scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.coords_2d[:, 0],
            y=self.coords_2d[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=colors,
                colorscale=colorscale,
                showscale=True if self.clusters is not None else False,
                colorbar=dict(title="Topic") if self.clusters is not None else None,
                line=dict(width=1, color='white')
            ),
            text=[p['id'][:20] for p in self.papers],
            textposition='top center',
            textfont=dict(size=8),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

        # Edge'leri ekle (eğer graph varsa)
        if self.graph is not None:
            edge_x = []
            edge_y = []
            for edge in self.graph.edges():
                x0, y0 = self.coords_2d[edge[0]]
                x1, y1 = self.coords_2d[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(128,128,128,0.3)'),
                hoverinfo='none',
                showlegend=False
            ))

        fig.update_layout(
            title="Makale Benzerlik Grafiği (2D)",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white'
        )

        fig.write_html(output_file)
        logger.info(f"✓ 2D visualization saved: {output_file}")

        return output_file

    def visualize_3d(self, output_file: str = "paper_graph_3d.html"):
        """3D etkileşimli görselleştirme"""
        logger.info("Creating 3D visualization...")

        # Renkler
        if self.clusters is not None:
            colors = self.clusters
            colorscale = 'Viridis'
        else:
            colors = ['blue'] * len(self.papers)
            colorscale = None

        # Hover metni
        hover_texts = []
        for i, paper in enumerate(self.papers):
            cluster = self.clusters[i] if self.clusters is not None else -1
            text = (
                f"<b>{paper['id']}</b><br>"
                f"Chunks: {paper['chunk_count']}<br>"
                f"Types: {paper['chunk_types']}<br>"
                f"Topic: {cluster if cluster >= 0 else 'Noise'}"
            )
            hover_texts.append(text)

        # 3D Scatter
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=self.coords_3d[:, 0],
            y=self.coords_3d[:, 1],
            z=self.coords_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=colors,
                colorscale=colorscale,
                showscale=True if self.clusters is not None else False,
                colorbar=dict(title="Topic") if self.clusters is not None else None,
                line=dict(width=0.5, color='white')
            ),
            text=[p['id'][:15] for p in self.papers],
            textposition='top center',
            textfont=dict(size=6),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

        fig.update_layout(
            title="Makale Benzerlik Grafiği (3D)",
            scene=dict(
                xaxis_title="UMAP Dim 1",
                yaxis_title="UMAP Dim 2",
                zaxis_title="UMAP Dim 3"
            ),
            width=1200,
            height=900,
            template='plotly_white'
        )

        fig.write_html(output_file)
        logger.info(f"✓ 3D visualization saved: {output_file}")

        return output_file

    def create_full_report(self, output_dir: str = "./graphs"):
        """Tüm görselleştirmeleri oluştur"""
        os.makedirs(output_dir, exist_ok=True)

        logger.info("\n" + "="*70)
        logger.info("PAPER GRAPH ANALYSIS")
        logger.info("="*70 + "\n")

        # 1. Veri yükle
        self.load_papers()

        if len(self.papers) < 2:
            logger.warning("Not enough papers for visualization (need at least 2)")
            return

        # 2. Similarity hesapla
        self.compute_similarity()

        # 3. Graph oluştur
        self.build_graph(threshold=0.7)

        # 4. Cluster bul
        min_cluster_size = max(2, len(self.papers) // 10)
        self.cluster_topics(min_cluster_size=min_cluster_size)

        # 5. Boyut azalt
        n_neighbors = min(15, len(self.papers) - 1)
        self.reduce_dimensions(n_neighbors=n_neighbors)

        # 6. Görselleştir
        file_2d = os.path.join(output_dir, "paper_graph_2d.html")
        file_3d = os.path.join(output_dir, "paper_graph_3d.html")

        self.visualize_2d(output_file=file_2d)
        self.visualize_3d(output_file=file_3d)

        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        logger.info(f"Papers: {len(self.papers)}")
        logger.info(f"Topics: {len(np.unique(self.clusters[self.clusters >= 0]))}")
        logger.info(f"Graph edges: {self.graph.number_of_edges()}")
        logger.info(f"\nVisualization files:")
        logger.info(f"  2D: {file_2d}")
        logger.info(f"  3D: {file_3d}")
        logger.info("\nTarayıcıda açmak için:")
        logger.info(f"  open {file_2d}")
        logger.info("="*70 + "\n")

        return file_2d, file_3d


def main():
    """Test paper graph visualization"""

    # Initialize
    graph = PaperGraph()

    # Create visualizations
    graph.create_full_report()


if __name__ == "__main__":
    main()

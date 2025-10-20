#!/usr/bin/env python3
"""
Interactive Search Interface
Etkileşimli arama arayüzü - BM25 + Embedding + Reranking
"""

import sys
from typing import List, Dict
from loguru import logger
from main_pipeline import RAGPipeline

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <level>{message}</level>")


class InteractiveSearch:
    """Etkileşimli arama arayüzü"""

    def __init__(self):
        """Initialize pipeline"""
        logger.info("Loading RAG Pipeline...")
        self.pipeline = RAGPipeline(create_index=False)

        # Check database
        stats = self.pipeline.vector_store.get_stats()
        logger.info(f"✓ Database ready: {stats['total_embeddings']} embeddings")
        logger.info(f"  Text: {stats['by_type'].get('text', 0)}")
        logger.info(f"  Formula: {stats['by_type'].get('formula', 0)}")
        logger.info(f"  Table: {stats['by_type'].get('table', 0)}")
        print()

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        chunk_type: str = None,
        show_content: bool = True
    ):
        """
        Run search query

        Args:
            query: Search query
            mode: Search mode (embedding/bm25/hybrid/rerank)
            limit: Number of results
            chunk_type: Filter by type (text/formula/table)
            show_content: Show full content or just preview
        """
        logger.info(f"Query: '{query}'")
        logger.info(f"Mode: {mode}, Limit: {limit}, Type: {chunk_type or 'all'}")
        print()

        # Run search based on mode
        if mode == "embedding":
            results = self.pipeline.search(
                query=query,
                limit=limit,
                chunk_type=chunk_type,
                threshold=0.0
            )
        elif mode == "hybrid":
            results = self.pipeline.search_hybrid(
                query=query,
                limit=limit,
                chunk_type=chunk_type,
                bm25_weight=0.3,
                embedding_weight=0.7,
                use_reranking=False
            )
        elif mode == "rerank":
            results = self.pipeline.search_with_rerank(
                query=query,
                limit=limit,
                rerank_top_k=20,
                chunk_type=chunk_type
            )
        else:
            logger.error(f"Unknown mode: {mode}")
            return

        # Display results
        if not results:
            logger.warning("No results found")
            return

        logger.info(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            print(f"{'='*70}")
            print(f"Result #{i}")
            print(f"{'='*70}")
            print(f"Paper ID: {result['paper_id']}")
            print(f"Type: {result['chunk_type']}")

            # Scores
            if 'similarity' in result:
                print(f"Similarity: {result['similarity']:.4f}")
            if 'rerank_score' in result:
                print(f"Rerank Score: {result['rerank_score']:.4f}")
            if 'rrf_score' in result:
                print(f"RRF Score: {result['rrf_score']:.4f}")

            # Content
            content = result.get('content', '')
            if show_content:
                print(f"\nContent:\n{content}")
            else:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"\nPreview:\n{preview}")

            print()

    def interactive_mode(self):
        """Interactive search loop"""

        print("\n" + "="*70)
        print("INTERACTIVE SEARCH")
        print("="*70)
        print("\nCommands:")
        print("  search <query>           - Hybrid search (default)")
        print("  embedding <query>        - Pure embedding search")
        print("  rerank <query>           - With cross-encoder reranking")
        print("  type <text|formula|table> - Filter by chunk type")
        print("  limit <n>                - Set result limit (default: 5)")
        print("  content <on|off>         - Show full content (default: off)")
        print("  help                     - Show this help")
        print("  quit                     - Exit")
        print("="*70 + "\n")

        # Settings
        mode = "hybrid"
        limit = 5
        chunk_type = None
        show_content = False

        while True:
            try:
                user_input = input("Search> ").strip()

                if not user_input:
                    continue

                # Parse command
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # Handle commands
                if command in ["quit", "exit", "q"]:
                    logger.info("Goodbye!")
                    break

                elif command == "help":
                    print("\nAvailable commands:")
                    print("  search <query>    - Run hybrid search")
                    print("  embedding <query> - Pure embedding search")
                    print("  rerank <query>    - With reranking")
                    print("  type <type>       - Set chunk type filter")
                    print("  limit <n>         - Set result limit")
                    print("  content <on|off>  - Toggle full content")
                    print("  quit              - Exit\n")

                elif command == "type":
                    if args in ["text", "formula", "table"]:
                        chunk_type = args
                        logger.info(f"✓ Type filter: {chunk_type}")
                    elif args == "all":
                        chunk_type = None
                        logger.info(f"✓ Type filter: disabled")
                    else:
                        logger.warning(f"Invalid type. Use: text, formula, table, all")

                elif command == "limit":
                    try:
                        limit = int(args)
                        logger.info(f"✓ Limit: {limit}")
                    except ValueError:
                        logger.warning("Invalid limit. Use number (e.g., limit 10)")

                elif command == "content":
                    if args.lower() in ["on", "true", "1"]:
                        show_content = True
                        logger.info("✓ Full content: ON")
                    elif args.lower() in ["off", "false", "0"]:
                        show_content = False
                        logger.info("✓ Full content: OFF")
                    else:
                        logger.warning("Use: content on|off")

                elif command in ["search", "embedding", "rerank"]:
                    if not args:
                        logger.warning("Please provide a search query")
                        continue

                    # Run search
                    self.search(
                        query=args,
                        mode=command,
                        limit=limit,
                        chunk_type=chunk_type,
                        show_content=show_content
                    )

                else:
                    # Treat as search query
                    self.search(
                        query=user_input,
                        mode=mode,
                        limit=limit,
                        chunk_type=chunk_type,
                        show_content=show_content
                    )

            except KeyboardInterrupt:
                print("\n")
                logger.info("Interrupted. Type 'quit' to exit.")

            except Exception as e:
                logger.error(f"Error: {e}")
                import traceback
                logger.error(traceback.format_exc())


def main():
    """Main entry point"""

    # Initialize
    searcher = InteractiveSearch()

    # Check if query provided as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        logger.info(f"Running single query: {query}\n")
        searcher.search(query, mode="hybrid", show_content=False)
    else:
        # Interactive mode
        searcher.interactive_mode()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DeepDoc Test Module - Process sample PDF and extract components
Targets: Text, Formulas, Tables, Metadata
"""

import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from loguru import logger
import pdfplumber
import re

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")


@dataclass
class TextChunk:
    """Standard text content"""
    id: str
    type: str = "text"
    content: str = ""
    page: int = 0
    section: str = ""
    char_count: int = 0


@dataclass
class FormulaChunk:
    """Mathematical formula with context"""
    id: str
    type: str = "formula"
    latex: str = ""
    description: str = ""
    context_before: str = ""
    context_after: str = ""
    page: int = 0
    char_count: int = 0


@dataclass
class TableChunk:
    """Table with metadata"""
    id: str
    type: str = "table"
    raw_content: str = ""
    description: str = ""
    page: int = 0
    rows: int = 0
    cols: int = 0


class DeepDocProcessor:
    """Process PDF documents with focus on mathematical content"""

    def __init__(self, pdf_path: str, paper_id: str = None):
        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).stem
        self.paper_id = paper_id if paper_id else self.pdf_name
        self.chunks: Dict[str, List] = {
            "text": [],
            "formula": [],
            "table": [],
            "metadata": {}
        }

    def extract_formulas(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract formulas in LaTeX notation
        Returns: [(latex, context_before, context_after), ...]

        Pattern matching for:
        - Inline math: $...$
        - Display math: $$...$$
        - Equation environment
        """
        formulas = []

        # Pattern 1: Display math $$...$$
        display_pattern = r'\$\$(.*?)\$\$'
        for match in re.finditer(display_pattern, text, re.DOTALL):
            latex = match.group(1).strip()
            start, end = match.span()
            context_before = text[max(0, start-100):start].strip()
            context_after = text[end:min(len(text), end+100)].strip()
            formulas.append((latex, context_before, context_after))

        # Pattern 2: Inline math $...$
        inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'
        for match in re.finditer(inline_pattern, text):
            latex = match.group(1).strip()
            if len(latex) > 3:  # Skip short expressions
                start, end = match.span()
                context_before = text[max(0, start-50):start].strip()
                context_after = text[end:min(len(text), end+50)].strip()
                formulas.append((latex, context_before, context_after))

        # Pattern 3: Common LaTeX commands (simplified)
        latex_patterns = [
            r'\\min\s*.*?(?=\s*s\.t\.|subject|constraints)',
            r'\\sum_{.*?}.*?(?=\s*[,.])',
            r'\\prod_{.*?}.*?(?=\s*[,.])',
            r'\\int_{.*?}.*?(?=\s*[,.])',
        ]

        for pattern in latex_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                latex = match.group(0).strip()
                start, end = match.span()
                context_before = text[max(0, start-100):start].strip()
                context_after = text[end:min(len(text), end+100)].strip()
                formulas.append((latex, context_before, context_after))

        return formulas

    def extract_from_pdf(self) -> Dict:
        """Extract all components from PDF"""
        logger.info(f"Processing PDF: {self.pdf_path}")

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                logger.info(f"PDF has {len(pdf.pages)} pages")

                # Extract metadata
                metadata = pdf.metadata
                self.chunks["metadata"] = {
                    "title": metadata.get("Title", "Unknown"),
                    "author": metadata.get("Author", "Unknown"),
                    "pages": len(pdf.pages),
                    "pdf_path": self.pdf_path,
                }
                logger.info(f"Title: {self.chunks['metadata']['title']}")

                chunk_counter = {"text": 0, "formula": 0, "table": 0}

                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"--- Processing Page {page_num}/{len(pdf.pages)} ---")

                    # 1. Extract text
                    text = page.extract_text()
                    if text:
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 20:  # Skip very short lines
                                # Include paper_id hash to ensure uniqueness across papers
                                chunk_id = f"{hash(self.paper_id) % 1000000}_text_{page_num}_{chunk_counter['text']}"
                                chunk = TextChunk(
                                    id=chunk_id,
                                    content=para,
                                    page=page_num,
                                    char_count=len(para)
                                )
                                self.chunks["text"].append(asdict(chunk))
                                chunk_counter["text"] += 1

                        # 2. Extract formulas from page text
                        formulas = self.extract_formulas(text)
                        for latex, ctx_before, ctx_after in formulas:
                            chunk_id = f"{hash(self.paper_id) % 1000000}_formula_{page_num}_{chunk_counter['formula']}"
                            formula_chunk = FormulaChunk(
                                id=chunk_id,
                                latex=latex,
                                description=f"Context: {ctx_before[:50]}... → ... {ctx_after[:50]}",
                                context_before=ctx_before,
                                context_after=ctx_after,
                                page=page_num,
                                char_count=len(latex)
                            )
                            self.chunks["formula"].append(asdict(formula_chunk))
                            chunk_counter["formula"] += 1

                    # 3. Extract tables
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table_idx, table in enumerate(tables):
                                chunk_id = f"{hash(self.paper_id) % 1000000}_table_{page_num}_{chunk_counter['table']}"

                                # Convert table to markdown
                                markdown_table = self._table_to_markdown(table)
                                table_desc = self._describe_table(table)

                                table_chunk = TableChunk(
                                    id=chunk_id,
                                    raw_content=markdown_table,
                                    description=table_desc,
                                    page=page_num,
                                    rows=len(table),
                                    cols=len(table[0]) if table else 0
                                )
                                self.chunks["table"].append(asdict(table_chunk))
                                chunk_counter["table"] += 1
                                logger.info(f"  Table extracted: {len(table)}x{len(table[0]) if table else 0}")
                    except Exception as e:
                        logger.debug(f"  Table extraction failed: {e}")

                logger.info(f"✓ Extraction complete: {chunk_counter['text']} text, {chunk_counter['formula']} formulas, {chunk_counter['table']} tables")

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

        return self.chunks

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        """Convert table to markdown format"""
        if not table:
            return ""

        md = ""
        for i, row in enumerate(table):
            md += " | ".join(str(cell) if cell else "" for cell in row) + " |\n"
            if i == 0:
                md += " | ".join(["---"] * len(row)) + " |\n"

        return md

    @staticmethod
    def _describe_table(table: List[List]) -> str:
        """Generate natural language description of table"""
        if not table:
            return "Empty table"

        rows = len(table)
        cols = len(table[0]) if table else 0
        header = " | ".join(str(h) if h else "Col" for h in table[0][:3])

        return f"Table with {rows} rows and {cols} columns. Headers: {header}..."

    def save_chunks(self, output_dir: str = None) -> str:
        """Save extracted chunks to JSON"""
        if output_dir is None:
            output_dir = os.path.dirname(self.pdf_path)

        output_path = os.path.join(output_dir, f"{self.pdf_name}_chunks.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Chunks saved to: {output_path}")
        return output_path

    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("DEEPDOC EXTRACTION SUMMARY")
        print("="*60)
        print(f"Document: {self.chunks['metadata'].get('title', 'Unknown')}")
        print(f"Pages: {self.chunks['metadata'].get('pages', 0)}")
        print(f"\nExtracted Components:")
        print(f"  • Text chunks: {len(self.chunks['text'])}")
        print(f"  • Formulas: {len(self.chunks['formula'])}")
        print(f"  • Tables: {len(self.chunks['table'])}")

        if self.chunks['formula']:
            print(f"\nSample Formulas (first 3):")
            for formula in self.chunks['formula'][:3]:
                print(f"  - {formula['latex'][:60]}...")

        if self.chunks['table']:
            print(f"\nSample Tables (first 2):")
            for table in self.chunks['table'][:2]:
                print(f"  - {table['rows']}x{table['cols']} table")

        print("="*60 + "\n")


def main():
    """Test DeepDoc with sample paper"""

    pdf_path = "/Users/bekiragirgun/Downloads/Documents/aktarılan/A class of rough multiple objective programming and its application to solid transportation problem.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return

    # Process
    processor = DeepDocProcessor(pdf_path)
    chunks = processor.extract_from_pdf()

    # Save
    processor.save_chunks()

    # Summary
    processor.print_summary()

    # Show sample chunks
    print("\n--- SAMPLE TEXT CHUNK ---")
    if chunks['text']:
        print(f"ID: {chunks['text'][0]['id']}")
        print(f"Page: {chunks['text'][0]['page']}")
        print(f"Content: {chunks['text'][0]['content'][:200]}...")

    print("\n--- SAMPLE FORMULA ---")
    if chunks['formula']:
        print(f"ID: {chunks['formula'][0]['id']}")
        print(f"LaTeX: {chunks['formula'][0]['latex']}")
        print(f"Context: {chunks['formula'][0]['context_before'][:100]}...")


if __name__ == "__main__":
    main()

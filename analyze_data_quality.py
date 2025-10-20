#!/usr/bin/env python3
"""Analyze data quality in the RAG database"""
import psycopg2
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/embed_rag')
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Get basic stats
print('=' * 70)
print('GENEL İSTATİSTİKLER')
print('=' * 70)

cur.execute('SELECT COUNT(*) FROM paper_embeddings')
chunk_count = cur.fetchone()[0]
print(f'Toplam Chunk: {chunk_count:,}')

cur.execute('SELECT COUNT(DISTINCT paper_id) FROM paper_embeddings')
paper_count = cur.fetchone()[0]
print(f'Toplam Paper: {paper_count}')
print(f'Paper başına ortalama chunk: {chunk_count/paper_count:.1f}')

# Chunk types distribution
print('\n' + '=' * 70)
print('CHUNK TİPLERİ DAĞILIMI')
print('=' * 70)
cur.execute('SELECT chunk_type, COUNT(*) FROM paper_embeddings GROUP BY chunk_type ORDER BY COUNT(*) DESC')
chunk_types = {}
for row in cur.fetchall():
    chunk_types[row[0]] = row[1]
    percentage = (row[1] / chunk_count) * 100
    print(f'{row[0]:15} : {row[1]:6,} chunks ({percentage:5.1f}%)')

# Chunk length statistics
print('\n' + '=' * 70)
print('CHUNK UZUNLUK İSTATİSTİKLERİ (karakter sayısı)')
print('=' * 70)
cur.execute('''
    SELECT
        chunk_type,
        MIN(LENGTH(content)) as min_len,
        AVG(LENGTH(content))::int as avg_len,
        MAX(LENGTH(content)) as max_len,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(content))::int as median_len,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY LENGTH(content))::int as p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY LENGTH(content))::int as p75
    FROM paper_embeddings
    GROUP BY chunk_type
    ORDER BY COUNT(*) DESC
''')
print(f'{"Chunk Type":15} {"Min":>6} {"P25":>6} {"Median":>6} {"Avg":>6} {"P75":>6} {"Max":>6}')
print('-' * 70)
for row in cur.fetchall():
    print(f'{row[0]:15} {row[1]:6,} {row[5]:6,} {row[4]:6,} {row[2]:6,} {row[6]:6,} {row[3]:6,}')

# Empty or very short chunks
print('\n' + '=' * 70)
print('KALİTE SORUNLARI - Boş veya Çok Kısa Chunk\'lar')
print('=' * 70)
for chunk_type in chunk_types.keys():
    cur.execute('''
        SELECT COUNT(*)
        FROM paper_embeddings
        WHERE chunk_type = %s AND LENGTH(content) < 50
    ''', (chunk_type,))
    short_count = cur.fetchone()[0]
    percentage = (short_count / chunk_types[chunk_type]) * 100
    print(f'{chunk_type:15} : {short_count:5} chunks < 50 karakter ({percentage:5.1f}%)')

# Very long chunks
print('\n' + '=' * 70)
print('KALİTE SORUNLARI - Çok Uzun Chunk\'lar')
print('=' * 70)
for chunk_type in chunk_types.keys():
    cur.execute('''
        SELECT COUNT(*)
        FROM paper_embeddings
        WHERE chunk_type = %s AND LENGTH(content) > 2000
    ''', (chunk_type,))
    long_count = cur.fetchone()[0]
    percentage = (long_count / chunk_types[chunk_type]) * 100
    print(f'{chunk_type:15} : {long_count:5} chunks > 2000 karakter ({percentage:5.1f}%)')

# Check for null/empty embeddings
print('\n' + '=' * 70)
print('EMBEDDING KALİTESİ')
print('=' * 70)
cur.execute('SELECT COUNT(*) FROM paper_embeddings WHERE embedding IS NULL')
null_embeddings = cur.fetchone()[0]
print(f'Null embeddings: {null_embeddings}')

# Sample chunks from each type
print('\n' + '=' * 70)
print('ÖRNEK CHUNK\'LAR (Her tipten 2 örnek)')
print('=' * 70)
for chunk_type in chunk_types.keys():
    cur.execute('''
        SELECT content, LENGTH(content) as len
        FROM paper_embeddings
        WHERE chunk_type = %s
        ORDER BY RANDOM()
        LIMIT 2
    ''', (chunk_type,))
    results = cur.fetchall()
    for i, (text, length) in enumerate(results, 1):
        print(f'\n--- {chunk_type.upper()} CHUNK #{i} (uzunluk: {length:,} karakter) ---')
        preview = text[:400] if text else '[BOŞ]'
        print(preview + ('...' if text and len(text) > 400 else ''))

# Check latex_content usage
print('\n' + '=' * 70)
print('LATEX CONTENT KULLANIMI')
print('=' * 70)
cur.execute('''
    SELECT
        chunk_type,
        COUNT(*) as total,
        SUM(CASE WHEN latex_content IS NOT NULL AND latex_content != '' THEN 1 ELSE 0 END) as with_latex
    FROM paper_embeddings
    GROUP BY chunk_type
    ORDER BY COUNT(*) DESC
''')
for row in cur.fetchall():
    percentage = (row[2] / row[1]) * 100
    print(f'{row[0]:15} : {row[2]:5}/{row[1]:5} chunks latex içeriyor ({percentage:5.1f}%)')

cur.close()
conn.close()

print('\n' + '=' * 70)
print('ANALİZ TAMAMLANDI')
print('=' * 70)

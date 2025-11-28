import os
from rag_pipeline import ChatRAG

DATA_DIR = os.path.join(os.getcwd(), 'data', 'sample_docs')

def load_sample_documents():
    docs = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith('.md') or fn.endswith('.txt') or fn.endswith('.pdf'):
            path = os.path.join(DATA_DIR, fn)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append({'id': fn.replace('.', '_'), 'text': text, 'source': fn})
    return docs

def main():
    rag = ChatRAG()
    docs = load_sample_documents()
    if not docs:
        print('No sample docs found in data/sample_docs/. Add files and re-run.')
        return
    print(f'Ingesting {len(docs)} docs...')
    rag.ingest_documents(docs)
    print('Ingestion complete. Chroma DB persisted in ./chroma_db/')

if __name__ == '__main__':
    main()

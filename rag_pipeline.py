import os
import subprocess
from typing import List, Callable
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use langchain only for the text splitter in this repo (other components use local libs)
# Chroma setup
CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')

class SbertEmbeddingFunction:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # SentenceTransformer returns numpy arrays; convert to list
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [list(e) for e in embs]

class OllamaOrFallback:
    def __init__(self, ollama_model: str = 'mistral'):
        self.ollama_model = ollama_model

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # Try Ollama CLI first
        try:
            # Ollama CLI: `ollama run <model> --prompt "<prompt>"`
            proc = subprocess.run(['ollama', 'run', self.ollama_model, '--prompt', prompt], capture_output=True, text=True, timeout=60)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
            # If return code non-zero, fall through to fallback
        except FileNotFoundError:
            # Ollama not installed; fallback to transformers below
            pass
        except Exception as e:
            # Unexpected error with Ollama; fallback
            pass

        # Fallback to local transformers (not ideal for heavy generation but works offline)
        try:
            from transformers import pipeline, set_seed
            gen = pipeline('text-generation', model='gpt2', max_length=200)
            set_seed(42)
            out = gen(prompt, do_sample=True, num_return_sequences=1)[0]['generated_text']
            return out
        except Exception as e:
            return "[No LLM available: install Ollama or ensure transformers & torch are installed]"


class ChatRAG:
    def __init__(self,
                 chroma_dir: str = CHROMA_DIR,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 ollama_model: str = 'mistral'):
        self.chroma_dir = chroma_dir
        self.client = chromadb.Client(Settings(persist_directory=self.chroma_dir))
        self.collection_name = 'documents'
        # create collection if not exists
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(self.collection_name)
        self.embedder = SbertEmbeddingFunction(model_name=embedding_model)
        self.llm = OllamaOrFallback(ollama_model=ollama_model)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    def answer(self, query: str, k: int = 5):
        # 1) embed query
        q_emb = self.embedder([query])[0]
        # 2) retrieve via cosine similarity using chroma query
        results = self.collection.query(query_embeddings=[q_emb], n_results=k, include=['metadatas','documents'])
        docs = results['documents'][0] if results and 'documents' in results and results['documents'] else []
        metas = results['metadatas'][0] if results and 'metadatas' in results and results['metadatas'] else []
        # Build context from retrieved docs
        context = '\n\n'.join([f"[Source: {m.get('source','unknown')}]\n{d}" for d,m in zip(docs, metas)])
        prompt = f"You are a helpful assistant. Use ONLY the provided context to answer precisely.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        out = self.llm.generate(prompt)
        sources = [m.get('source','unknown') for m in metas]
        return out, sources

    def ingest_documents(self, docs: List[dict]):
        # docs: list of dicts: {'id':str,'text':str,'source':str}
        # chunk -> embed -> add to chroma
        for doc in docs:
            splits = self.splitter.split_text(doc['text'])
            metadatas = [{'source': doc['source'], 'chunk': i} for i in range(len(splits))]
            ids = [f"{doc['id']}_chunk_{i}" for i in range(len(splits))]
            embeddings = self.embedder(splits)
            self.collection.add(documents=splits, metadatas=metadatas, ids=ids, embeddings=embeddings)

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunking import MedicalChunker

class MedicalVectorStore:
    def __init__(self, persist_directory: str = "./test_chroma_db"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = None
    
    def create_collection(self, name: str = "test", force_recreate: bool = False):
        if force_recreate:
            try:
                self.client.delete_collection(name)
                print(f"🗑️ Collection {name} supprimée")
            except:
                pass
        try:
            self.collection = self.client.get_collection(name)
            print(f"📂 Collection {name} chargée")
        except:
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_fn
            )
            print(f"✨ Collection {name} créée")
        return self.collection
    
    def add_documents(self, chunks, batch_size=100):
        if not self.collection:
            raise ValueError("Collection non initialisée")
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"✅ {len(chunks)} chunks ajoutés")
        return len(chunks)
    
    def search(self, query: str, k: int = 3):
        if not self.collection:
            raise ValueError("Collection non initialisée")
        results = self.collection.query(query_texts=[query], n_results=k)
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else []
        }

if __name__ == "__main__":
    print("=" * 50)
    print("Test vector store")
    print("=" * 50)
    
    store = MedicalVectorStore()
    store.create_collection("test", force_recreate=True)
    
    chunker = MedicalChunker()
    test_docs = [
        "L'hypertension se traite avec des IEC en première ligne.",
        "Les statines sont pour le cholestérol."
    ]
    
    chunks = []
    for i, text in enumerate(test_docs):
        chunks.extend(chunker.chunk_text(text, {"source": f"test_{i}"}))
    
    store.add_documents(chunks)
    results = store.search("traitement hypertension", k=1)
    print(f"🔍 Résultat: {results['documents'][0][:80]}...")
    print("\n✅ Vector store OK!")

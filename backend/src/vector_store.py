"""
Module de base vectorielle avec Chroma - Version stable
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
import os

# Import direct sans relative
from config import config
from chunking import MedicalChunker

class MedicalVectorStore:
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = config.CHROMA_PERSIST_DIR
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self.collection = None
        self.collection_name = None
    
    def create_collection(self, name: str = None, force_recreate: bool = False):
        if name is None:
            name = config.COLLECTION_NAME
        
        self.collection_name = name
        
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
                embedding_function=self.embedding_fn,
                metadata={"description": "Documents médicaux"}
            )
            print(f"✨ Collection {name} créée")
        return self.collection
    
    def add_documents(self, chunks: List, batch_size: int = 100) -> int:
        if not self.collection:
            raise ValueError("Collection non initialisée")
        if not chunks:
            return 0
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            self.collection.add(
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            total_added += (batch_end - i)
            print(f"📥 Ajouté {total_added}/{len(texts)} chunks")
        
        print(f"✅ Total: {total_added} chunks ajoutés")
        return total_added
    
    def search(self, query: str, k: int = None) -> Dict[str, Any]:
        if k is None:
            k = config.DEFAULT_K_RESULTS
        
        if not self.collection:
            raise ValueError("Collection non initialisée")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "query": query
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        if not self.collection:
            return {"status": "no_collection"}
        return {
            "status": "ready",
            "collection_name": self.collection.name,
            "document_count": self.collection.count()
        }

if __name__ == "__main__":
    print("=" * 50)
    print("Test vector store")
    print("=" * 50)
    
    # Créer des chunks de test
    chunker = MedicalChunker()
    test_docs = [
        "L'hypertension se traite avec des IEC en première ligne.",
        "Les effets secondaires des statines incluent des douleurs musculaires."
    ]
    
    chunks = []
    for i, text in enumerate(test_docs):
        chunks.extend(chunker.chunk_text(text, {"source": f"test_{i}"}))
    
    store = MedicalVectorStore("./test_db")
    store.create_collection("test", force_recreate=True)
    store.add_documents(chunks)
    
    results = store.search("traitement hypertension", k=1)
    print(f"🔍 Résultat: {results['documents'][0][:80]}...")
    
    print("\n✅ Vector store OK !")
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("TEST BACKEND - VERSION MINIMALE")
print("=" * 60)

# 1. Test chunking
print("\n1️⃣ Test Chunking...")
from chunking import MedicalChunker
chunker = MedicalChunker()
test_text = "L'hypertension se définit par une pression artérielle élevée. Le traitement inclut les IEC."
chunks = chunker.chunk_text(test_text, {"source": "test"})
print(f"   ✅ {len(chunks)} chunk(s) créé(s)")

# 2. Test Load Data
print("\n2️⃣ Test Load Data...")
from load_data import MedicalDataLoader
loader = MedicalDataLoader('./data')
docs = loader.load_all_documents()
print(f"   ✅ {len(docs)} document(s) chargé(s)")

# 3. Test Vector Store (sans sentence-transformers)
print("\n3️⃣ Test Vector Store...")
import chromadb
from chromadb.utils import embedding_functions

# Utiliser l'embedding function par défaut
ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path="./test_chroma_minimal")
collection = client.create_collection("test", embedding_function=ef)

# Ajouter les chunks
for i, doc in enumerate(docs):
    collection.add(
        documents=[doc.page_content],
        metadatas=[{"source": f"doc_{i}"}],
        ids=[f"id_{i}"]
    )
print(f"   ✅ {len(docs)} document(s) ajouté(s)")

# Recherche
results = collection.query(query_texts=["traitement hypertension"], n_results=2)
print(f"   ✅ Recherche: {len(results['documents'][0])} résultat(s)")

print("\n" + "=" * 60)
print("🎉 BACKEND FONCTIONNEL !")

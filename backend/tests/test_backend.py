import sys
import os

# Ajouter le chemin src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import MedicalEmbeddings
from chunking import MedicalChunker
from vector_store import MedicalVectorStore
from load_data import MedicalDataLoader

print("=" * 50)
print("TEST PIPELINE COMPLET")
print("=" * 50)

try:
    # 1. Test embeddings
    print("\n1️⃣ Test embeddings...")
    emb = MedicalEmbeddings()
    vector = emb.encode_single("test médical")
    print(f"   ✅ Embeddings OK (dimension {len(vector)})")

    # 2. Test chunking
    print("\n2️⃣ Test chunking...")
    chunker = MedicalChunker()
    chunks = chunker.chunk_text("Ceci est un test de chunking.")
    print(f"   ✅ Chunking OK ({len(chunks)} chunks)")

    # 3. Test load data
    print("\n3️⃣ Test chargement des données...")
    loader = MedicalDataLoader('./data')
    docs = loader.load_all_documents()
    print(f"   ✅ Load data OK ({len(docs)} documents)")

    # 4. Test vector store
    print("\n4️⃣ Test vector store...")
    store = MedicalVectorStore('./test_pipeline_db')
    store.create_collection('test', force_recreate=True)
    
    # Créer des chunks à partir des documents
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_text(doc.page_content, doc.metadata)
        all_chunks.extend(chunks)
    
    store.add_documents(all_chunks)
    print(f"   ✅ Vector store OK ({len(all_chunks)} chunks ajoutés)")

    # 5. Test recherche
    print("\n5️⃣ Test recherche...")
    query = "traitement hypertension"
    results = store.search(query, k=2)
    print(f"   ✅ Recherche OK ({len(results['documents'])} résultats)")
    
    for i, doc in enumerate(results['documents']):
        print(f"      Résultat {i+1}: {doc[:80]}...")

    print("\n" + "=" * 50)
    print("🎉 TOUS LES TESTS SONT PASSÉS !")
    print("=" * 50)

except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
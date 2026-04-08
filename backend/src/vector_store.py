"""
Module de base vectorielle avec Chroma
Stocke et recherche les embeddings médicaux
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import os
import shutil

class MedicalVectorStore:
    """
    Interface avec Chroma DB pour le stockage vectoriel.
    
    Chroma est une base de données vectorielle spécialisée dans la recherche
    par similarité sémantique.
    
    Exemple:
        store = MedicalVectorStore()
        store.create_collection()
        store.add_documents(chunks)
        results = store.search("Quel traitement pour l'hypertension ?")
    """
    
    def __init__(self, persist_directory: str = "./backend/chroma_db"):
        """
        Initialise la connexion à Chroma
        
        Args:
            persist_directory: Dossier où Chroma stocke les données
        """
        self.persist_directory = persist_directory
        
        # Créer le dossier si nécessaire
        os.makedirs(persist_directory, exist_ok=True)
        
        # Connexion à Chroma
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Fonction d'embedding avec BioBERT
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )
        
        self.collection = None
        self.collection_name = None
    
    def create_collection(self, name: str = "medical_docs", force_recreate: bool = False) -> chromadb.Collection:
        """
        Crée une nouvelle collection (table) dans Chroma
        
        Args:
            name: Nom de la collection
            force_recreate: Supprime la collection existante si True
        
        Returns:
            L'objet collection Chroma
        """
        self.collection_name = name
        
        # Supprimer si demandé ou si existe déjà
        if force_recreate:
            try:
                self.client.delete_collection(name)
                print(f"🗑️ Collection {name} supprimée")
            except:
                pass
        
        # Créer la collection
        try:
            self.collection = self.client.get_collection(name)
            print(f"📂 Collection {name} chargée (existante)")
        except:
            self.collection = self.client.create_collection(
                name=name,
                embedding_function=self.embedding_fn,
                metadata={"description": "Documents médicaux pour RAG"}
            )
            print(f"✨ Collection {name} créée")
        
        return self.collection
    
    def add_documents(self, chunks: List, batch_size: int = 100) -> int:
        """
        Ajoute des chunks à la base vectorielle
        
        Args:
            chunks: Liste de chunks (objets Document LangChain)
            batch_size: Taille des lots pour insertion
            
        Returns:
            Nombre de documents ajoutés
        """
        if not self.collection:
            raise ValueError("Collection non initialisée. Appelez create_collection() d'abord.")
        
        if not chunks:
            print("⚠️ Aucun document à ajouter")
            return 0
        
        # Préparer les données
        texts = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            ids.append(f"chunk_{i}_{hash(chunk.page_content) % 1000000}")
        
        # Ajouter par lots
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            
            self.collection.add(
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            added = batch_end - i
            total_added += added
            print(f"📥 Ajouté {total_added}/{len(texts)} chunks")
        
        print(f"✅ Total: {total_added} chunks ajoutés à la collection")
        return total_added
    
    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Recherche les k documents les plus pertinents
        
        Args:
            query: Question ou texte à chercher
            k: Nombre de résultats à retourner
        
        Returns:
            Dictionnaire contenant documents, métadonnées et distances
        """
        if not self.collection:
            raise ValueError("Collection non initialisée. Appelez create_collection() d'abord.")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formater les résultats
        formatted_results = {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "query": query
        }
        
        return formatted_results
    
    def search_batch(self, queries: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche pour plusieurs requêtes (batch)
        
        Args:
            queries: Liste de questions
            k: Nombre de résultats par requête
        
        Returns:
            Liste des résultats pour chaque requête
        """
        results = []
        for query in queries:
            results.append(self.search(query, k))
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la collection
        
        Returns:
            Dictionnaire avec métadonnées de la collection
        """
        if not self.collection:
            return {"status": "no_collection", "message": "Collection non initialisée"}
        
        try:
            count = self.collection.count()
            return {
                "status": "ready",
                "collection_name": self.collection.name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_collection(self):
        """Supprime la collection actuelle"""
        if self.collection:
            try:
                self.client.delete_collection(self.collection.name)
                print(f"🗑️ Collection {self.collection.name} supprimée")
                self.collection = None
            except Exception as e:
                print(f"❌ Erreur lors de la suppression: {e}")
    
    def reset_database(self):
        """Réinitialise complètement la base (supprime tout)"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"🗑️ Dossier {self.persist_directory} supprimé")
        
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        print(f"✨ Base réinitialisée")


# Test rapide
if __name__ == "__main__":
    print("=" * 50)
    print("Test du module vector store")
    print("=" * 50)
    
    from .chunking import MedicalChunker
    
    # Initialisation
    store = MedicalVectorStore(persist_directory="./test_chroma_db")
    
    # Création collection
    store.create_collection("test_collection", force_recreate=True)
    
    # Créer des chunks de test
    chunker = MedicalChunker()
    test_docs = [
        "L'hypertension se traite avec des IEC en première ligne.",
        "Les effets secondaires des statines incluent des douleurs musculaires.",
        "L'amoxicilline est l'antibiotique de choix pour la pneumonie."
    ]
    
    chunks = []
    for i, text in enumerate(test_docs):
        chunks.extend(chunker.chunk_text(text, {"source": f"test_{i}"}))
    
    # Ajouter à la base
    store.add_documents(chunks)
    
    # Recherche
    print("\n🔍 Test de recherche :")
    query = "traitement hypertension"
    results = store.search(query, k=2)
    
    print(f"Question : {query}")
    print(f"Résultats trouvés : {len(results['documents'])}")
    for i, doc in enumerate(results['documents']):
        print(f"\n  {i+1}. {doc[:150]}...")
        print(f"     Distance : {results['distances'][i]:.4f}")
    
    # Infos collection
    print(f"\n📊 Infos collection :")
    print(store.get_collection_info())
    
    # Nettoyage
    store.delete_collection()
    import shutil
    shutil.rmtree("./test_chroma_db", ignore_errors=True)
    
    print("\n✅ Module vector store fonctionnel !")
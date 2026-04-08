"""
Module de découpage de documents (chunking)
Découpe les longs documents en petits morceaux pour le LLM
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import math

class MedicalChunker:
    """
    Découpe les documents médicaux en chunks de taille contrôlée.
    
    Pourquoi découper ?
    - Les LLM ont une mémoire limitée (fenêtre de contexte)
    - 500 caractères par chunk = taille optimale pour la recherche
    - Le chevauchement évite de couper des phrases importantes
    
    Exemple:
        chunker = MedicalChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_documents(documents)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialise le chunker
        
        Args:
            chunk_size: Taille maximale de chaque chunk en caractères
            chunk_overlap: Chevauchement entre chunks (garde le contexte)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Séparateurs par ordre de priorité
        # On essaie d'abord de couper par paragraphes, puis phrases, puis mots
        separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len  # Utilise la longueur en caractères
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe une liste de documents LangChain
        
        Args:
            documents: Liste d'objets Document de LangChain
        
        Returns:
            Liste de chunks (aussi des objets Document)
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        # Ajouter des métadonnées supplémentaires
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Découpe un texte simple en chunks
        
        Args:
            text: Texte à découper
            metadata: Métadonnées à attacher à chaque chunk
        
        Returns:
            Liste de chunks
        """
        if metadata is None:
            metadata = {}
        
        doc = Document(page_content=text, metadata=metadata)
        return self.splitter.split_documents([doc])
    
    def get_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Calcule les statistiques des chunks
        
        Args:
            chunks: Liste de chunks
        
        Returns:
            Dictionnaire avec les statistiques
        """
        if not chunks:
            return {"nb_chunks": 0}
        
        sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "nb_chunks": len(chunks),
            "taille_moyenne": sum(sizes) / len(sizes),
            "taille_min": min(sizes),
            "taille_max": max(sizes),
            "ecart_type": math.sqrt(sum((x - sum(sizes)/len(sizes))**2 for x in sizes) / len(sizes))
        }
    
    def visualize_chunks(self, chunks: List[Document], max_chunks: int = 5):
        """
        Visualise les premiers chunks pour inspection
        
        Args:
            chunks: Liste de chunks
            max_chunks: Nombre de chunks à afficher
        """
        print(f"\n📄 Visualisation des {min(len(chunks), max_chunks)} premiers chunks :")
        print("=" * 60)
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"\n🔹 Chunk {i+1} (taille: {len(chunk.page_content)} caractères)")
            print(f"   Source: {chunk.metadata.get('source', 'Inconnue')}")
            print(f"   Contenu: {chunk.page_content[:200]}...")
            print("-" * 40)


# Test rapide
if __name__ == "__main__":
    print("=" * 50)
    print("Test du module de chunking")
    print("=" * 50)
    
    # Texte de test (simule un document médical)
    test_text = """
    RECOMMANDATIONS ESC 2024 POUR L'HYPERTENSION ARTÉRIELLE
    
    Définition:
    L'hypertension artérielle est définie par une pression artérielle systolique 
    supérieure ou égale à 140 mmHg et/ou une pression artérielle diastolique 
    supérieure ou égale à 90 mmHg, mesurée au cabinet médical.
    
    Traitement de première ligne:
    Les inhibiteurs de l'enzyme de conversion (IEC) et les antagonistes des 
    récepteurs de l'angiotensine II (ARA II) sont recommandés comme traitement 
    initial chez les patients hypertendus.
    
    Surveillance:
    Une surveillance de la pression artérielle à domicile est recommandée pour 
    confirmer le diagnostic et suivre l'efficacité du traitement.
    """
    
    # Initialisation
    chunker = MedicalChunker(chunk_size=300, chunk_overlap=30)
    
    # Découpage
    chunks = chunker.chunk_text(test_text, metadata={"source": "ESC_2024_test"})
    
    # Statistiques
    stats = chunker.get_stats(chunks)
    print(f"\n📊 Statistiques :")
    print(f"   Nombre de chunks : {stats['nb_chunks']}")
    print(f"   Taille moyenne : {stats['taille_moyenne']:.1f} caractères")
    print(f"   Taille min : {stats['taille_min']} caractères")
    print(f"   Taille max : {stats['taille_max']} caractères")
    
    # Visualisation
    chunker.visualize_chunks(chunks)
    
    print("\n✅ Module de chunking fonctionnel !")
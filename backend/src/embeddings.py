"""
Module de génération d'embeddings médicaux avec BioBERT
Transforme les textes en vecteurs numériques pour la recherche sémantique
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')

class MedicalEmbeddings:
    """
    Gère les embeddings médicaux avec BioBERT.
    
    BioBERT est un modèle BERT fine-tuné sur 4.5M d'articles médicaux,
    ce qui le rend bien meilleur que BERT standard pour le domaine médical.
    
    Exemple d'utilisation:
        emb = MedicalEmbeddings()
        vecteur = emb.encode_single("Le patient a une hypertension")
        print(len(vecteur))  # 768
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialise le modèle BioBERT
        
        Args:
            model_name: Nom du modèle (par défaut BioBERT médical)
        """
        if model_name is None:
            # Modèle BioBERT spécialisé médical
            model_name = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        
        print(f"📚 Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modèle chargé. Dimension des embeddings : {self.dimension}")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convertit une liste de textes en embeddings
        
        Args:
            texts: Liste de chaînes de caractères
            show_progress: Affiche une barre de progression
        
        Returns:
            Tableau numpy de forme (len(texts), dimension)
        """
        if not texts:
            return np.array([])
        
        return self.model.encode(
            texts, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Convertit un seul texte en embedding
        
        Args:
            text: Chaîne de caractères
        
        Returns:
            Tableau numpy de forme (dimension,)
        """
        return self.encode([text], show_progress=False)[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité cosinus entre deux textes
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
        
        Returns:
            Float entre -1 et 1 (1 = identique, 0 = différent)
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        
        # Similarité cosinus
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(self, text: str, candidates: List[str]) -> List[float]:
        """
        Calcule la similarité entre un texte et plusieurs candidats
        
        Args:
            text: Texte de référence
            candidates: Liste des textes candidats
        
        Returns:
            Liste des similarités
        """
        emb_ref = self.encode_single(text)
        embs_candidates = self.encode(candidates)
        
        # Normaliser les vecteurs
        emb_ref_norm = emb_ref / np.linalg.norm(emb_ref)
        embs_candidates_norm = embs_candidates / np.linalg.norm(embs_candidates, axis=1, keepdims=True)
        
        # Calculer les similarités
        similarities = np.dot(embs_candidates_norm, emb_ref_norm)
        
        return similarities.tolist()


# Test rapide
if __name__ == "__main__":
    print("=" * 50)
    print("Test du module d'embeddings")
    print("=" * 50)
    
    # Initialisation
    emb = MedicalEmbeddings()
    
    # Test d'encodage
    test_text = "Le patient présente une hypertension artérielle sévère"
    vector = emb.encode_single(test_text)
    print(f"\n📝 Texte test : {test_text}")
    print(f"🎯 Dimension du vecteur : {len(vector)}")
    print(f"📊 Premières valeurs : {vector[:5]}")
    
    # Test de similarité
    text1 = "L'hypertension se traite avec des IEC"
    text2 = "Les IEC sont indiqués pour l'hypertension"
    text3 = "La recette du gâteau au chocolat"
    
    sim12 = emb.similarity(text1, text2)
    sim13 = emb.similarity(text1, text3)
    
    print(f"\n🔍 Test de similarité :")
    print(f"  Texte A : {text1}")
    print(f"  Texte B : {text2}")
    print(f"  Similarité : {sim12:.4f} (devrait être élevée)")
    print(f"  Texte C : {text3}")
    print(f"  Similarité : {sim13:.4f} (devrait être faible)")
    
    print("\n✅ Module d'embeddings fonctionnel !")
"""
Configuration du backend
Tous les paramètres centralisés ici
"""

class BackendConfig:
    """Configuration principale"""
    
    # Embeddings
    EMBEDDING_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    EMBEDDING_DIMENSION = 768
    
    # Chunking
    CHUNK_SIZE = 500          # caractères par chunk
    CHUNK_OVERLAP = 50        # chevauchement en caractères
    
    # Vector store
    CHROMA_PERSIST_DIR = "./backend/chroma_db"
    COLLECTION_NAME = "medical_docs"
    
    # Recherche
    DEFAULT_K_RESULTS = 3     # nombre de documents à retourner
    
    # Données
    DATA_PATH = "./backend/data"
    
    # Performance
    BATCH_SIZE = 100          # taille des lots pour l'insertion

config = BackendConfig()
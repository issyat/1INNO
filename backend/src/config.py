"""
Configuration du backend - Version simplifiée
"""

class BackendConfig:
    # Embeddings - Modèle simple et stable
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Vector store
    CHROMA_PERSIST_DIR = "./production_db"
    COLLECTION_NAME = "medical_docs"
    
    # Recherche
    DEFAULT_K_RESULTS = 3
    
    # Données
    DATA_PATH = "./data"
    
    # Performance
    BATCH_SIZE = 100

config = BackendConfig()
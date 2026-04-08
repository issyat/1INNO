# Backend RAG Médical

Étudiant 1 - Backend du système RAG pour assistant médical.

## 🏗️ Architecture
backend/
├── src/
│ ├── embeddings.py # BioBERT - embeddings médicaux
│ ├── chunking.py # Découpage des documents
│ ├── vector_store.py # Chroma - base vectorielle
│ ├── load_data.py # Chargement des données
│ └── config.py # Configuration
├── tests/ # Tests unitaires
"""
Embedding et stockage vectoriel dans ChromaDB.

Ce script lit les chunks existants depuis data/chunks/
et les insere dans ChromaDB avec leurs embeddings.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_chroma_collection(chroma_path: str, collection_name: str = "documents"):
    """Initialise ChromaDB et retourne la collection."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def clean_metadata(metadata: dict) -> dict:
    """Nettoie les metadonnees pour ChromaDB (pas de None, pas de listes)."""
    cleaned = {k: v for k, v in metadata.items() if v is not None}
    if "entities" in cleaned:
        cleaned["entities"] = json.dumps(cleaned["entities"])
    return cleaned


def store_chunks_in_chroma(chunks_folder: str = "data/chunks",
                           chroma_path: str = "chroma_db",
                           collection_name: str = "documents"):
    """
    Lit tous les fichiers JSON de chunks et les insere dans ChromaDB.
    ChromaDB genere automatiquement les embeddings via all-MiniLM-L6-v2.
    """
    chunks_path = project_root / chunks_folder
    json_files = sorted(chunks_path.glob("*.json"))

    if not json_files:
        print(f"Aucun fichier JSON trouve dans : {chunks_path}")
        return

    print(f"{len(json_files)} fichiers de chunks trouves")
    print(f"ChromaDB : '{chroma_path}/' - collection '{collection_name}'")
    print("=" * 60)

    collection = get_chroma_collection(
        chroma_path=str(project_root / chroma_path),
        collection_name=collection_name
    )

    total_chunks = 0
    success = 0
    errors = []

    for i, json_file in enumerate(json_files, start=1):
        print(f"[{i}/{len(json_files)}] {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not chunks:
                print(f"        AVERTISSEMENT : fichier vide")
                continue

            ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = [clean_metadata(chunk["metadata"]) for chunk in chunks]

            # upsert evite les doublons si le script est relance
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            total_chunks += len(chunks)
            success += 1
            print(f"        OK : {len(chunks)} chunks inseres")

        except Exception as e:
            print(f"        ERREUR : {e}")
            errors.append({"file": json_file.name, "error": str(e)})

    print("=" * 60)
    print(f"Fichiers traites  : {success}/{len(json_files)}")
    if errors:
        print(f"Erreurs           : {len(errors)}")
        for err in errors:
            print(f"  - {err['file']}: {err['error']}")
    print(f"Total chunks      : {total_chunks}")
    print(f"Collection ChromaDB '{collection_name}' : {collection.count()} chunks au total")


if __name__ == "__main__":
    store_chunks_in_chroma()

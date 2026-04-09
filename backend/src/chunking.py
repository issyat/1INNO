"""
Module de découpage de documents - Version sans langchain
"""

from typing import List, Dict, Any

class Document:
    """Simulation simple de Document LangChain"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class MedicalChunker:
    """Découpe les documents en chunks sans langchain"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Découpe une liste de documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Découpe un texte en chunks"""
        if metadata is None:
            metadata = {}
        
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Chercher la dernière fin de phrase si possible
            if end < text_length:
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                # Trouver le meilleur point de coupure
                if last_period > start + self.chunk_size // 2:
                    end = last_period + 1
                elif last_newline > start + self.chunk_size // 2:
                    end = last_newline + 1
                elif last_space > start + self.chunk_size // 2:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = len(chunks)
                chunks.append(Document(chunk_text, chunk_metadata))
            
            start = end - self.chunk_overlap
            if start <= end:
                start = end
        
        return chunks
    
    def get_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Statistiques des chunks"""
        if not chunks:
            return {"nb_chunks": 0}
        sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "nb_chunks": len(chunks),
            "taille_moyenne": sum(sizes) / len(sizes),
            "taille_min": min(sizes),
            "taille_max": max(sizes)
        }

if __name__ == "__main__":
    print("=" * 50)
    print("Test du chunking")
    print("=" * 50)
    chunker = MedicalChunker()
    # Texte plus petit pour le test
    text = "Ceci est un test. Voici une phrase. Et encore une autre phrase pour tester."
    chunks = chunker.chunk_text(text)
    print(f"✅ {len(chunks)} chunks créés")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk.page_content)} caractères")
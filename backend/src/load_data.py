"""
Module de chargement des données - Version sans langchain
"""

from pathlib import Path
from typing import List

# Import direct sans relative
from chunking import Document

class MedicalDataLoader:
    """Charge les documents depuis le dossier data/"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_documents(self) -> List[Document]:
        """Charge tous les fichiers .txt"""
        documents = []
        txt_files = list(self.data_path.rglob("*.txt"))
        
        if txt_files:
            print(f"📄 Chargement de {len(txt_files)} fichiers TXT...")
            for txt_file in txt_files:
                try:
                    content = txt_file.read_text(encoding='utf-8')
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(txt_file.name), "path": str(txt_file)}
                    )
                    documents.append(doc)
                    print(f"   ✅ {txt_file.name}: {len(content)} caractères")
                except Exception as e:
                    print(f"   ⚠️ {txt_file.name}: {e}")
        
        if not documents:
            print(f"⚠️ Aucun document trouvé dans {self.data_path}")
        else:
            print(f"\n📊 Total: {len(documents)} documents chargés")
        
        return documents

if __name__ == "__main__":
    loader = MedicalDataLoader()
    docs = loader.load_all_documents()
    print(f"✅ {len(docs)} documents chargés")
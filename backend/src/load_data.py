"""
Module de chargement des données médicales - Version Windows compatible
"""

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from pathlib import Path
from typing import List
import os

class MedicalDataLoader:
    """Charge les documents médicaux depuis le dossier data/"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_all_documents(self) -> List[Document]:
        """Charge tous les fichiers .txt du dossier data/"""
        documents = []
        
        txt_files = list(self.data_path.rglob("*.txt"))
        if txt_files:
            print(f"📄 Chargement de {len(txt_files)} fichiers TXT...")
            for txt_file in txt_files:
                try:
                    loader = TextLoader(str(txt_file), encoding='utf-8', autodetect_encoding=True)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"   ✅ {txt_file.name}: {len(docs)} segment(s)")
                except Exception as e:
                    print(f"   ⚠️ Erreur sur {txt_file.name}: {e}")
        
        if not documents:
            print(f"⚠️ Aucun document trouvé dans {self.data_path.absolute()}")
        else:
            print(f"\n📊 Total: {len(documents)} documents chargés")
        
        return documents
    
    def load_single_file(self, filepath: str) -> List[Document]:
        """Charge un seul fichier"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
        
        loader = TextLoader(str(filepath), encoding='utf-8', autodetect_encoding=True)
        return loader.load()
    
    def create_sample_data(self):
        """Crée un fichier exemple"""
        sample_file = self.data_path / "exemple_medical.txt"
        sample_content = '''RECOMMANDATIONS HYPERTENSION

L'hypertension artérielle (HTA) est définie par une pression artérielle 
systolique ≥ 140 mmHg et/ou une pression artérielle diastolique ≥ 90 mmHg.

Traitement de première ligne:
- Inhibiteurs de l'enzyme de conversion (IEC)
- Antagonistes des récepteurs de l'angiotensine II (ARA II)

Source: Recommandations HAS 2024
'''
        sample_file.write_text(sample_content, encoding='utf-8')
        print(f"✅ Fichier exemple créé : {sample_file}")


if __name__ == "__main__":
    loader = MedicalDataLoader()
    loader.create_sample_data()
    docs = loader.load_all_documents()
    print(f"📊 {len(docs)} document(s) chargé(s)")
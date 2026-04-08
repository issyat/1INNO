"""
Module de chargement des données médicales
Charge les documents depuis le dossier data/
"""

from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from pathlib import Path
from typing import List, Optional
import os

class MedicalDataLoader:
    """
    Charge les documents médicaux depuis le dossier data/.
    
    Supporte les formats TXT et PDF.
    
    Exemple:
        loader = MedicalDataLoader("./backend/data")
        documents = loader.load_all_documents()
    """
    
    def __init__(self, data_path: str = "./backend/data"):
        """
        Initialise le chargeur
        
        Args:
            data_path: Chemin vers le dossier contenant les documents
        """
        self.data_path = Path(data_path)
        
        # Créer le dossier si inexistant
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-dossiers recommandés
        (self.data_path / "guidelines").mkdir(exist_ok=True)
        (self.data_path / "articles").mkdir(exist_ok=True)
        (self.data_path / "livres").mkdir(exist_ok=True)
    
    def load_all_documents(self) -> List[Document]:
        """
        Charge tous les documents (TXT et PDF) récursivement
        
        Returns:
            Liste d'objets Document LangChain
        """
        documents = []
        
        # Charger les fichiers TXT
        txt_files = list(self.data_path.rglob("*.txt"))
        if txt_files:
            print(f"📄 Chargement de {len(txt_files)} fichiers TXT...")
            txt_loader = DirectoryLoader(
                str(self.data_path),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'autodetect_encoding': True}
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"   ✅ {len(txt_docs)} documents TXT chargés")
        
        # Charger les fichiers PDF
        pdf_files = list(self.data_path.rglob("*.pdf"))
        if pdf_files:
            print(f"📄 Chargement de {len(pdf_files)} fichiers PDF...")
            try:
                pdf_loader = DirectoryLoader(
                    str(self.data_path),
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                print(f"   ✅ {len(pdf_docs)} documents PDF chargés")
            except Exception as e:
                print(f"   ⚠️ Erreur chargement PDF: {e}")
        
        if not documents:
            print("⚠️ Aucun document trouvé. Veuillez ajouter des fichiers .txt ou .pdf dans backend/data/")
            print(f"   Dossier attendu : {self.data_path.absolute()}")
        else:
            print(f"\n📊 Total : {len(documents)} documents chargés")
            self._print_summary(documents)
        
        return documents
    
    def load_single_file(self, filepath: str) -> List[Document]:
        """
        Charge un seul fichier
        
        Args:
            filepath: Chemin vers le fichier
        
        Returns:
            Liste d'objets Document
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
        
        if filepath.suffix.lower() == '.txt':
            loader = TextLoader(str(filepath), autodetect_encoding=True)
        elif filepath.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(filepath))
        else:
            raise ValueError(f"Format non supporté : {filepath.suffix}. Utilisez .txt ou .pdf")
        
        documents = loader.load()
        print(f"✅ Chargé : {filepath.name} ({len(documents)} segments)")
        
        return documents
    
    def load_from_directory(self, subdirectory: str) -> List[Document]:
        """
        Charge tous les documents d'un sous-dossier spécifique
        
        Args:
            subdirectory: Nom du sous-dossier (ex: "guidelines")
        
        Returns:
            Liste d'objets Document
        """
        target_path = self.data_path / subdirectory
        
        if not target_path.exists():
            raise FileNotFoundError(f"Dossier non trouvé : {target_path}")
        
        documents = []
        
        # TXT
        txt_loader = DirectoryLoader(
            str(target_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        )
        documents.extend(txt_loader.load())
        
        # PDF
        try:
            pdf_loader = DirectoryLoader(
                str(target_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())
        except:
            pass
        
        print(f"✅ Chargé {len(documents)} documents depuis {subdirectory}/")
        return documents
    
    def _print_summary(self, documents: List[Document]):
        """Affiche un résumé des documents chargés"""
        print("\n📋 Résumé des documents :")
        for i, doc in enumerate(documents[:5]):  # Affiche les 5 premiers
            source = doc.metadata.get('source', 'Inconnu')
            source_name = Path(source).name if source != 'Inconnu' else 'Inconnu'
            size = len(doc.page_content)
            print(f"   {i+1}. {source_name} ({size} caractères)")
        
        if len(documents) > 5:
            print(f"   ... et {len(documents) - 5} autres documents")
    
    def create_sample_data(self):
        """
        Crée des fichiers d'exemple pour tester le système
        """
        sample_file = self.data_path / "guidelines" / "exemple_hypertension.txt"
        
        sample_content = """
RECOMMANDATIONS POUR L'HYPERTENSION ARTÉRIELLE

DÉFINITION:
L'hypertension artérielle (HTA) est définie par une pression artérielle systolique ≥ 140 mmHg 
et/ou une pression artérielle diastolique ≥ 90 mmHg, mesurée au cabinet médical.

CLASSIFICATION:
- HTA légère: PAS 140-159 ou PAD 90-99 mmHg
- HTA modérée: PAS 160-179 ou PAD 100-109 mmHg  
- HTA sévère: PAS ≥ 180 ou PAD ≥ 110 mmHg

TRAITEMENT DE PREMIÈRE LIGNE:
Les inhibiteurs de l'enzyme de conversion (IEC) et les antagonistes des récepteurs 
de l'angiotensine II (ARA II) sont recommandés comme traitement initial.

EFFETS SECONDAIRES DES IEC:
- Toux sèche (10-15% des patients)
- Hyperkaliémie (rare)
- Angio-œdème (très rare)

SOURCE: Recommandations HAS 2024
"""
        
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(sample_content, encoding='utf-8')
        print(f"✅ Fichier exemple créé : {sample_file}")


# Test rapide
if __name__ == "__main__":
    print("=" * 50)
    print("Test du module de chargement de données")
    print("=" * 50)
    
    loader = MedicalDataLoader("./test_data")
    
    # Créer un fichier exemple
    loader.create_sample_data()
    
    # Charger les documents
    documents = loader.load_all_documents()
    
    if documents:
        print(f"\n📄 Premier document :")
        print(f"   Source : {documents[0].metadata.get('source', 'Inconnue')}")
        print(f"   Contenu : {documents[0].page_content[:200]}...")
    
    # Nettoyage
    import shutil
    shutil.rmtree("./test_data", ignore_errors=True)
    
    print("\n✅ Module de chargement fonctionnel !")
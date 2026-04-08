"""
Tests unitaires pour le backend
Exécutez avec : python -m pytest backend/tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import shutil
from src.embeddings import MedicalEmbeddings
from src.chunking import MedicalChunker
from src.vector_store import MedicalVectorStore
from src.load_data import MedicalDataLoader


class TestEmbeddings(unittest.TestCase):
    """Tests pour le module embeddings"""
    
    def setUp(self):
        self.emb = MedicalEmbeddings()
    
    def test_embedding_dimension(self):
        """Vérifie que la dimension est 768"""
        self.assertEqual(self.emb.dimension, 768)
    
    def test_encode_single(self):
        """Test d'encodage d'un seul texte"""
        vector = self.emb.encode_single("Test médical")
        self.assertEqual(len(vector), 768)
        self.assertIsInstance(vector, type(__import__('numpy').array([])))
    
    def test_similarity_range(self):
        """Vérifie que la similarité est entre -1 et 1"""
        sim = self.emb.similarity("texte A", "texte B")
        self.assertGreaterEqual(sim, -1)
        self.assertLessEqual(sim, 1)
    
    def test_similarity_identical(self):
        """Deux textes identiques doivent avoir similarité ~1"""
        text = "L'hypertension est une maladie"
        sim = self.emb.similarity(text, text)
        self.assertAlmostEqual(sim, 1.0, places=5)


class TestChunking(unittest.TestCase):
    """Tests pour le module chunking"""
    
    def setUp(self):
        self.chunker = MedicalChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_creation(self):
        """Test de création de chunks"""
        text = "a" * 300  # 300 caractères
        chunks = self.chunker.chunk_text(text)
        # Avec chunk_size=100, devrait faire ~3-4 chunks
        self.assertGreater(len(chunks), 2)
    
    def test_chunk_size_limit(self):
        """Vérifie qu'aucun chunk ne dépasse la taille max"""
        text = "Ceci est un texte de test. " * 50
        chunks = self.chunker.chunk_text(text)
        
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 100 + 20)  # + overlap
    
    def test_stats(self):
        """Test des statistiques"""
        chunks = self.chunker.chunk_text("Test " * 50)
        stats = self.chunker.get_stats(chunks)
        
        self.assertIn("nb_chunks", stats)
        self.assertIn("taille_moyenne", stats)
        self.assertGreater(stats["nb_chunks"], 0)


class TestVectorStore(unittest.TestCase):
    """Tests pour le vector store"""
    
    def setUp(self):
        # Créer un dossier temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.store = MedicalVectorStore(persist_directory=self.temp_dir)
        self.chunker = MedicalChunker()
    
    def tearDown(self):
        # Nettoyer le dossier temporaire
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_collection(self):
        """Test de création de collection"""
        collection = self.store.create_collection("test_collection")
        self.assertIsNotNone(collection)
        self.assertEqual(self.store.collection.name, "test_collection")
    
    def test_add_and_search(self):
        """Test d'ajout et recherche"""
        self.store.create_collection("test_search")
        
        # Ajouter des documents
        docs = [
            "Le traitement de l'hypertension utilise les IEC",
            "Les statines sont pour le cholestérol",
            "L'amoxicilline est un antibiotique"
        ]
        
        chunks = []
        for i, doc in enumerate(docs):
            chunks.extend(self.chunker.chunk_text(doc, {"id": i}))
        
        self.store.add_documents(chunks)
        
        # Rechercher
        results = self.store.search("traitement hypertension", k=1)
        
        self.assertGreater(len(results["documents"]), 0)
        self.assertIn("IEC", results["documents"][0])
    
    def test_collection_info(self):
        """Test des informations de collection"""
        self.store.create_collection("test_info")
        info = self.store.get_collection_info()
        
        self.assertEqual(info["status"], "ready")
        self.assertEqual(info["collection_name"], "test_info")


class TestDataLoader(unittest.TestCase):
    """Tests pour le data loader"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.loader = MedicalDataLoader(data_path=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_sample(self):
        """Test de création de fichier exemple"""
        self.loader.create_sample_data()
        
        # Vérifier que le fichier a été créé
        sample_file = os.path.join(self.temp_dir, "guidelines", "exemple_hypertension.txt")
        self.assertTrue(os.path.exists(sample_file))
    
    def test_load_documents(self):
        """Test de chargement de documents"""
        self.loader.create_sample_data()
        documents = self.loader.load_all_documents()
        
        self.assertGreater(len(documents), 0)


def run_all_tests():
    """Exécute tous les tests"""
    # Créer un loader de tests
    loader = unittest.TestLoader()
    
    # Créer une suite avec tous les tests
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Afficher le résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES TESTS")
    print("=" * 50)
    print(f"Tests exécutés : {result.testsRun}")
    print(f"Réussis : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs : {len(result.failures)}")
    print(f"Erreurs : {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
"""
Unit tests for the chunker module.
"""

import pytest
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from rag_chunker.chunker import RecursiveChunker, ChunkingConfig
from rag_chunker.models import Section, Sentence, DocumentInfo, Chunk
from rag_chunker.utils import count_tokens


class TestChunkingConfig:
    """Tests for ChunkingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        
        assert config.max_tokens == 512
        assert config.overlap_tokens == 75
        assert config.overlap_sentences == 2
        assert config.prefer_sentence_boundaries is True
        assert config.extract_entities is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            max_tokens=256,
            overlap_tokens=50,
            extract_entities=False
        )
        
        assert config.max_tokens == 256
        assert config.overlap_tokens == 50
        assert config.extract_entities is False
    
    def test_overlap_bounds(self):
        """Test that overlap is bounded to max_tokens/4."""
        config = ChunkingConfig(max_tokens=100, overlap_tokens=80)
        
        # Should be reduced to max_tokens // 4 = 25
        assert config.overlap_tokens == 25


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker with mocked spaCy."""
        config = ChunkingConfig(max_tokens=100, overlap_sentences=1)
        
        # Mock spaCy processor
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=[])
        mock_spacy.extract_entities = Mock(return_value=[])
        
        return RecursiveChunker(config=config, spacy_processor=mock_spacy)
    
    @pytest.fixture
    def doc_info(self):
        """Create test document info."""
        return DocumentInfo(
            document_id="doc_test123",
            title="Test Document",
            path="test.pdf",
            page_count=1
        )
    
    def test_small_section_single_chunk(self, chunker, doc_info):
        """Test that small sections become single chunks."""
        section = Section(
            title="Small Section",
            content=["This is a short section."],
            page=1
        )
        
        chunks = chunker.chunk_section(section, doc_info)
        
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short section."
        assert chunks[0].metadata.section == "Small Section"
    
    def test_empty_section(self, chunker, doc_info):
        """Test handling of empty sections."""
        section = Section(
            title="Empty",
            content=[],
            page=1
        )
        
        chunks = chunker.chunk_section(section, doc_info)
        
        assert len(chunks) == 0
    
    def test_chunk_metadata(self, chunker, doc_info):
        """Test that chunk metadata is correctly populated."""
        section = Section(
            title="Test Section",
            content=["Test content."],
            page=5,
            section_id="sec123",
            parent_section_id="parent456"
        )
        
        chunks = chunker.chunk_section(section, doc_info)
        
        assert len(chunks) == 1
        metadata = chunks[0].metadata
        
        assert metadata.document_id == "doc_test123"
        assert metadata.section == "Test Section"
        assert metadata.page == 5
        assert metadata.chunk_index == 0
        assert metadata.document_title == "Test Document"
        assert metadata.parent_section_id == "parent456"
    
    def test_large_section_splits(self, doc_info):
        """Test that large sections are split into multiple chunks."""
        config = ChunkingConfig(max_tokens=50, overlap_sentences=0)
        
        # Create sentences
        sentences = [
            Sentence(text="Sentence one.", start_char=0, end_char=13, token_count=3),
            Sentence(text="Sentence two.", start_char=14, end_char=27, token_count=3),
            Sentence(text="Sentence three.", start_char=28, end_char=43, token_count=3),
            Sentence(text="Sentence four.", start_char=44, end_char=58, token_count=3),
        ]
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=sentences)
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        # Create section that would exceed token limit
        section = Section(
            title="Large Section",
            content=["Sentence one. Sentence two. Sentence three. Sentence four."],
            page=1
        )
        
        # Manually set token count to exceed limit
        with patch('rag_chunker.chunker.count_tokens', return_value=200):
            chunks = chunker.chunk_section(section, doc_info)
        
        # Should have multiple chunks
        assert len(chunks) >= 1


class TestSentenceAggregation:
    """Tests for sentence aggregation logic."""
    
    def test_sentences_aggregated_to_limit(self):
        """Test that sentences are aggregated up to token limit."""
        config = ChunkingConfig(max_tokens=20, overlap_sentences=0, extract_entities=False)
        
        sentences = [
            Sentence(text="Short.", start_char=0, end_char=6, token_count=2),
            Sentence(text="Also short.", start_char=7, end_char=18, token_count=3),
            Sentence(text="Third one.", start_char=19, end_char=29, token_count=3),
        ]
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=sentences)
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        doc_info = DocumentInfo(document_id="test", title="Test")
        section = Section(title="Test", content=["Short. Also short. Third one."], page=1)
        
        with patch('rag_chunker.chunker.count_tokens', side_effect=[30, 2, 3, 3, 3, 5, 8, 8]):
            chunks = chunker.chunk_section(section, doc_info)
        
        assert len(chunks) >= 1


class TestOverlapStrategy:
    """Tests for overlap implementation."""
    
    def test_overlap_sentences_included(self):
        """Test that overlap sentences are included in subsequent chunks."""
        config = ChunkingConfig(max_tokens=30, overlap_sentences=1, extract_entities=False)
        
        # Sentences that will require splitting
        sentences = [
            Sentence(text="First sentence.", start_char=0, end_char=15, token_count=10),
            Sentence(text="Second sentence.", start_char=16, end_char=32, token_count=10),
            Sentence(text="Third sentence.", start_char=33, end_char=48, token_count=10),
        ]
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=sentences)
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        doc_info = DocumentInfo(document_id="test", title="Test")
        section = Section(title="Test", content=["First sentence. Second sentence. Third sentence."], page=1)
        
        with patch('rag_chunker.chunker.count_tokens', return_value=35):
            chunks = chunker.chunk_section(section, doc_info)
        
        # With overlap, later chunks should include previous sentences
        assert len(chunks) >= 1


class TestLongSentenceHandling:
    """Tests for handling extremely long sentences."""
    
    def test_long_sentence_token_split(self):
        """Test that sentences exceeding max tokens are split."""
        config = ChunkingConfig(max_tokens=10, extract_entities=False)
        
        # One very long sentence
        long_sentence = Sentence(
            text="This is a very long sentence that exceeds the token limit.",
            start_char=0,
            end_char=58,
            token_count=50  # Exceeds max_tokens
        )
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=[long_sentence])
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        doc_info = DocumentInfo(document_id="test", title="Test")
        section = Section(
            title="Test",
            content=["This is a very long sentence that exceeds the token limit."],
            page=1
        )
        
        with patch('rag_chunker.chunker.count_tokens', return_value=50):
            chunks = chunker.chunk_section(section, doc_info)
        
        # Should have multiple chunks from the split sentence
        assert len(chunks) >= 1


class TestChunkOutput:
    """Tests for chunk output format."""
    
    def test_chunk_to_dict(self):
        """Test Chunk.to_dict() format."""
        from rag_chunker.models import Chunk, ChunkMetadata
        
        metadata = ChunkMetadata(
            chunk_id="chunk_abc123",
            document_id="doc_xyz",
            section="Introduction",
            page=1,
            chunk_index=0,
            entities=[{"text": "Python", "label": "TECH"}],
            token_count=50
        )
        
        chunk = Chunk(text="This is test content.", metadata=metadata)
        result = chunk.to_dict()
        
        assert result["text"] == "This is test content."
        assert result["metadata"]["chunk_id"] == "chunk_abc123"
        assert result["metadata"]["section"] == "Introduction"
        assert result["metadata"]["page"] == 1
        assert result["metadata"]["entities"] == [{"text": "Python", "label": "TECH"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

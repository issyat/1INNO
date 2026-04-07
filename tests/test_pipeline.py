"""
Integration tests for the complete chunking pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import json

sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from rag_chunker.pipeline import (
    chunk_document, ChunkingPipeline, PipelineConfig
)
from rag_chunker.models import Section, DocumentInfo


class TestPipelineConfig:
    """Tests for PipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        
        assert config.max_tokens == 512
        assert config.overlap_sentences == 2
        assert config.enable_semantic_fallback is False
        assert config.spacy_model == "en_core_web_sm"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            max_tokens=256,
            enable_semantic_fallback=True,
            semantic_coherence_threshold=0.7
        )
        
        assert config.max_tokens == 256
        assert config.enable_semantic_fallback is True
        assert config.semantic_coherence_threshold == 0.7


class TestChunkingPipeline:
    """Tests for ChunkingPipeline."""
    
    @pytest.fixture
    def mock_parser(self):
        """Create a mock parser."""
        mock = Mock()
        mock.parse = Mock(return_value=(
            [
                Section(
                    title="Test Section",
                    content=["This is test content."],
                    page=1
                )
            ],
            DocumentInfo(
                document_id="doc_test",
                title="Test Document",
                page_count=1
            )
        ))
        return mock
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ChunkingPipeline()
        
        assert pipeline.parser is not None
        assert pipeline.spacy is not None
        assert pipeline.chunker is not None
    
    def test_pipeline_with_config(self):
        """Test pipeline with custom config."""
        config = PipelineConfig(max_tokens=256)
        pipeline = ChunkingPipeline(config=config)
        
        assert pipeline.config.max_tokens == 256
        assert pipeline.chunker.config.max_tokens == 256
    
    @patch('rag_chunker.pipeline.PDFParser')
    @patch('rag_chunker.pipeline.SpacyProcessor')
    def test_process_returns_list(self, mock_spacy_cls, mock_parser_cls):
        """Test that process returns a list of dicts."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse = Mock(return_value=(
            [Section(title="Test", content=["Content."], page=1)],
            DocumentInfo(document_id="test", title="Test")
        ))
        mock_parser_cls.return_value = mock_parser
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=[])
        mock_spacy.extract_entities = Mock(return_value=[])
        mock_spacy_cls.return_value = mock_spacy
        
        pipeline = ChunkingPipeline()
        pipeline.parser = mock_parser
        pipeline.spacy = mock_spacy
        
        with patch('rag_chunker.chunker.count_tokens', return_value=10):
            result = pipeline.process("test.pdf")
        
        assert isinstance(result, list)
        if result:
            assert "text" in result[0]
            assert "metadata" in result[0]


class TestChunkDocument:
    """Tests for the chunk_document function."""
    
    @patch('rag_chunker.pipeline.ChunkingPipeline')
    def test_chunk_document_default_config(self, mock_pipeline_cls):
        """Test chunk_document with default config."""
        mock_pipeline = Mock()
        mock_pipeline.process = Mock(return_value=[
            {"text": "Test", "metadata": {"chunk_id": "test"}}
        ])
        mock_pipeline_cls.return_value = mock_pipeline
        
        result = chunk_document("test.pdf")
        
        assert mock_pipeline_cls.called
        assert result == [{"text": "Test", "metadata": {"chunk_id": "test"}}]
    
    @patch('rag_chunker.pipeline.ChunkingPipeline')
    def test_chunk_document_with_kwargs(self, mock_pipeline_cls):
        """Test chunk_document with keyword arguments."""
        mock_pipeline = Mock()
        mock_pipeline.process = Mock(return_value=[])
        mock_pipeline_cls.return_value = mock_pipeline
        
        chunk_document("test.pdf", max_tokens=256, extract_entities=False)
        
        # Verify config was created with kwargs
        call_args = mock_pipeline_cls.call_args
        config = call_args.kwargs.get('config') or call_args.args[0]
        
        assert config.max_tokens == 256
        assert config.extract_entities is False


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_content_handling(self):
        """Test handling of empty/malformed content."""
        from rag_chunker.models import Section
        
        section = Section(title="Empty", content=[], page=1)
        assert section.full_text == ""
    
    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        from rag_chunker.utils import clean_text
        
        result = clean_text("   \n\t\n   ")
        assert result == ""
    
    def test_deterministic_chunk_ids(self):
        """Test that chunk IDs are deterministic."""
        from rag_chunker.utils import generate_chunk_id
        
        id1 = generate_chunk_id("doc1", "section1", 0)
        id2 = generate_chunk_id("doc1", "section1", 0)
        id3 = generate_chunk_id("doc1", "section1", 1)
        
        assert id1 == id2  # Same inputs = same ID
        assert id1 != id3  # Different index = different ID
    
    def test_deterministic_document_ids(self):
        """Test that document IDs are deterministic."""
        from rag_chunker.utils import generate_document_id
        
        id1 = generate_document_id("/path/to/doc.pdf")
        id2 = generate_document_id("/path/to/doc.pdf")
        id3 = generate_document_id("/path/to/other.pdf")
        
        assert id1 == id2
        assert id1 != id3


class TestShortDocument:
    """Tests simulating a short document."""
    
    def test_short_document_single_chunk(self):
        """Test that a very short document results in a single chunk."""
        from rag_chunker.chunker import RecursiveChunker, ChunkingConfig
        from rag_chunker.models import Section, DocumentInfo
        
        config = ChunkingConfig(max_tokens=512)
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=[])
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        section = Section(
            title="Short Section",
            content=["This is a short document."],
            page=1
        )
        doc_info = DocumentInfo(document_id="short_doc", title="Short")
        
        with patch('rag_chunker.chunker.count_tokens', return_value=10):
            chunks = chunker.chunk_section(section, doc_info)
        
        assert len(chunks) == 1
        assert "short document" in chunks[0].text


class TestLongStructuredDocument:
    """Tests simulating a long structured document."""
    
    def test_multiple_sections_multiple_chunks(self):
        """Test that a long document with multiple sections produces multiple chunks."""
        from rag_chunker.chunker import RecursiveChunker, ChunkingConfig
        from rag_chunker.models import Section, DocumentInfo, Sentence
        
        config = ChunkingConfig(max_tokens=50, overlap_sentences=0, extract_entities=False)
        
        # Create sentences for splitting
        sentences = [
            Sentence(text="Sentence one.", start_char=0, end_char=13, token_count=20),
            Sentence(text="Sentence two.", start_char=14, end_char=27, token_count=20),
            Sentence(text="Sentence three.", start_char=28, end_char=43, token_count=20),
        ]
        
        mock_spacy = Mock()
        mock_spacy.segment_sentences = Mock(return_value=sentences)
        mock_spacy.extract_entities = Mock(return_value=[])
        
        chunker = RecursiveChunker(config=config, spacy_processor=mock_spacy)
        
        sections = [
            Section(title="Introduction", content=["Intro content here."], page=1),
            Section(title="Methods", content=["Methods content here."], page=2),
            Section(title="Results", content=["Results content here."], page=3),
        ]
        
        doc_info = DocumentInfo(
            document_id="long_doc",
            title="Long Document",
            page_count=10
        )
        
        all_chunks = []
        with patch('rag_chunker.chunker.count_tokens', return_value=100):
            for section in sections:
                chunks = chunker.chunk_section(section, doc_info, len(all_chunks))
                all_chunks.extend(chunks)
        
        # Should have chunks from multiple sections
        assert len(all_chunks) >= 3
        
        # Verify section attribution
        section_names = set(c.metadata.section for c in all_chunks)
        assert "Introduction" in section_names or "Methods" in section_names


class TestMessyDocument:
    """Tests simulating a messy/unstructured document."""
    
    def test_no_headings_page_fallback(self):
        """Test that documents without headings fall back to page-based sections."""
        from rag_chunker.parser import PDFParser
        from rag_chunker.models import TextBlock, BlockType
        
        # This tests the _group_by_pages method
        parser = PDFParser()
        
        # Create blocks without headings
        blocks = [
            TextBlock(text="Content on page 1", block_type=BlockType.PARAGRAPH, page=1),
            TextBlock(text="More content page 1", block_type=BlockType.PARAGRAPH, page=1),
            TextBlock(text="Content on page 2", block_type=BlockType.PARAGRAPH, page=2),
        ]
        
        doc_info = DocumentInfo(document_id="test", title="Test")
        
        sections = parser._group_by_pages(blocks, doc_info)
        
        assert len(sections) == 2
        assert sections[0].title == "Page 1"
        assert sections[1].title == "Page 2"
    
    def test_table_blocks_preserved(self):
        """Test that table content is marked appropriately."""
        from rag_chunker.parser import PDFParser
        from rag_chunker.models import TextBlock, BlockType
        
        parser = PDFParser()
        
        blocks = [
            TextBlock(text="Introduction", block_type=BlockType.HEADING, page=1, is_bold=True, font_size=16),
            TextBlock(text="Name\tAge\nJohn\t25", block_type=BlockType.TABLE, page=1, font_size=12),
        ]
        
        doc_info = DocumentInfo(document_id="test", title="Test")
        
        # Test section grouping with table
        sections = parser._group_into_sections(blocks, doc_info)
        
        assert len(sections) >= 1
        # Table should be wrapped with markers
        table_found = any("[TABLE]" in s.full_text for s in sections)
        assert table_found


class TestOutputFormat:
    """Tests for output format compatibility."""
    
    def test_output_json_serializable(self):
        """Test that output is JSON serializable."""
        from rag_chunker.models import Chunk, ChunkMetadata
        
        metadata = ChunkMetadata(
            chunk_id="chunk_123",
            document_id="doc_456",
            section="Test",
            page=1,
            chunk_index=0,
            entities=[{"text": "Test", "label": "ORG"}]
        )
        
        chunk = Chunk(text="Test content", metadata=metadata)
        result = chunk.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        assert parsed["text"] == "Test content"
        assert parsed["metadata"]["chunk_id"] == "chunk_123"
    
    def test_output_ready_for_vector_db(self):
        """Test that output format is ready for vector DB insertion."""
        from rag_chunker.models import Chunk, ChunkMetadata
        
        metadata = ChunkMetadata(
            chunk_id="chunk_789",
            document_id="doc_abc",
            section="Introduction",
            page=1,
            chunk_index=0
        )
        
        chunk = Chunk(text="This is embedable content.", metadata=metadata)
        result = chunk.to_dict()
        
        # Required fields for vector DB
        assert "text" in result
        assert len(result["text"]) > 0
        assert "metadata" in result
        assert "chunk_id" in result["metadata"]
        assert "document_id" in result["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

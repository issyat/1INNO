"""
Semantic fallback module for handling low-coherence sections.

This module provides optional semantic grouping when sections contain
mixed topics that would benefit from semantic clustering rather than
sequential chunking.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .models import Sentence, Section, Chunk, ChunkMetadata, DocumentInfo
from .utils import count_tokens, generate_chunk_id


@dataclass
class SemanticConfig:
    """Configuration for semantic processing."""
    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality
    coherence_threshold: float = 0.6  # Below this triggers semantic grouping
    min_cluster_size: int = 2
    max_tokens: int = 512


class SemanticGrouper:
    """
    Provides semantic fallback for low-coherence sections.
    
    Features:
    - Sentence embedding computation
    - Coherence detection
    - Semantic clustering/grouping
    - Chunk reconstruction from semantic groups
    
    This module is optional and can be toggled on/off.
    """
    
    def __init__(self, config: Optional[SemanticConfig] = None):
        """
        Initialize semantic grouper.
        
        Args:
            config: Semantic processing configuration.
        """
        self.config = config or SemanticConfig()
        self._model = None
        self._enabled = True
    
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic features. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    @property
    def enabled(self) -> bool:
        """Check if semantic processing is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable semantic processing."""
        self._enabled = value
    
    def compute_embeddings(self, sentences: List[Sentence]) -> np.ndarray:
        """
        Compute embeddings for a list of sentences.
        
        Args:
            sentences: List of Sentence objects.
            
        Returns:
            Numpy array of shape (num_sentences, embedding_dim).
        """
        if not sentences:
            return np.array([])
        
        model = self._get_model()
        texts = [s.text for s in sentences]
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def compute_coherence(self, sentences: List[Sentence]) -> float:
        """
        Compute coherence score for a section.
        
        Coherence is measured as average pairwise cosine similarity
        between sentence embeddings. High coherence = single topic.
        
        Args:
            sentences: List of sentences to analyze.
            
        Returns:
            Coherence score between 0 and 1.
        """
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by definition
        
        embeddings = self.compute_embeddings(sentences)
        
        # Compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Average off-diagonal elements (excluding self-similarity)
        n = len(sentences)
        mask = ~np.eye(n, dtype=bool)
        coherence = similarity_matrix[mask].mean()
        
        return float(coherence)
    
    def should_use_semantic(self, sentences: List[Sentence]) -> bool:
        """
        Determine if semantic grouping should be used.
        
        Args:
            sentences: Sentences from a section.
            
        Returns:
            True if section has low coherence and should use semantic grouping.
        """
        if not self._enabled:
            return False
        
        if len(sentences) < 4:  # Too few sentences to cluster
            return False
        
        coherence = self.compute_coherence(sentences)
        return coherence < self.config.coherence_threshold
    
    def cluster_sentences(self, sentences: List[Sentence], 
                         n_clusters: Optional[int] = None) -> List[List[Sentence]]:
        """
        Cluster sentences by semantic similarity.
        
        Args:
            sentences: Sentences to cluster.
            n_clusters: Number of clusters. If None, determined automatically.
            
        Returns:
            List of sentence groups (clusters).
        """
        if len(sentences) < self.config.min_cluster_size:
            return [sentences]
        
        embeddings = self.compute_embeddings(sentences)
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Heuristic: sqrt(n) clusters, bounded
            n_clusters = max(2, min(len(sentences) // 3, int(np.sqrt(len(sentences)))))
        
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42,  # Deterministic
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)
            
        except ImportError:
            # Fallback: simple agglomerative approach
            return self._simple_clustering(sentences, embeddings, n_clusters)
        
        # Group sentences by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(sentences[i])
        
        # Filter empty clusters and sort by first sentence position
        clusters = [c for c in clusters if c]
        clusters.sort(key=lambda c: c[0].start_char)
        
        return clusters
    
    def _simple_clustering(self, sentences: List[Sentence],
                          embeddings: np.ndarray,
                          n_clusters: int) -> List[List[Sentence]]:
        """Simple clustering without sklearn."""
        # Greedy approach: group similar consecutive sentences
        if len(sentences) <= n_clusters:
            return [[s] for s in sentences]
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)
        
        clusters = []
        current_cluster = [sentences[0]]
        current_embedding = normalized[0]
        
        for i in range(1, len(sentences)):
            similarity = np.dot(current_embedding, normalized[i])
            
            if similarity > 0.7 and len(clusters) < n_clusters - 1:
                # Similar enough, add to current cluster
                current_cluster.append(sentences[i])
                # Update cluster centroid
                current_embedding = np.mean(normalized[list(range(len(current_cluster)))], axis=0)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [sentences[i]]
                current_embedding = normalized[i]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def group_semantically(self, sentences: List[Sentence],
                          section: Section,
                          document_info: DocumentInfo,
                          start_index: int = 0) -> List[Chunk]:
        """
        Create chunks by semantic grouping.
        
        Args:
            sentences: Sentences to group.
            section: Source section.
            document_info: Document metadata.
            start_index: Starting chunk index.
            
        Returns:
            List of chunks created from semantic groups.
        """
        if not sentences:
            return []
        
        # Cluster sentences
        clusters = self.cluster_sentences(sentences)
        
        chunks = []
        chunk_index = start_index
        
        for cluster in clusters:
            # Build chunk from cluster
            cluster_chunks = self._create_chunks_from_cluster(
                cluster, section, document_info, chunk_index
            )
            chunks.extend(cluster_chunks)
            chunk_index += len(cluster_chunks)
        
        return chunks
    
    def _create_chunks_from_cluster(self, sentences: List[Sentence],
                                   section: Section,
                                   document_info: DocumentInfo,
                                   start_index: int) -> List[Chunk]:
        """Create one or more chunks from a sentence cluster."""
        # Aggregate sentences up to token limit
        chunks = []
        current_texts = []
        current_tokens = 0
        current_entities = []
        
        for sentence in sentences:
            if current_tokens + sentence.token_count > self.config.max_tokens:
                # Create chunk from accumulated sentences
                if current_texts:
                    chunk = self._create_chunk(
                        " ".join(current_texts),
                        section,
                        document_info,
                        start_index + len(chunks),
                        current_entities
                    )
                    chunks.append(chunk)
                
                current_texts = [sentence.text]
                current_tokens = sentence.token_count
                current_entities = list(sentence.entities)
            else:
                current_texts.append(sentence.text)
                current_tokens += sentence.token_count
                current_entities.extend(sentence.entities)
        
        # Last chunk
        if current_texts:
            chunk = self._create_chunk(
                " ".join(current_texts),
                section,
                document_info,
                start_index + len(chunks),
                current_entities
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str,
                     section: Section,
                     document_info: DocumentInfo,
                     chunk_index: int,
                     entities: List[dict]) -> Chunk:
        """Create a single chunk."""
        chunk_id = generate_chunk_id(
            document_info.document_id,
            f"{section.title}_semantic",
            chunk_index
        )
        
        # Deduplicate entities
        seen = set()
        unique_entities = []
        for ent in entities:
            key = (ent.get("text"), ent.get("label"))
            if key not in seen:
                seen.add(key)
                unique_entities.append({"text": ent["text"], "label": ent["label"]})
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_info.document_id,
            section=section.title,
            page=section.page,
            chunk_index=chunk_index,
            entities=unique_entities if unique_entities else None,
            parent_section_id=section.parent_section_id,
            document_title=document_info.title,
            token_count=count_tokens(text)
        )
        
        return Chunk(text=text, metadata=metadata)

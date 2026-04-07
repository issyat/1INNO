"""
spaCy processor for sentence segmentation and entity extraction.
"""

from typing import List, Dict, Optional, Any
from .models import Sentence
from .utils import count_tokens


class SpacyProcessor:
    """
    Handles NLP processing using spaCy.
    
    Features:
    - Sentence segmentation
    - Named entity extraction
    - Lazy model loading
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy processor.
        
        Args:
            model_name: Name of spaCy model to load.
                       Default is "en_core_web_sm" for efficiency.
        """
        self.model_name = model_name
        self._nlp = None
    
    def _get_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                # Model not installed, try to download
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", self.model_name
                ])
                self._nlp = spacy.load(self.model_name)
            
            # Optimize for sentence segmentation
            # Disable unused pipeline components for speed
            if "ner" not in self._nlp.pipe_names:
                # If NER not available, that's fine
                pass
        
        return self._nlp
    
    def segment_sentences(self, text: str, 
                         extract_entities: bool = False) -> List[Sentence]:
        """
        Segment text into sentences using spaCy.
        
        Args:
            text: Text to segment.
            extract_entities: Whether to extract named entities.
            
        Returns:
            List of Sentence objects with metadata.
        """
        if not text or not text.strip():
            return []
        
        nlp = self._get_nlp()
        doc = nlp(text)
        
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            # Extract entities for this sentence if requested
            entities = []
            if extract_entities:
                for ent in doc.ents:
                    # Check if entity overlaps with sentence
                    if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                        entities.append({
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char - sent.start_char,
                            "end": ent.end_char - sent.start_char
                        })
            
            sentence = Sentence(
                text=sent_text,
                start_char=sent.start_char,
                end_char=sent.end_char,
                entities=entities,
                token_count=count_tokens(sent_text)
            )
            sentences.append(sentence)
        
        return sentences
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process.
            
        Returns:
            List of entity dictionaries with text and label.
        """
        if not text:
            return []
        
        nlp = self._get_nlp()
        doc = nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        return entities
    
    def process_batch(self, texts: List[str], 
                     extract_entities: bool = False,
                     batch_size: int = 50) -> List[List[Sentence]]:
        """
        Process multiple texts efficiently using spaCy's pipe.
        
        Args:
            texts: List of texts to process.
            extract_entities: Whether to extract entities.
            batch_size: Batch size for processing.
            
        Returns:
            List of sentence lists, one per input text.
        """
        if not texts:
            return []
        
        nlp = self._get_nlp()
        results = []
        
        # Use spaCy's pipe for efficient batch processing
        for doc in nlp.pipe(texts, batch_size=batch_size):
            sentences = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue
                
                entities = []
                if extract_entities:
                    for ent in doc.ents:
                        if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                            entities.append({
                                "text": ent.text,
                                "label": ent.label_,
                                "start": ent.start_char - sent.start_char,
                                "end": ent.end_char - sent.start_char
                            })
                
                sentence = Sentence(
                    text=sent_text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    entities=entities,
                    token_count=count_tokens(sent_text)
                )
                sentences.append(sentence)
            
            results.append(sentences)
        
        return results
    
    def get_sentence_count(self, text: str) -> int:
        """
        Quickly count sentences in text.
        
        Args:
            text: Text to count sentences in.
            
        Returns:
            Number of sentences.
        """
        if not text:
            return 0
        
        nlp = self._get_nlp()
        doc = nlp(text)
        return len(list(doc.sents))


# Global processor instance for convenience
_default_processor: Optional[SpacyProcessor] = None


def get_default_processor() -> SpacyProcessor:
    """Get or create default spaCy processor."""
    global _default_processor
    if _default_processor is None:
        _default_processor = SpacyProcessor()
    return _default_processor

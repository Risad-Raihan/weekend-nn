# 20 RAG Development Warmup Exercises

A comprehensive collection of coding exercises to master Retrieval Augmented Generation (RAG) development, progressing from fundamentals to enterprise-level implementations.

## 🏗️ Foundation Tier (Exercises 1-5)
*Master the mathematical and conceptual foundations of RAG*

### Exercise 1: Text Similarity Calculator
```python
import math
from collections import Counter

def cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts based on token frequencies.
    Returns a value between 0 (no similarity) and 1 (identical).
    
    Key concepts:
    - Tokenization and frequency counting
    - Vector operations (dot product, magnitude)
    - Similarity normalization
    """
    pass

# Test cases
doc1 = "machine learning is powerful"
doc2 = "machine learning algorithms are powerful tools"
doc3 = "cats and dogs are pets"
```

### Exercise 2: TF-IDF Calculator
```python
import math
from collections import Counter
from typing import List, Dict

def calculate_tfidf(documents: List[str]) -> List[Dict[str, float]]:
    """
    Calculate TF-IDF scores for each term in each document.
    
    Formula: TF-IDF = TF × IDF = (term_count/total_terms) × log(total_docs/docs_containing_term)
    
    Returns a list where each element is a dict of {term: tfidf_score}
    for the corresponding document.
    
    Key concepts:
    - Term frequency analysis
    - Inverse document frequency
    - Document importance scoring
    """
    pass

# Test cases
docs = [
    "machine learning algorithms are powerful",
    "machine learning is a subset of artificial intelligence", 
    "deep learning uses neural networks",
    "algorithms solve complex problems"
]
```

### Exercise 3: Document Chunking with Overlap
```python
from typing import List

def smart_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks while trying to preserve sentence boundaries.
    Essential for RAG since you need to break large documents into searchable pieces.
    
    Key concepts:
    - Sentence boundary detection
    - Context preservation with overlap
    - Handling edge cases (short texts, no sentences)
    
    Args:
        text: Input document text
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
    
    Returns:
        List of text chunks with preserved context
    """
    pass

# Test with a long article about AI, chunk_size=100, overlap=20
```

### Exercise 4: Query Expansion with Synonyms
```python
from typing import Dict, List

def expand_query(query: str, synonym_dict: Dict[str, List[str]]) -> List[str]:
    """
    Generate multiple versions of a query using synonyms to improve retrieval.
    Returns original query + expanded variants.
    
    Key concepts:
    - Vocabulary expansion for better recall
    - Handling multi-word synonyms
    - Query variation generation
    
    Args:
        query: Original search query
        synonym_dict: Mapping of words to their synonyms
    
    Returns:
        List of query variations including original
    """
    pass

# Test: "machine learning" → ["machine learning", "ML algorithms", "artificial intelligence training"]
synonym_dict = {
    "machine": ["ML", "automated", "artificial"],
    "learning": ["training", "education", "algorithms"]
}
```

### Exercise 5: Semantic Search Simulator
```python
from typing import List, Tuple

def semantic_search(query: str, documents: List[str], top_k: int = 3) -> List[Tuple[int, float, str]]:
    """
    Combine multiple similarity methods (cosine + TF-IDF + keyword matching) 
    to rank documents. Return (doc_index, combined_score, explanation).
    
    Key concepts:
    - Hybrid search strategies
    - Score combination and weighting
    - Ranking explanation generation
    
    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of top results to return
    
    Returns:
        List of tuples: (document_index, combined_score, explanation)
    """
    pass

# Should handle queries like "How do neural networks learn?" matching docs about "backpropagation"
```

## 🔧 Core RAG Tier (Exercises 6-10)
*Build production-ready RAG system components*

### Exercise 6: Multi-Modal Document Processor
```python
from typing import Dict, List

def extract_and_chunk_mixed_content(file_path: str) -> Dict[str, List[str]]:
    """
    Extract text from PDFs, handle tables/images, and create structured chunks.
    Return {"text_chunks": [...], "table_chunks": [...], "metadata": [...]}
    Handle different content types that need different retrieval strategies.
    
    Key concepts:
    - Multi-format document processing
    - Content type identification
    - Structure-aware chunking
    
    Args:
        file_path: Path to document (PDF, DOCX, etc.)
    
    Returns:
        Dictionary with categorized content chunks
    """
    pass

# Test with PDFs containing research papers, financial reports, technical docs
```

### Exercise 7: Conversational Memory Manager
```python
from typing import List, Dict

def manage_conversation_context(chat_history: List[Dict], current_query: str, max_context: int) -> str:
    """
    Intelligently summarize chat history and combine with current query for RAG retrieval.
    Handle context compression, entity tracking, and query disambiguation.
    
    Key concepts:
    - Conversation context management
    - Query rewriting with history
    - Context window optimization
    
    Args:
        chat_history: List of {"role": "user/assistant", "content": "..."}
        current_query: Current user question
        max_context: Maximum context length
    
    Returns:
        Enhanced query string with relevant context
    """
    pass

# Handle: "What about the pricing?" (needs context: "Tell me about OpenAI's GPT models")
```

### Exercise 8: Dynamic Knowledge Graph Builder
```python
from typing import Dict

def build_entity_graph(documents: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract entities and relationships from docs to build a knowledge graph.
    Return {entity: {related_entity: connection_strength, ...}}
    Use for relationship-aware retrieval.
    
    Key concepts:
    - Named entity recognition
    - Relationship extraction
    - Graph-based document understanding
    
    Args:
        documents: List of document texts
    
    Returns:
        Graph as adjacency dict with weighted connections
    """
    pass

# Extract: "OpenAI" connects to "GPT", "Altman", "ChatGPT" with different weights
```

### Exercise 9: Retrieval Strategy Optimizer
```python
from typing import Dict

def adaptive_retrieval_strategy(query: str, query_type: str, corpus_stats: Dict) -> Dict[str, float]:
    """
    Dynamically adjust retrieval parameters based on query characteristics.
    Return weights for: {semantic_search: 0.6, keyword_search: 0.3, graph_search: 0.1}
    Different query types need different strategies.
    
    Key concepts:
    - Query classification
    - Adaptive parameter tuning
    - Multi-strategy retrieval
    
    Args:
        query: User query
        query_type: Classification (factual, analytical, comparative, etc.)
        corpus_stats: Statistics about document collection
    
    Returns:
        Dictionary of search method weights
    """
    pass

# "What is X?" → high semantic, "When did X happen?" → high keyword + temporal
```

### Exercise 10: Real-time RAG Cache Manager
```python
from typing import Optional, List
from datetime import datetime

class RAGCache:
    """
    Cache retrieval results and embeddings. Handle cache invalidation when docs update.
    Include cache warming, similarity-based lookup, and freshness management.
    
    Key concepts:
    - Performance optimization through caching
    - Cache invalidation strategies  
    - Similarity-based cache lookup
    """
    
    def smart_cache_retrieval(self, query: str, similarity_threshold: float = 0.85) -> Optional[List[str]]:
        """
        Check cache for similar queries and return cached results if found.
        
        Args:
            query: Current search query
            similarity_threshold: Minimum similarity to use cached result
            
        Returns:
            Cached results if similar query found, None otherwise
        """
        pass
    
    def update_cache_on_doc_change(self, changed_docs: List[str]) -> None:
        """
        Intelligently invalidate related cache entries when documents change.
        
        Args:
            changed_docs: List of document IDs that were modified
        """
        pass
```

## 🚀 Advanced RAG Tier (Exercises 11-15)
*Create intelligent, context-aware RAG systems*

### Exercise 11: Hierarchical Document Summarizer
```python
from typing import Dict, List

def create_hierarchical_summaries(document: str, levels: int = 3) -> Dict[str, List[str]]:
    """
    Create multi-level document summaries for better retrieval granularity.
    Level 1: Sentence-level chunks, Level 2: Paragraph summaries, Level 3: Section summaries
    Return {level: [chunks_at_that_level]} for hierarchical RAG search.
    
    Key concepts:
    - Multi-granularity information representation
    - Hierarchical search strategies
    - Content abstraction levels
    
    Args:
        document: Full document text
        levels: Number of abstraction levels to create
    
    Returns:
        Dictionary mapping levels to chunk lists
    """
    pass

# Enable: Search detailed chunks first, fall back to broader summaries if needed
```

### Exercise 12: Cross-Document Relationship Finder
```python
from typing import List, Dict

def find_cross_document_connections(documents: List[str], entities: List[str]) -> Dict[str, List[Dict]]:
    """
    Find how the same entities/topics are discussed across different documents.
    Return connections like: {"AI safety": [{"doc_id": 1, "stance": "concerned", "evidence": "..."}, ...]}
    Enable multi-document reasoning.
    
    Key concepts:
    - Multi-document analysis
    - Perspective aggregation
    - Cross-reference identification
    
    Args:
        documents: List of document texts
        entities: List of entities to track across documents
    
    Returns:
        Dictionary mapping entities to their mentions across documents
    """
    pass

# Handle: "What do different sources say about climate change?" - aggregate perspectives
```

### Exercise 13: Query Intent Classifier & Router
```python
from typing import Any, Dict

class QueryRouter:
    """
    Classify query intent and route to appropriate RAG strategy.
    Different question types need different RAG approaches.
    
    Key concepts:
    - Intent classification
    - Strategy selection
    - Dynamic routing logic
    """
    
    def classify_and_route(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent and route to appropriate RAG strategy:
        - Factual: Use precise keyword + semantic search
        - Analytical: Use multi-document synthesis 
        - Comparative: Use cross-reference search
        - Temporal: Use time-aware retrieval
        
        Args:
            query: User query to classify and route
        
        Returns:
            Dictionary with intent classification and routing decision
        """
        pass

# "Compare X and Y" → different strategy than "What is X?"
```

### Exercise 14: Confidence-Aware Response Generator
```python
from typing import List, Dict, Any

def generate_answer_with_confidence(query: str, retrieved_chunks: List[str]) -> Dict[str, Any]:
    """
    Generate answer and calculate confidence scores based on:
    - Source agreement/disagreement
    - Information completeness
    - Query-chunk relevance alignment
    Return: {answer: str, confidence: float, evidence_quality: str, gaps: List[str]}
    
    Key concepts:
    - Confidence estimation
    - Source reliability assessment
    - Answer quality metrics
    
    Args:
        query: User question
        retrieved_chunks: Retrieved document chunks
    
    Returns:
        Dictionary with answer and confidence metrics
    """
    pass

# Should detect: "Sources disagree" or "Insufficient information" or "High confidence"
```

### Exercise 15: RAG Performance Profiler
```python
from typing import List, Dict, Any

class RAGProfiler:
    """
    Comprehensive RAG system evaluation and performance analysis.
    
    Key concepts:
    - System performance measurement
    - Bottleneck identification
    - Quality metrics tracking
    """
    
    def profile_rag_pipeline(self, queries: List[str], ground_truth: List[str]) -> Dict[str, Any]:
        """
        Comprehensive RAG system evaluation:
        - Retrieval quality (precision@k, recall@k)
        - Answer faithfulness vs hallucination rate
        - Latency breakdown (retrieval vs generation)
        - Token efficiency metrics
        Return detailed performance report with bottleneck identification.
        
        Args:
            queries: Test queries
            ground_truth: Expected answers
        
        Returns:
            Comprehensive performance analysis report
        """
        pass

# Full system diagnostics: where is your RAG system failing and why?
```

## 🏢 Enterprise RAG Tier (Exercises 16-20)
*Build enterprise-grade, secure, multi-system RAG architecture*

### Exercise 16: Adaptive Chunk Size Optimizer
```python
from typing import List, Dict, Any
from datetime import datetime

class DynamicChunker:
    """
    Dynamically determine optimal chunk sizes based on document characteristics
    and usage patterns.
    
    Key concepts:
    - Performance-driven optimization
    - Content-aware chunking
    - Usage pattern analysis
    """
    
    def optimize_chunk_size(self, document: str, query_patterns: List[str]) -> Dict[str, Any]:
        """
        Dynamically determine optimal chunk sizes based on:
        - Document structure (headers, paragraphs, code blocks)
        - Historical query patterns and retrieval performance
        - Content density and semantic coherence
        Return: {optimal_size: int, chunk_strategy: str, performance_prediction: float}
        
        Args:
            document: Document to analyze
            query_patterns: Historical query patterns for similar documents
        
        Returns:
            Optimized chunking parameters and performance prediction
        """
        pass

# Different docs need different chunking: code vs legal vs research papers
```

### Exercise 17: Multi-Language RAG Handler
```python
from typing import List, Dict
from datetime import datetime

def cross_language_retrieval(query: str, multilingual_docs: Dict[str, List[str]]) -> List[Dict]:
    """
    Handle RAG across multiple languages:
    - Detect query language
    - Translate queries for cross-language search
    - Score relevance across languages with translation confidence
    - Return: [{"doc": str, "lang": str, "score": float, "translation_quality": float}]
    
    Key concepts:
    - Language detection
    - Cross-language information retrieval
    - Translation quality assessment
    
    Args:
        query: Search query in any language
        multilingual_docs: Documents organized by language
    
    Returns:
        Ranked results with language and translation metadata
    """
    pass

# Query in English, find relevant docs in Spanish/French/Chinese with confidence scores
```

### Exercise 18: Temporal-Aware Document Retrieval
```python
from datetime import datetime
from typing import List, Dict

def time_sensitive_retrieval(query: str, documents: List[Dict], query_time: datetime) -> List[Dict]:
    """
    Weight document relevance by temporal factors:
    - Recency decay for time-sensitive topics
    - Historical importance for reference material
    - Trend detection for evolving topics
    - Seasonal relevance patterns
    Return ranked docs with temporal reasoning explanation.
    
    Key concepts:
    - Time-aware information retrieval
    - Temporal relevance modeling
    - Historical vs current information balance
    
    Args:
        query: Search query
        documents: Documents with timestamps and metadata
        query_time: When the query was made
    
    Returns:
        Time-weighted document rankings with explanations
    """
    pass

# "Latest AI developments" prioritizes recent docs, "History of AI" weighs historical importance
```

### Exercise 19: RAG Security & Privacy Filter
```python
from typing import List, Dict

class SecureRAGFilter:
    """
    Implement security layers for enterprise RAG systems.
    
    Key concepts:
    - Data privacy protection
    - Access control enforcement
    - Security threat detection
    """
    
    def sanitize_retrieval_pipeline(self, query: str, user_context: Dict, documents: List[Dict]) -> Dict:
        """
        Implement security layers for RAG:
        - PII detection and masking in documents
        - Access control based on user permissions
        - Query injection detection and prevention
        - Sensitive information leak prevention
        Return: {filtered_docs: List, security_alerts: List, redacted_content: Dict}
        
        Args:
            query: User query to analyze for security risks
            user_context: User permissions and context
            documents: Documents to filter and secure
        
        Returns:
            Secured documents and security analysis results
        """
        pass

# Prevent: retrieving HR docs for non-HR users, leaking SSNs, prompt injection attacks
```

### Exercise 20: Meta-RAG System Orchestrator
```python
from typing import Dict, Any

class MetaRAG:
    """
    Coordinate multiple specialized RAG systems for complex enterprise needs.
    
    Key concepts:
    - Multi-system orchestration
    - Load balancing and failover
    - Result aggregation and conflict resolution
    """
    
    def orchestrate_rag_ensemble(self, query: str, available_systems: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple specialized RAG systems:
        - Route queries to specialized RAG instances (code, medical, legal)
        - Aggregate and rank results from multiple systems
        - Detect conflicting information across systems
        - Provide unified answer with source attribution
        - Handle system failures and fallbacks
        
        Args:
            query: User query
            available_systems: Dictionary of available RAG systems
        
        Returns:
            Orchestrated response with multi-system results
        """
        pass

# Manage: CodeRAG + MedicalRAG + GeneralRAG, choosing best combination per query
```

---

## 📚 Learning Path Recommendations

### Week 1: Foundation (Exercises 1-5)
- Master similarity calculations and basic text processing
- Understand TF-IDF and document chunking strategies
- Build query expansion and hybrid search capabilities

### Week 2: Core RAG (Exercises 6-10)
- Handle real-world document formats and conversational context
- Implement knowledge graphs and adaptive retrieval
- Add caching for production performance

### Week 3: Advanced RAG (Exercises 11-15)
- Build hierarchical and cross-document analysis
- Implement intelligent query routing and confidence scoring
- Create comprehensive evaluation frameworks

### Week 4: Enterprise RAG (Exercises 16-20)
- Optimize for different content types and languages
- Add temporal awareness and security layers
- Build multi-system orchestration capabilities

## 🎯 Success Metrics

By completing these exercises, you will have built:
- A complete RAG evaluation framework
- Multi-modal document processing pipeline
- Conversational and temporal-aware retrieval
- Enterprise security and privacy controls
- Multi-system RAG orchestration platform

Each exercise builds practical skills you'll use in production RAG systems!

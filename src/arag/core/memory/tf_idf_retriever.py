"""TF-IDF based retriever for memory recall.

Directly extracted from Re-MEMR1 memory module.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfRetriever:
    """
    A class to handle TF-IDF retrieval using a tokenizer.
    The vectorizer is fitted once upon initialization.
    """
    
    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: Optional tokenizer with a .tokenize() method.
                      If None, uses default whitespace tokenization.
        """
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.vectorizer = TfidfVectorizer(tokenizer=self._llm_tokenizer)
        else:
            self.vectorizer = TfidfVectorizer(tokenizer=self._simple_tokenizer)

    def _llm_tokenizer(self, text):
        """
        Custom tokenizer method that uses the instance's tokenizer.
        This method is passed to TfidfVectorizer.
        """
        lower_text = text.lower()
        
        # Use the LLM tokenizer
        tokens = self.tokenizer.tokenize(lower_text)
        
        # Normalize tokens to handle subword artifacts (like 'Ġ')
        normalized_tokens = [token.replace('Ġ', '') for token in tokens]
        return normalized_tokens
    
    def _simple_tokenizer(self, text):
        """Simple whitespace tokenizer as fallback."""
        return text.lower().split()

    def retrieve(self, query, corpus, top_k=3):
        """
        Retrieves the top_k most similar documents for a given query.
        
        Args:
            query: Query string
            corpus: List of document strings
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, similarity_score)
        """
        if not query or not corpus:
            return [(None, 0.0) for _ in range(top_k)]
        if not isinstance(corpus, list):
            corpus = list(corpus)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except Exception as e:
            return [(None, 0.0) for _ in range(top_k)]
        
        q_vec = self.vectorizer.transform([query])

        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        
        top_ids = np.argsort(sims)[::-1][:top_k]
        return [(corpus[i], sims[i]) for i in top_ids]
    
    def top1_retrieve(self, query, corpus):
        """Retrieve only the top-1 most similar document."""
        return self.retrieve(query, corpus, top_k=1)[0][0]

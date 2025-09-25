from collections import Counter

def tokenize_and_count(text:str) -> dict:
    """
    Tokenizes text and calculates the frequency of each unique token
    """

    #converting lowercase and split into whitespace
    tokens = text.lower().split()

    token_frequencies = Counter(tokens)

    return dict (token_frequencies)

text_rag = "RAG is a powerful technique. RAG stands for Retrieval Augmented Generation."
text_simple = "The quick brown fox jumps over the quick brown fox."

rag_counts = tokenize_and_count(text_rag)
simple_counts = tokenize_and_count(text_simple)

print(f"Input text: '{text_rag}'")
print(f"Token counts: '{rag_counts}'")

print(f"Input text: '{text_simple}'")
print(f"Token counts: '{simple_counts}'")

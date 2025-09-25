import math
from collections import Counter

def cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts based on token frequencies.
    Returns a value between 0 (no similarity) and 1 (identical).
    """
    #tokenize
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    #frequency
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)

    #get all unique tokens
    all_tokens = set(freq1.keys()) | set(freq2.keys()) 

    #create freq vector for both
    vector1 = [freq1.get(token, 0) for token in all_tokens]
    vector2 = [freq2.get(token, 0) for token in all_tokens]

    #dot product
    dot_product =  sum (v1 * v2 for  v1, v2 in  zip (vector1, vector2))

    #calculate magnitude
    magnitude1 = math.sqrt(sum(v1 * v1 for  v1 in vector1))
    magnitude2 = math.sqrt(sum(v2 * v2 for v2 in vector2))

    #handle edge cases
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    #cosine simliarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# Test cases
doc1 = "machine learning is powerful"
doc2 = "machine learning algorithms are powerful tools"
doc3 = "cats and dogs are pets"

print(f"Doc1: '{doc1}'")
print(f"Doc2: '{doc2}'")
print(f"Doc3: '{doc3}'")
print()

sim_1_2 = cosine_similarity(doc1, doc2)
sim_1_3 = cosine_similarity(doc1, doc3)
sim_2_3 = cosine_similarity(doc2, doc3)

print(f"Similarity between doc1 and doc2: {sim_1_2:.3f}")
print(f"Similarity between doc1 and doc3: {sim_1_3:.3f}")
print(f"Similarity between doc2 and doc3: {sim_2_3:.3f}")

# Let's also show what the vectors look like for doc1 and doc2
print("\n--- Debug Info for doc1 vs doc2 ---")
tokens1 = doc1.lower().split()
tokens2 = doc2.lower().split()
freq1 = Counter(tokens1)
freq2 = Counter(tokens2)
all_tokens = sorted(set(freq1.keys()) | set(freq2.keys()))

print(f"All unique tokens: {all_tokens}")
vector1 = [freq1.get(token, 0) for token in all_tokens]
vector2 = [freq2.get(token, 0) for token in all_tokens]
print(f"Vector1 (doc1): {vector1}")
print(f"Vector2 (doc2): {vector2}")

dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
magnitude1 = math.sqrt(sum(v1 * v1 for v1 in vector1))
magnitude2 = math.sqrt(sum(v2 * v2 for v2 in vector2))
print(f"Dot product: {dot_product}")
print(f"Magnitude1: {magnitude1:.3f}")
print(f"Magnitude2: {magnitude2:.3f}")
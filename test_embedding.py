from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "ERPNext is an ERP system",
    "ERPNext manages business operations",
    "Python is a programming language",
    "A company uses ERP software"
]

# Convert sentences into vectors
vectors = model.encode(sentences)

print("Number of sentences:", len(sentences))
print("Vector dimensions:", vectors.shape)

for sentence, vector in zip(sentences, vectors):
    print("\nSentence:", sentence)
    print("Vector:", vector[:5])
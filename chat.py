import chromadb

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# 1. Load embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="company_documents"
)


# 3. Connect to LM Studio
llm = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Replace with your actual loaded Qwen model ID
MODEL_NAME = "qwen/qwen2.5-vl-7b"


# 4. Retrieve relevant documents
def retrieve_documents(question):

    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


# 5. Generate answer using Qwen
def generate_answer(question):

    documents, metadatas = retrieve_documents(question)

    context = "\n\n".join([
        f"Source: {meta['source']}\n{doc}"
        for doc, meta in zip(documents, metadatas)
    ])

    prompt = f"""
    Answer the question using only the
    information provided in the context.

    If the answer is not available in the
    context, say that you don't know.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful company knowledge assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return answer, metadatas


# 6. Chat loop
while True:

    question = input("\nAsk a question: type 'exit' or 'quit' to stop\n> ")

    if question.lower() in ["exit", "quit"]:
        break

    answer, sources = generate_answer(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    for source in sources:
        print("-", source["source"])
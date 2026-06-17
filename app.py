from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ollama import chat
import faiss
import numpy as np

# =========================
# 1. Read PDF
# =========================

reader = PdfReader("data/sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

# =========================
# 2. Split into chunks
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")

# =========================
# 3. Create embeddings
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("\nEmbedding shape:")
print(embeddings.shape)

# =========================
# 4. Build FAISS index
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

print(f"\nVectors stored: {index.ntotal}")

# =========================
# 5. Ask user question
# =========================

question = input("\nAsk a question: ")

# =========================
# 6. Search relevant chunks
# =========================

question_embedding = model.encode([question])

D, I = index.search(
    np.array(question_embedding).astype("float32"),
    k=3
)

# =========================
# 7. Build context
# =========================

context = ""

for idx in I[0]:
    context += chunks[idx]
    context += "\n\n"

# =========================
# 8. Create prompt
# =========================

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
'I could not find that information in the document.'
"""

# =========================
# 9. Ask local LLM (Ollama)
# =========================

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# =========================
# 10. Print answer
# =========================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.message.content)
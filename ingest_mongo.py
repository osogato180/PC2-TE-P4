import os
from pathlib import Path

import pymongo
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pymongo.operations import SearchIndexModel

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
PDF_PATH = Path("material/p4_reglamento_biblioteca.pdf")
DB_NAME = "biblioteca_rag_db"
COL_NAME = "reglamento_chunks"
INDEX_NAME = "vector_index"


def leer_pdf(ruta: Path) -> str:
    reader = PdfReader(str(ruta))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def chunkear(texto: str):
    fijos = [texto[i : i + 500] for i in range(0, len(texto), 500) if texto[i : i + 500].strip()]
    recursivos = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(texto)
    return fijos, recursivos


def embed_doc(client: genai.Client, texto: str) -> list[float]:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return list(resp.embeddings[0].values)


def crear_indice(col, dims: int):
    modelo = SearchIndexModel(
        definition={
            "fields": [
                {"type": "vector", "path": "embedding", "similarity": "cosine", "numDimensions": dims},
                {"type": "filter", "path": "tecnica"},
            ]
        },
        name=INDEX_NAME,
        type="vectorSearch",
    )
    try:
        col.create_search_index(model=modelo)
        print("Índice vectorial creado.")
    except Exception as e:
        print(f"Aviso índice: {e}")


def main():
    if not GOOGLE_API_KEY or not MONGODB_URI:
        raise ValueError("Define GOOGLE_API_KEY y MONGODB_URI como variables de entorno.")

    client_genai = genai.Client(api_key=GOOGLE_API_KEY)
    client_mongo = pymongo.MongoClient(MONGODB_URI)
    col = client_mongo[DB_NAME][COL_NAME]

    texto = leer_pdf(PDF_PATH)
    fijos, recursivos = chunkear(texto)
    print(f"Chars: {len(texto)} | Fijos: {len(fijos)} | Recursivos: {len(recursivos)}")

    primer_emb = embed_doc(client_genai, fijos[0])
    crear_indice(col, len(primer_emb))

    col.delete_many({"tecnica": {"$in": ["fijo_500", "recursivo_500_50"]}})
    docs = []

    for i, ch in enumerate(fijos):
        emb = primer_emb if i == 0 else embed_doc(client_genai, ch)
        docs.append({"chunk_id": f"fijo_{i}", "tecnica": "fijo_500", "texto": ch, "embedding": emb})

    for i, ch in enumerate(recursivos):
        docs.append({
            "chunk_id": f"rec_{i}",
            "tecnica": "recursivo_500_50",
            "texto": ch,
            "embedding": embed_doc(client_genai, ch),
        })

    col.insert_many(docs)
    print(f"Insertados {len(docs)} documentos en {DB_NAME}.{COL_NAME} (dims={len(primer_emb)}).")


if __name__ == "__main__":
    main()

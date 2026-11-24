import csv
import os
from pathlib import Path

import pandas as pd
import pymongo
from google import genai
from google.genai import types
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

st.set_page_config(page_title="P4 RAG Biblioteca", layout="wide")
st.title("RAG Reglamento de Biblioteca Digital (P4)")


def get_secret(name: str):
    if os.getenv(name):
        return os.getenv(name)
    try:
        if name in st.secrets:
            return st.secrets[name]
        if "app" in st.secrets and name in st.secrets["app"]:
            return st.secrets["app"][name]
    except Exception:
        pass
    return None


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
MONGODB_URI = get_secret("MONGODB_URI")

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("Configura GOOGLE_API_KEY y MONGODB_URI en Secrets o entorno.")
    st.stop()


@st.cache_resource
def init_clients():
    g_client = genai.Client(api_key=GOOGLE_API_KEY)
    m_client = pymongo.MongoClient(MONGODB_URI)
    col = m_client["biblioteca_rag_db"]["reglamento_chunks"]
    return g_client, col


client_genai, collection = init_clients()

PDF_PATH = Path("material/p4_reglamento_biblioteca.pdf")
CSV_PATH = Path("datos/p4_preguntas.csv")
INDEX_NAME = "vector_index"


def embed_query(texto: str) -> list[float]:
    resp = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(resp.embeddings[0].values)


def buscar_top3(pregunta: str, tecnica: str) -> list[dict]:
    q_emb = embed_query(pregunta)
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": q_emb,
                "filter": {"tecnica": tecnica},
                "numCandidates": 50,
                "limit": 3,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "tecnica": 1,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def calc_metricas(posiciones: list[int | None]) -> dict:
    hit = sum(1 for p in posiciones if p in (1, 2, 3)) / len(posiciones)
    mrr = sum((1.0 / p) for p in posiciones if p in (1, 2, 3)) / len(posiciones)
    return {"Hit@3": round(hit, 4), "MRR": round(mrr, 4)}


def responder(pregunta: str, fragmentos: list[dict]) -> str:
    ctx = "\n\n".join(f"[{f['chunk_id']}] {f['texto']}" for f in fragmentos)
    prompt = (
        "Responde SOLO usando el CONTEXTO. Si no consta en el CONTEXTO, responde exactamente: NO_CONSTA.\n"
        "Cita entre corchetes el chunk_id usado (ej. [rec_1]).\n\n"
        f"CONTEXTO:\n{ctx}\n\nPREGUNTA:\n{pregunta}"
    )
    res = client_genai.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return (res.text or "").strip()


t1, t2, t3 = st.tabs(["4.1 Resumen Chunks", "4.2 Hit@3 y MRR", "4.3 Chat RAG"])

with t1:
    if PDF_PATH.exists():
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(PDF_PATH)).pages).strip()
        fijos = [txt[i : i + 500] for i in range(0, len(txt), 500) if txt[i : i + 500].strip()]
        recs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(txt)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chars PDF", len(txt))
        c2.metric("Fijo 500", len(fijos))
        c3.metric("Recursivo 500/50", len(recs))
    else:
        st.warning("Falta material/p4_reglamento_biblioteca.pdf")

with t2:
    preguntas = []
    if CSV_PATH.exists():
        with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
            preguntas = [r.get("pregunta") or next(iter(r.values())) for r in csv.DictReader(f)]
    preguntas = preguntas[:4] if len(preguntas) >= 4 else [
        "Pregunta 1 del reglamento",
        "Pregunta 2 del reglamento",
        "Pregunta 3 del reglamento",
        "Pregunta 4 del reglamento",
    ]
    tec = st.selectbox("Técnica", ["fijo_500", "recursivo_500_50"])
    q = st.selectbox("Pregunta", preguntas)
    if st.button("Consultar Top-3"):
        for i, h in enumerate(buscar_top3(q, tec), 1):
            st.markdown(f"**#{i} `{h['chunk_id']}`** (score `{h['score']:.4f}`): {h['texto']}")

    st.caption("Posiciones relevantes (1, 2, 3 o None) para las 4 preguntas:")
    cf = st.columns(4)
    cr = st.columns(4)
    pf, pr = [], []
    for i in range(4):
        pf.append(None if cf[i].selectbox(f"Fijo Q{i+1}", ["None", "1", "2", "3"], key=f"pf{i}") == "None" else int(cf[i].selectbox(f"Fijo Q{i+1}", ["None", "1", "2", "3"], key=f"pf_v{i}")))
        pr.append(None if cr[i].selectbox(f"Rec Q{i+1}", ["None", "1", "2", "3"], key=f"pr{i}") == "None" else int(cr[i].selectbox(f"Rec Q{i+1}", ["None", "1", "2", "3"], key=f"pr_v{i}")))

    df = pd.DataFrame([
        {"Tecnica": "fijo_500", "Posiciones": str(pf), **calc_metricas(pf)},
        {"Tecnica": "recursivo_500_50", "Posiciones": str(pr), **calc_metricas(pr)},
    ])
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Tecnica")[["Hit@3", "MRR"]])

with t3:
    tec_c = st.radio("Técnica chat", ["recursivo_500_50", "fijo_500"], horizontal=True)
    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    for m in st.session_state.msgs:
        st.chat_message(m["role"]).markdown(m["content"])

    prompt_in = st.chat_input("Escribe tu pregunta...")
    if prompt_in:
        st.session_state.msgs.append({"role": "user", "content": prompt_in})
        st.chat_message("user").markdown(prompt_in)
        with st.chat_message("assistant"):
            frags = buscar_top3(prompt_in, tec_c)
            out = responder(prompt_in, frags)
            st.markdown(out)
            with st.expander("Fragmentos recuperados"):
                for f in frags:
                    st.markdown(f"**{f['chunk_id']}** (`{f['score']:.4f}`): {f['texto']}")
            st.session_state.msgs.append({"role": "assistant", "content": out})

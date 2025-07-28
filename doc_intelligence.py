import os
import json
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIntelligence:
    def __init__(self, input_dir="./outputs", pdfs_dir="./pdf", output_json_path="./json",  embedding_model_name='all-MiniLM-L6-v2'):
        self.input_dir = input_dir
        self.pdfs_dir = pdfs_dir
        self.output_json_path = output_json_path
        os.makedirs(self.output_json_path, exist_ok=True)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("Model loaded.")

    def _parse_outline_json(self, pdf_path):
        json_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
        json_path = os.path.join(self.input_dir, json_name)
        if not os.path.exists(json_path):
            print(f"Warning: Could not find outline JSON for {pdf_path}")
            return None, ""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        outline_entries = data.get("outline", [])
        if not outline_entries:
            return None, data.get("title", "")
        outline_df = pd.DataFrame(outline_entries)
        return outline_df, data.get("title", "")

    def _chunk_document(self, pdf_path):
        outline_df, title = self._parse_outline_json(pdf_path)
        if outline_df is None or outline_df.empty:
            return []
        chunks = []
        for _, row in outline_df.iterrows():
            if row["level"] in ["H1", "H2", "H3"]:
                chunks.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": row.get("page", 0),
                    "section_title": row["text"],
                    "content": row["text"]  # Currently using title only, can be extended
                })
        return chunks

    def find_relevant_sections(self, persona, job_to_be_done):
        document_names = [f for f in os.listdir(self.pdfs_dir) if f.lower().endswith('.pdf')]
        pdf_paths = [os.path.join(self.pdfs_dir, name) for name in document_names]

        if not pdf_paths:
            print("Warning: No PDFs found in the specified directory.")
            return {"error": "No PDFs found."}

        query = f"As a {persona}, my goal is to {job_to_be_done}."
        query_embedding = self.embedding_model.encode([query])
        all_chunks = []

        for pdf_path in pdf_paths:
            chunks = self._chunk_document(pdf_path)
            if chunks:
                all_chunks.extend(chunks)

        if not all_chunks:
            return {"error": "No content could be extracted."}

        chunk_contents = [chunk['content'] for chunk in all_chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_contents)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

        for i, chunk in enumerate(all_chunks):
            chunk['relevance_score'] = similarities[i]

        sorted_chunks = sorted(all_chunks, key=lambda x: x['relevance_score'], reverse=True)

        output = {
            "metadata": {
                "input_documents": [os.path.basename(p) for p in pdf_paths],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

        for i, chunk in enumerate(sorted_chunks[:5]):
            output["extracted_sections"].append({
                "document": chunk['document'],
                "section_title": chunk['section_title'],
                "importance_rank": i + 1,
                "page_number": chunk['page_number']
            })
            output["subsection_analysis"].append({
                "document": chunk['document'],
                "refined_text": chunk['content'],
                "page_number": chunk['page_number']
            })

        output_path = os.path.join(self.output_json_path, "summary_output.json")
        with open(output_path, 'w', encoding='utf-8') as jf:
            json.dump(output, jf, indent=4, ensure_ascii=False)

        print(f"Output written to: {output_path}")
        return output

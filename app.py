import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline

app = Flask(__name__)

# Load models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Directory containing PDF files
DATA_FOLDER = "data"

# Extract text from PDFs
def extract_text_from_pdfs(folder_path):
    documents = []
    titles = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append(text)
                titles.append(file_name)
    return documents, titles

# Build FAISS index
def build_faiss_index(documents):
    embeddings = embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Preprocess and index the data
documents, doc_titles = extract_text_from_pdfs(DATA_FOLDER)
faiss_index = build_faiss_index(documents)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    error_message = None
    relevant_doc = None
    if request.method == "POST":
        question = request.form.get("question").strip()
        if not question:
            error_message = "Please enter a valid question."
        else:
            try:
                # Generate question embedding
                question_embedding = embedding_model.encode([question], convert_to_tensor=True).cpu().numpy()
                
                # Search for the most relevant document
                distances, indices = faiss_index.search(question_embedding, k=1)
                if distances[0][0] < 0.8:  # Adjust threshold for relevance
                    error_message = "Not a relevant question. Please ask a question related to the uploaded documents."
                else:
                    relevant_idx = indices[0][0]
                    relevant_doc = documents[relevant_idx]
                    
                    # Generate a descriptive answer
                    qa_result = qa_model({"question": question, "context": relevant_doc})
                    answer = qa_result['answer']
            except Exception as e:
                error_message = f"Error processing your question: {str(e)}"

    return render_template("index.html", answer=answer, error_message=error_message, doc_title=doc_titles[indices[0][0]] if relevant_doc else None)

if __name__ == "__main__":
    app.run(debug=True)

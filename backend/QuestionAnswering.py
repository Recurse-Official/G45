import os
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import re

# Flask setup
app = Flask(__name__)
CORS(app)

# Folder where the PDFs are stored
folder_path = os.path.expanduser("~/Downloads/Circulars")

# Function to convert PDF pages to text using OCR
def ocr_from_pdf(file_path):
    pages = convert_from_path(file_path, 300)
    text = ""
    for page_number, page in enumerate(pages, 1):
        page_text = pytesseract.image_to_string(page)
        text += f"--- Page {page_number} ---\n{page_text}\n"
    return text

# Function to extract dates from text
def extract_date_from_text(text):
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',   # Matches formats like 12/31/2020 or 31-12-2020
        r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})',  # Matches formats like December 31, 2020
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1)
            for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
    return datetime.min  # Default if no date found

# Extract text and dates from PDFs
def extract_text_from_pdfs(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = ocr_from_pdf(file_path)
            date = extract_date_from_text(text)
            documents.append({"filename": filename, "text": text, "date": date})
    return documents

# Load documents
documents = extract_text_from_pdfs(folder_path)

# Initialize models
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Create document embeddings
document_embeddings = [embedding_model.encode(doc['text']) for doc in documents]
dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

# Search function
def retrieve_answer(query, embedding_model, index, documents):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k=5)

    # Collect candidate documents
    candidates = [
        {"text": documents[i]['text'], "filename": documents[i]['filename'], "date": documents[i]['date'], "distance": distances[0][idx]}
        for idx, i in enumerate(indices[0])
    ]

    # Sort candidates by date (most recent first)
    candidates = sorted(candidates, key=lambda x: x['date'], reverse=True)

    return candidates

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query', '')
    user_date = data.get('date', None)

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Retrieve candidate answers
        candidates = retrieve_answer(query, embedding_model, index, documents)

        # If a user-specified date is provided, filter candidates
        if user_date:
            user_date = datetime.strptime(user_date, "%Y-%m-%d")
            candidates = [c for c in candidates if c['date'] == user_date]

            if not candidates:
                return jsonify({'error': 'No documents found for the specified date'}), 404

        # If multiple dates are available and no date is specified, return the dates
        if len(set(c['date'] for c in candidates)) > 1 and not user_date:
            dates = list({c['date'].strftime("%Y-%m-%d") for c in candidates})
            return jsonify({'message': 'Multiple dates found. Please specify a date.', 'dates': dates})

        # Default to the most recent document
        best_match = candidates[0]
        context = best_match['text']
        answer = qa_pipeline(question=query, context=context)['answer']

        return jsonify({'answer': answer, 'filename': best_match['filename'], 'date': best_match['date'].strftime("%Y-%m-%d")})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

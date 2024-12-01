import os
import random
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Suppress the huggingface_hub symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load pre-trained models
logging.info("Loading models...")
question_model_name = "valhalla/t5-base-qg-hl"
qa_model_name = "deepset/roberta-large-squad2"

try:
    question_tokenizer = T5Tokenizer.from_pretrained(question_model_name)
    question_model = T5ForConditionalGeneration.from_pretrained(question_model_name)
    qa_pipeline = pipeline("question-answering", model=qa_model_name)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

# Default context from Circulars
DEFAULT_CONTEXT = ""
CIRCULARS_FOLDER = os.path.expanduser("~/Downloads/Circulars")

def load_circulars_text(folder_path):
    """Load and combine text from all PDFs in the Circulars folder."""
    global DEFAULT_CONTEXT
    try:
        logging.info("Loading text from Circulars folder...")
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "rb") as pdf_file:
                    text = extract_text_from_pdf(pdf_file)
                    combined_text += clean_text(text) + "\n"
        DEFAULT_CONTEXT = combined_text.strip()
        logging.info("Circulars text loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading Circulars folder: {str(e)}")

def extract_text_from_pdf(pdf):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        raise

def clean_text(text):
    """Clean the extracted text."""
    return text.replace('\n', ' ').strip()

def generate_questions(text, total_questions=16):
    """Generate questions from the given text."""
    try:
        wrapped_text = textwrap.wrap(text, width=512)
        questions = []
        for i in range(min(total_questions, len(wrapped_text))):
            input_text = f"generate questions: {wrapped_text[i]}"
            input_ids = question_tokenizer.encode(input_text, return_tensors="pt")
            outputs = question_model.generate(input_ids, max_length=64, num_beams=4, do_sample=False)
            question = question_tokenizer.decode(outputs[0], skip_special_tokens=True)
            questions.append(question)
        return questions
    except Exception as e:
        logging.error(f"Error generating questions: {str(e)}")
        raise

def answer_questions(questions, context):
    """Answer questions using the provided context."""
    try:
        answers = []
        for question in questions:
            result = qa_pipeline(question=question, context=context)
            answers.append({"question": question, "answer": result.get('answer', 'No answer found').capitalize()})
        return answers
    except Exception as e:
        logging.error(f"Error answering questions: {str(e)}")
        raise

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload or fallback to Circulars data."""
    try:
        context = DEFAULT_CONTEXT

        # Check if a file is uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':  # If file is provided
                logging.info("Processing uploaded file.")
                text = extract_text_from_pdf(file)
                context = clean_text(text)
            else:
                logging.info("No file uploaded. Using default Circulars context.")
        else:
            logging.info("No file uploaded. Using default Circulars context.")

        # If context is still empty (e.g., no file and no Circulars data), return error
        if not context:
            logging.error("No context available for question answering.")
            return jsonify({"error": "No input context provided"}), 400

        # Generate questions and answers from the context (uploaded file or Circulars)
        questions = generate_questions(context)
        answers = answer_questions(questions, context)

        return jsonify({"questions": questions, "answers": answers})
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Preload the Circulars context before starting the server
    load_circulars_text(CIRCULARS_FOLDER)
    app.run(debug=True)

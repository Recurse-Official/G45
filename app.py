from flask import Flask, render_template, request
import PyPDF2
from transformers import pipeline

app = Flask(__name__)

# Initialize the question-answering pipeline with a more advanced model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to split text into smaller chunks for better processing
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

@app.route("/", methods=["GET", "POST"])
def home():
    answers = []
    error_message = None
    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")
        questions = request.form.get("questions").strip().splitlines()

        # Check if the user has uploaded a PDF
        if not pdf_file:
            error_message = "Please upload a PDF file."
            return render_template("index.html", error_message=error_message)

        # Save the PDF file temporarily to extract text
        pdf_path = f"temp.pdf"
        pdf_file.save(pdf_path)

        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_path)

        # Split the text into smaller chunks
        chunks = list(split_text_into_chunks(pdf_text))

        # Loop through each question and get the answer
        for question in questions:
            full_answer = ""
            for chunk in chunks:
                try:
                    # Get the answer using Hugging Face pipeline
                    result = qa_pipeline(
                        {
                            'context': chunk,
                            'question': question
                        },
                        max_answer_len=200  # Allow longer, descriptive answers
                    )
                    full_answer += result['answer'] + " "
                except Exception as e:
                    error_message = f"Error processing question: {str(e)}"
                    break

            # Append the final answer for the question
            answers.append({
                "question": question,
                "answer": full_answer.strip()
            })

    return render_template("index.html", answers=answers, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)

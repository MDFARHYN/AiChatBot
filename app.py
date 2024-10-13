"""
When running on Jupyter Notebook, don’t forget to uncomment this line to automatically install all required packages:
!pip install gradio transformers datasets sentence-transformers pandas
"""
import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Initialize models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load toxicity and zero-shot classifiers
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Similarity threshold and max token limits
SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_LENGTH = 512
MAX_QUESTION_LENGTH = 128

# Helper function to preprocess text
def preprocess_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(re.sub(r'[^\w\s]', '', word) for word in words)

# Load and process the CSV file
def load_csv(file_path):
    try:
        # Try loading with UTF-8 encoding, fallback to other encodings if necessary
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
        
        if data.empty or "Question/Issue" not in data.columns or "Answer/Response" not in data.columns:
            return None, None, "Missing required columns in the CSV file."

        data["combined"] = (
            "Question/Issue: " + data["Question/Issue"].fillna('') +
            "\nAnswer/Response: " + data["Answer/Response"].fillna('')
        )

        chunk_texts = data["combined"].tolist()
        embeddings = embedding_model.encode([preprocess_text(text) for text in chunk_texts], convert_to_tensor=True)

        return chunk_texts, embeddings, None
    except Exception as e:
        return None, None, f"Error during CSV loading or processing: {str(e)}"

# Answer questions based on the CSV data
def answer_question(file, question):
    if len(question) > MAX_QUESTION_LENGTH:
        question = question[:MAX_QUESTION_LENGTH]

    chunk_texts, embeddings, error_message = load_csv(file.name)
    if error_message:
        return error_message

    try:
        # Step 1: Detect if the content is toxic
        toxicity_results = toxicity_classifier(question)
        toxicity_score = toxicity_results[0]['score']
        toxicity_label = toxicity_results[0]['label']

        toxicity_message = f"Toxicity Score: {toxicity_score} (Label: {toxicity_label})"

        # Step 2: Zero-shot classification for inappropriate content
        candidate_labels = [
            "illegal activity", "drug dealing", "selling weapons", "money laundering",
            "cybercrime", "hacking services", "terrorism", "violence", "fraud", 
            "human trafficking", "child exploitation", "blackmail", "safe question",
            "protecting against cybercrime", "security measures", "learning about safety", "general advice"
        ]
        zero_shot_result = zero_shot_classifier(question, candidate_labels=candidate_labels)
        highest_label = zero_shot_result['labels'][0]
        highest_score = zero_shot_result['scores'][0]

        zero_shot_message = f"Zero-Shot Classification: {highest_label} (Score: {highest_score})"

        # Check for toxicity and illegal content
        if toxicity_label == 'toxic' and toxicity_score > 0.6:
            return f"Your question contains inappropriate content. Please rephrase and try again.\n\n{toxicity_message}\n{zero_shot_message}"

        if highest_label in ["illegal activity", "drug dealing", "selling weapons", "money laundering",
                             "cybercrime", "hacking services", "terrorism", "violence", "fraud", 
                             "human trafficking", "child exploitation", "blackmail"] and highest_score > 0.3:
            return f"Your question appears to discuss illegal or inappropriate content. Please rephrase and try again.\n\n{toxicity_message}\n{zero_shot_message}"

        # Step 3: Find the most relevant answer
        question = preprocess_text(question)
        question_embedding = embedding_model.encode(question, convert_to_tensor=True)

        similarities = util.cos_sim(question_embedding, embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:3]

        if similarities[top_indices[0]] < SIMILARITY_THRESHOLD:
            return f"No relevant information found.\n\n{toxicity_message}\n{zero_shot_message}"

        best_context = chunk_texts[top_indices[0]]
        if len(best_context) > MAX_CONTEXT_LENGTH:
            best_context = best_context[:MAX_CONTEXT_LENGTH]

        # Extract parts of the answer
        question_issue = re.search(r"Question/Issue:\s*(.*?)\s*Answer/Response:", best_context, re.DOTALL)
        answer_response = re.search(r"Answer/Response:\s*(.*?)$", best_context, re.DOTALL)

        question_issue_text = question_issue.group(1).strip() if question_issue else ""
        answer_response_text = answer_response.group(1).strip() if answer_response else ""

        # Return structured context and answer with diagnostic info
        structured_context = (
            f"Question/Issue: {question_issue_text}\n\n"
            f"Answer/Response: {answer_response_text}"
        )
        return f"Context:\n{structured_context}\n\nAnswer:\n{answer_response_text}\n\n{toxicity_message}\n{zero_shot_message}"
    except Exception as e:
        print("Error in answer_question function:", e)
        return "Error encountered while processing your request."

# Set up Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# CSV-based Chatbot Support System")
    csv_file = gr.File(label="Upload CSV File")
    question = gr.Textbox(label="Enter your question")
    answer_text = gr.Textbox(label="Answer from CSV Data", lines=10)

    submit_btn = gr.Button("Get Answer")
    submit_btn.click(fn=answer_question, inputs=[csv_file, question], outputs=answer_text)

demo.launch(share=True)

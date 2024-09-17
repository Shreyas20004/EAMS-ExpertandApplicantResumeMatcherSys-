import os
from flask import Flask, request, jsonify, render_template
import docx
import PyPDF2
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Define directories
UPLOAD_FOLDER = 'uploads'
EXPERT_FOLDER = 'expert_portfolios'

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPERT_FOLDER, exist_ok=True)

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')


# Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


# Extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# Extract text from TXT
def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read()


# Unified function to extract text based on file extension
def extract_text_from_file(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".docx":
        return extract_text_from_docx(file_path)
    elif extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")


# Preprocess text: tokenization and stopword removal
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens
    return [word for word in tokens if word not in stopwords.words('english')]


# Extract top N keywords using TF-IDF
def extract_keywords(text, top_n=10):
    preprocessed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(max_features=top_n)
    X = vectorizer.fit_transform([" ".join(preprocessed_text)])
    return vectorizer.get_feature_names_out()


# Function to compute domain match (e.g., Jaccard similarity)
def calculate_domain_similarity(applicant_domains, expert_domains):
    intersection = set(applicant_domains).intersection(set(expert_domains))
    union = set(applicant_domains).union(set(expert_domains))
    return len(intersection) / len(union) if union else 0


# Load expert profiles from expert portfolios
def load_expert_profiles():
    expert_profiles = []
    for file_name in os.listdir(EXPERT_FOLDER):
        file_path = os.path.join(EXPERT_FOLDER, file_name)
        text = extract_text_from_file(file_path)
        domains = extract_keywords(text, top_n=5)
        expert_profiles.append({
            'name': os.path.splitext(file_name)[0],
            'domains': domains
        })
    return expert_profiles


# Match the extracted domains from the applicant with experts
def match_experts_with_applicant(applicant_domains, experts):
    for expert in experts:
        domain_similarity = calculate_domain_similarity(applicant_domains, expert['domains'])
        expert['domain_similarity'] = domain_similarity
    return sorted(experts, key=lambda x: x['domain_similarity'], reverse=True)


# Flask route to handle file uploads for resumes
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            # Extract text from the uploaded resume file
            extracted_text = extract_text_from_file(file_path)

            # Extract applicant's domains using NLP
            applicant_domains = extract_keywords(extracted_text, top_n=5)

            # Load expert profiles from portfolios
            expert_profiles = load_expert_profiles()

            # Match and rank experts
            ranked_experts = match_experts_with_applicant(applicant_domains, expert_profiles)

            # Prepare the result
            result = [{'name': expert['name'], 'domain_similarity': expert['domain_similarity']} for expert in
                      ranked_experts]
            return jsonify({'status': 'success', 'experts': result})

        except ValueError as e:
            return jsonify({'error': str(e)})


# Flask route to handle file uploads for expert portfolios
@app.route('/upload_expert', methods=['POST'])
def upload_expert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(EXPERT_FOLDER, file.filename)
        file.save(file_path)

        return jsonify({'status': 'success', 'message': f'Expert portfolio {file.filename} uploaded successfully'})


# Route for the main HTML page
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)

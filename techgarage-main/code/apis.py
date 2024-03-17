from datetime import datetime
import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import uuid
from flask_sqlalchemy import SQLAlchemy


load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

# Path to the uploaded_documents folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_documents')
REPORT_FOLDER = os.path.join(os.getcwd(), 'reports_documents')
# Dictionary to store the mapping of taskId to userId
TASK_ID_MAPPING = {}

def generate_task_id():
    # Generate a UUID (Universally Unique Identifier)
    task_id = uuid.uuid4()
    return str(task_id)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    year = db.Column(db.Integer)
    user_id = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_path = db.Column(db.String(255), nullable=False)
                          
    def __repr__(self):
        return f"UploadedFile('{self.id}', '{self.filename}', '{self.year}', '{self.user_id}', '{self.created_at}')"
    
class TaskFileMapping(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    report_year = db.Column(db.String(4), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    file_path = db.Column(db.String(255), nullable=False)

# Function to save the uploaded file
def save_uploaded_file(file, year, file_path):
    if file:
        year_folder = os.path.join(UPLOAD_FOLDER, str(year))
        if not os.path.exists(year_folder):
            os.makedirs(year_folder)
    
        # Save the file to the year-specific folder
        #file_path = os.path.join(year_folder, file.filename)
        file.save(file_path)

        # Save file information to the database
        new_uploaded_file = UploadedFile(filename=file.filename, year=year, user_id=1, file_path=file_path)
        db.session.add(new_uploaded_file)
        db.session.commit()

        return new_uploaded_file.id
    else:
        return jsonify({"error": "No file provided"}), 400

# Functions from the provided code snippet
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """Answer the questions as precise as possible using the provided context. If the answer is not contained in the context ,say "answer is not available in context" \n\n Context: \n {context}?\n Question: \n {question} \n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ.get("GOOGLE_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# API endpoints
#Upload the ESG Reports, one or many files, sample files which are given
@app.route('/esgreports/upload', methods=['POST'])
def upload_esg_reports():
    pdf_files = request.files.getlist('documentName')
    year_of_report = request.form.get('YearOfReport')

    # Save the uploaded files and file information to the database
    uploaded_files = []

    year_folder = os.path.join(UPLOAD_FOLDER, str(year_of_report))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    for file in pdf_files:
        # Save the file to the year-specific folder
        file_path = os.path.join(year_folder, file.filename)
        tracking_id = save_uploaded_file(file, year_of_report, file_path)
        uploaded_files.append(file.filename)

    # Process the uploaded PDF files
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Example response
    response = {
        "status": "success",
        "message": "Document Uploaded Successfully",
        "trackerId": tracking_id
    }
    return jsonify(response)

#retrieve all the document uploaded for a year
@app.route('/esgreports/retrieve', methods=['POST'])
def retrieve_esg_reports():
    report_year = request.json.get('reportYear')

    # Query the database for files uploaded in the specified year
    files_in_year = UploadedFile.query.filter_by(year=report_year).all()

    # If no files found, return an error response
    if not files_in_year:
        return jsonify({"error": f"No documents found for year {report_year}"}), 404

    # Prepare the response with filenames and file paths
    response = {
        "documents": [{"filename": file.filename, "filepath": file.file_path} for file in files_in_year]
    }
    return jsonify(response)

#Upload the Survey Questionnire
@app.route('/questionnaire/generatefirstdraft/pdf', methods=['POST'])
def generate_first_draft_pdf():
    # Use request.files to access the uploaded questionnaire
    survey_questionnaire_files = request.files.getlist('SurveyQuestionnaireDocumentName')
    generate_report_for_year = request.form.get('generateReportforYear')
    user_id = request.form.get('userId')

    year_folder = os.path.join(REPORT_FOLDER, str(generate_report_for_year))
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    
    task_id = generate_task_id()
    for file in survey_questionnaire_files:
        file_path = os.path.join(year_folder, file.filename)
        save_uploaded_file(file, generate_report_for_year, file_path)
        mapping = TaskFileMapping(task_id=task_id, user_id=user_id, file_name=file.filename, report_year=generate_report_for_year, file_path=file_path)
        db.session.add(mapping)
    
    db.session.commit()

    # Process the uploaded PDF files
    raw_text = get_pdf_text(survey_questionnaire_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Example response
    response = {
        "taskid": task_id,
        "status": "in_progress",
        "createAt": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(response)

#Find status of the Survey Questionnire
@app.route('/questionnaire/generatefirstdraft/pdf/<report_year>/<task_id>/status', methods=['GET'])
def get_first_draft_status(report_year, task_id):
    # Check if the task_id exists in the TaskFileMapping table
    mappings = TaskFileMapping.query.filter_by(task_id=task_id).all()
    if not mappings:
        return jsonify({"error": "Invalid task ID"}), 404

    # Get the user_id associated with the task_id
    user_id = mappings[0].user_id

    # Check if reports for the specified year exist in REPORT_FOLDER
    files_for_year = TaskFileMapping.query.filter_by(user_id=user_id, report_year=report_year).all()
    if not files_for_year:
        return jsonify({"error": "No reports found for the specified year"}), 404

    # Get the status of the task from your task management system or database
    status = "done"  # Placeholder status

    # Get the file paths from the database query result
    base_folder = os.path.join(REPORT_FOLDER, str(report_year))
    report_paths = [os.path.join(base_folder, file.file_name) for file in files_for_year]

    # Example response
    response = {
        "status": "success",
        "message": "Reports found for the specified year",
        "report_paths": report_paths,
        "task_status": status
    }
    return jsonify(response)

#Download the First Draft Report
@app.route('/firstdraftreport/download/result/<report_year>', methods=['GET'])
def download_first_draft_report(report_year):
    if report_year not in REPORT_FOLDER:
        return jsonify({"error": "Report not found for the given year"}), 404

    reports = REPORT_FOLDER[report_year]
    # Assuming you want to return the file paths
    report_paths = [report["documentName"] for report in reports]

    # Example response
    response = {
        "status": "success",
        "message": "Reports found for the given year",
        "report_paths": report_paths
    }
    return jsonify(response)

#Retrieve response for the specific question
@app.route('/questionnaire/generatefirstdraft/generateAnswer', methods=['POST'])
def generate_question_answer():
    report_year = request.json.get('reportYear')
    input_question = request.json.get('inputQuestion')

    # Get the answer using the user_input function
    answer = user_input(input_question)

    # Return the appropriate response
    response = {
        "reportYear": report_year,
        "questionnireSummary": {
            "response": answer,
            # Add other necessary fields
        }
    }
    return jsonify(response)

#Ping service
@app.route('/esgreports/keepalive/ping', methods=['GET'])
def ping():
    # Perform a basic health check
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = "API is running"
    return jsonify({"status": "OK", "message": message, "timestamp": current_time})

def create_tables():
    with app.app_context():
        #db.drop_all()
        db.create_all()

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
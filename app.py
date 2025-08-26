# Import necessary libraries
from flask import Flask, render_template, request   # Flask: web framework; render_template: load HTML; request: handle form data
from werkzeug.utils import secure_filename           # secure_filename: ensures safe file names for uploads
import os                                           # os: handle folders and paths
import pickle                                       # pickle: load your saved ML objects
from pdfplumber import open as pdf_open            # pdfplumber: read PDF files
from docx import Document                           # python-docx: read Word (.docx) files
import re

# ----------------------------
# Step 0: Define cleanResume function
# ----------------------------
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s*', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText



# ----------------------------
# Step 1: Initialize Flask app
# ----------------------------
app = Flask(__name__)  # Creates the Flask application

# ----------------------------
# Step 2: Configure upload folder
# ----------------------------
app.config['UPLOAD_FOLDER'] = 'uploads'            # Folder where uploaded resumes will be stored temporarily
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Make sure the folder exists, create if it doesn't

# ----------------------------
# Step 3: Load ML objects
# ----------------------------
# Load the trained KNN classifier
with open('models/clf.pkl', 'rb') as f:
    clf = pickle.load(f)

# Load the trained TF-IDF vectorizer
with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load the LabelEncoder that maps numeric labels to actual category names
with open('models/encoder.pkl', 'rb') as f:
    le = pickle.load(f)





# ----------------------------
# Step 4: Define prediction function
# ----------------------------
def pred(input_resume):
    """
    This function takes the resume text, cleans it, converts it to features using TF-IDF,
    predicts the category using KNN, and returns the human-readable category name.
    """
    cleaned_text = cleanResume(input_resume)                 # Clean the resume text (remove stopwords, punctuation, lowercase, etc.)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()  # Convert text to numerical features using TF-IDF, convert to dense array
    predicted_category = clf.predict(vectorized_text)        # Predict the numeric category using KNN
    predicted_category_name = le.inverse_transform(predicted_category)  # Convert numeric label to actual category name
    return predicted_category_name[0]                        # Return the first (and only) predicted category

# ----------------------------
# Step 5: Home route
# ----------------------------
@app.route('/')
def home():
    """
    Renders the main HTML page where the user can upload a resume.
    """
    return render_template('index.html')                     # Load the index.html template from the templates folder

# ----------------------------
# Step 6: Predict route
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict_resume():
    """
    Handles the uploaded resume file, extracts its text, predicts the category, 
    and returns the result to the browser.
    """
    if 'resume' not in request.files:                        # Check if a file was uploaded
        return "No file uploaded"

    file = request.files['resume']                            # Get the uploaded file
    if file.filename == '':                                   # Check if a file was selected
        return "No selected file"

    # Save the uploaded file safely to the uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    # ----------------------------
    # Step 6a: Extract text based on file type
    # ----------------------------
    text = ''
    if file.filename.endswith('.txt'):                        # If file is a text file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file.filename.endswith('.pdf'):                      # If file is a PDF
        with pdf_open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
    elif file.filename.endswith('.docx'):                     # If file is a Word document
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:                                                     # Unsupported file type
        return "Unsupported file type. Please upload .txt, .pdf, or .docx"

    # ----------------------------
    # Step 6b: Predict the category
    # ----------------------------
    #result = pred(text)                                       # Call the pred() function to get category prediction

    # Return the result in browser
    #return render_template('index.html', result=result)

    result = pred(text)
    os.remove(filepath)  # keeps uploads folder clean
    return render_template('index.html', result=result)

# ----------------------------
# Step 7: Run the Flask app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)  # Start the Flask server in debug mode (shows errors in terminal)

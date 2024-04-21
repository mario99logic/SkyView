from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Define the path for the uploaded images. You may need to adjust the path depending on your project structure.
app.config['UPLOAD_FOLDER'] = '../uploadedImages'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def home():
    # This route will render your home page template.
    return render_template('index.html')


# Function to check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('images')  # Get the list of uploaded files

        # Check if any files are uploaded
        if not uploaded_files:
            return 'No files uploaded', 400

        filenames = []

        for file in uploaded_files:
            # Check if file has a name and is allowed
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)

        # If no allowed files are uploaded
        if not filenames:
            return 'No allowed files uploaded', 400

        return 'Files uploaded successfully: ' + ', '.join(filenames)

    return render_template('uploadImages1.html')


@app.route('/buildModel')
def buildModel():
    return render_template('buildModel1.html')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess
import pickle
from calculateParameters import compute_parameters

from src.Objects.objects import parse_data_to_objects, all_planets_json

app = Flask(__name__)

# Define the path for the uploaded images. You may need to adjust the path depending on your project structure.
app.config['UPLOAD_FOLDER'] = '../images/uploadedImages'
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
def build_model():
    return render_template('buildModel1.html')


@app.route('/buildModel/clientSimulation')
def build_model_client_simulation():
    # Load and render the next page in the building model process
    return render_template('clientSimulation.html')


@app.route('/buildModel', methods=['POST'])
def simulate_model():
    data = request.json  # Get JSON data sent from JavaScript
    objects = data['objects']  # Extract objects list from JSON
    stars, planets = parse_data_to_objects(objects)
    print("Received objects:", planets)
    with open('planets_data.pkl', 'wb') as f:
        pickle.dump(planets, f)
    return jsonify({'status': 'success', 'message': 'Simulation started and objects received successfully!'})



@app.route('/simulationModel1')  # opens the simulation lobby page for the built-in simulation
def simulate_model1():
    return render_template('simulationModel.html')


@app.route('/start-simulation')  # This function activate the simulation for the built-in model
def start_simulation():
    try:
        # Runs the pygame script as a separate process
        subprocess.Popen(['python', 'planet_Simulation_Above.py'])
        return "Simulation started"
    except Exception as e:
        return str(e)

@app.route('/start-simulation2')  # This function activate the simulation for the built-in model
def start_simulation2():
    try:
        # Runs the pygame script as a separate process
        subprocess.Popen(['python', 'image_From_Earth.py'])
        return "Simulation started"
    except Exception as e:
        return str(e)


@app.route('/buildModel2')  # This function activate the simulation for the client model
def start_simulation_client():
    try:
        # Runs the pygame script as a separate process
        subprocess.Popen(['python', 'planet_Simulation_Orbit.py'])
        return "Simulation started"
    except Exception as e:
        return str(e)

@app.route('/buildModelPOV')  # This function activate the simulation for the client model
def start_simulation_pov():
    try:
        # Runs the pygame script as a separate process
        subprocess.Popen(['python', 'image_From_Earth_2.py'])
        return "Simulation started"
    except Exception as e:
        return str(e)


@app.route('/NasaPicture')
def nasa_page():
    return render_template('pictureOfTheDay.html')

@app.route('/NasaSearch')
def nasa_search():
    return render_template('nasaSearch.html')

@app.route('/Resources')
def open_resources():
    return render_template('Resources.html')

@app.route('/CompareParameters')
def open_compare():
    return render_template('compareParameters.html')

@app.route('/get_planets')
def get_planets():
    return all_planets_json

@app.route('/compute-results', methods=['POST'])
def handle_compute_results():
    try:
        result = compute_parameters()
        if 'error' in result:
            return jsonify(result), 400  # Custom error handling
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from red_digit_detector import detect_digits_and_sum

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files or 'student_name' not in request.form:
        return "Missing fields", 400

    file = request.files['image']
    student_name = request.form['student_name'].strip()

    if file.filename == '' or not student_name:
        return "Missing image or name", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"output_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        file.save(input_path)

        digits, total = detect_digits_and_sum(input_path, output_path)

        return render_template(
            'result.html',
            student_name=student_name,
            input_image=filename,
            output_image=output_filename,
            digits=digits,
            total=total
        )

    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)

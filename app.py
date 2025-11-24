from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
# ===========================================
# 🔍 STEP 1: Import Required Libraries
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# ===========================================
# 📁 STEP 2: Define Paths
# ===========================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
model_path = 'vgg16_final_model.h5'   # Path to saved model
dataset_path = 'Test'                # Folder used for class labels

# ===========================================
# 🧠 STEP 3: Load Model and Class Labels
# ===========================================
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Get class names (sorted alphabetically like training)
class_labels = sorted(os.listdir(dataset_path))
print(f"Loaded {len(class_labels)} classes:")
print(class_labels)
# Load your trained model
try:
    model = load_model('vgg16_final_model.h5')
    print("Model loaded successfully")
except:
    print("Warning: Could not load model.h5")
    model = None

# Class labels (adjust based on your model's training)
CLASS_LABELS = ['fresh_apple', 'rotten_apple', 'fresh_banana', 'rotten_banana', 
                'fresh_orange', 'rotten_orange', 'fresh_tomato', 'rotten_tomato']

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_fruit(image_path):
    if model is None:
        return "Model not available", 0.0
    
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return "Error processing image", 0.0
    
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = CLASS_LABELS[predicted_class_index]
    return predicted_class, confidence

def get_suggestion(prediction, confidence):
    fruit_type = prediction.split('_')[1]  # Extract fruit name
    status = prediction.split('_')[0]      # Extract freshness status
    
    suggestions = {
        'fresh': f"This {fruit_type} appears fresh and healthy! It's good for consumption.",
        'rotten': f"This {fruit_type} appears to be rotten. It's recommended to avoid consumption."
    }
    
    base_suggestion = suggestions.get(status, "Unable to determine fruit condition.")
    
    if confidence < 0.7:
        base_suggestion += " Note: Confidence in this prediction is low. Please verify visually."
    
    return base_suggestion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        flash('Please login to upload images.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = load_img(filepath, target_size=(224, 224))
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            # ===========================================
            # 🧮 STEP 5: Prediction
            # ===========================================
            pred = model.predict(x)
            pred_index = np.argmax(pred)
            pred_class = class_labels[pred_index]
            confidence = np.max(pred) * 100
            # ===========================================
            # 🍎 STEP 6: Extract Freshness and Fruit Type
            # ===========================================
            if 'fresh' in pred_class.lower():
                freshness = "fresh"
            elif 'rotten' in pred_class.lower():
                freshness = "rotten"
            else:
                freshness = "Unknown"
            # Remove freshness word to isolate fruit type
            fruit_name = pred_class.lower().replace('fresh', '').replace('rotten', '').strip()
            fruit_name = fruit_name.capitalize()
            
            # Make prediction
            #prediction, confidence = predict_fruit(filepath)
            prediction=freshness+"_"+fruit_name
            print(prediction)
            suggestion = get_suggestion(prediction, confidence)
            
            # Save prediction to database
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO predictions (user_id, filename, prediction, confidence) VALUES (?, ?, ?, ?)',
                         (session['user_id'], filename, prediction, float(confidence)))
            conn.commit()
            conn.close()
            
            return render_template('result.html', 
                                 prediction=prediction,
                                 confidence=confidence,
                                 suggestion=suggestion,
                                 filename=filename)
        else:
            flash('Invalid file type! Please upload an image.', 'error')
    
    return render_template('upload.html')

@app.route('/results')
def results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT filename, prediction, confidence, timestamp 
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    ''', (session['user_id'],))
    predictions = cursor.fetchall()
    conn.close()
    
    return render_template('results.html', predictions=predictions)

if __name__ == '__main__':
    init_db()
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

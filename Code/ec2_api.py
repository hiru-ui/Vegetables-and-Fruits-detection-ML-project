import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

# Load model
model = load_model('FV.h5')

labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

def prepare_image(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(f"Predicted class: {y_class}")
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = labels[y]
        print(f"Predicted label: {res}")
        return res.capitalize()
    except Exception as e:
        print(f"Error in prepare_image: {e}")
        raise e

def fetch_calories(prediction):
    try:
        app_id = 'af090f89'
        app_key = '7efd4f2fdc1e1fa089916e68c19feac8'
        url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
        headers = {
            'x-app-id': app_id,
            'x-app-key': app_key,
            'Content-Type': 'application/json'
        }
        data = {
            'query': prediction,
            'timezone': 'US/Eastern'
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        if 'foods' in result and len(result['foods']) > 0:
            calories = result['foods'][0]['nf_calories']
            return f"{calories} kcal per 100g"
        else:
            return "Calorie info not found"
    except Exception as e:
        print(f"Error fetching calories: {e}")
        return "Can't fetch calories"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create upload directory if it doesn't exist
os.makedirs("./upload_images", exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return jsonify(message="Fruits & Vegetables Detection API is running!", status="success")

@app.route('/predict', methods=['POST'])
def infer_image():
    try:
        if 'file' not in request.files:
            return jsonify(error="Please try again. The Image doesn't exist"), 400

        file = request.files.get('file')
        if file.filename == '':
            return jsonify(error="No file selected"), 400
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify(error="Invalid file type. Please upload an image file."), 400

        # Save the uploaded file
        img_bytes = file.read()
        img_path = "./upload_images/test.jpg"
        
        with open(img_path, "wb") as img:
            img.write(img_bytes)
        
        # Make prediction
        result = prepare_image(img_path)
        
        # Get nutritional info
        calories = fetch_calories(result)
        
        # Determine category
        fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
        vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip']
        
        category = "Vegetable" if result in vegetables else "Fruit"
        
        return jsonify({
            'prediction': result,
            'category': category,
            'calories': calories,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in infer_image: {e}")
        return jsonify(error=f"An error occurred: {str(e)}"), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(status="healthy", message="API is working correctly")

if __name__ == '__main__':
    print("Starting Flask API server...")
    print("API will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Home: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
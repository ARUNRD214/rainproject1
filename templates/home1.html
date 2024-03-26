from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = None  # Load your model
sc = None  # Load your scaler

@app.route('/prediction', methods=['POST'])
def prediction():
    # Extract sensor data from JSON payload
    sensor_data = request.json
    temperature = sensor_data['temperature']
    humidity = sensor_data['humidity']
    moisture = sensor_data['moisture']

    # Modify this part to match the required input format for your model
    model_input = np.array([temperature, temperature, 0, humidity, humidity, temperature, temperature, 0]).reshape(1, -1)
    final_input = sc.transform(model_input)

    # Make prediction
    output = model.predict(final_input)[0]

    # Return prediction result as JSON
    if output == 0:
        return jsonify({"prediction": "Tomorrow will be a SUNNY DAY"})
    else:
        return jsonify({"prediction": "Tomorrow will be a RAINY DAY"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

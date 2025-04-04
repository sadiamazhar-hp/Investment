from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from main import get_stock_prediction

app = Flask(__name__)
CORS(app)  # Allow React to connect

@app.route("/api/data", methods=["GET"])
def get_data():
    return jsonify({"message": "STOCK PREDICTIONS!"})

@app.route("/api/insert", methods=["POST"])
def insert_data():
    data = request.json  # Get data from React
    return jsonify({"received": data})

@app.route("/api/predict", methods=["POST"])  # New endpoint for stock prediction
def predict_stock():
    data = request.json
    quote = data.get("quote")  # Get stock symbol from request
    
    if not quote:
        return jsonify({"error": "Stock quote is required"}), 400
    
    result = get_stock_prediction(quote)  # Call your function
    print(result)
    return jsonify(result)  # Send JSON response

# Path to your static folder
STATIC_FOLDER = "./static"

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable auto-reload

from flask import Flask, request, jsonify, render_template
from models.ai_model import generate_response

app = Flask(__name__)

# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the AI response
@app.route('/ask', methods=['POST'])
def ask():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Check if 'query' is in the data
    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    user_query = data['query']
    
    # Generate a response using the AI model
    response = generate_response(user_query)
    
    # Return the response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

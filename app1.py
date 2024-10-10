from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

# HTML Route for the main page
@app.route("/")
def index():
    return render_template("index2.html")


# Route to handle the initial input and send to another Flask app
@app.route("/send-to-another-flask", methods=["POST"])
def send_to_another_flask():
    data = request.get_json()
    user_input = data.get("Query")
    
    if user_input:
        print(f"Received input: {user_input}")
        
        # Send the input to another Flask app (assuming you have a second Flask app running)
        response = requests.post("http://127.0.0.1:5001/process", json={"Query": user_input})
        
        if response.status_code == 200:
            json_data = response.json()
            print(f"Response from another Flask: {json_data}")  # Debugging print
            
            # Return the JSON received from another Flask to the frontend
            return jsonify(json_data)
        else:
            print(f"Error: Unable to reach the second Flask app. Status code: {response.status_code}")
            return jsonify({"error": "Error in sending input to another Flask app"}), 500
    else:
        return jsonify({"error": "Invalid input"}), 400


# Route to handle selections and send them back to another Flask app
@app.route("/submit-selections-to-another-flask", methods=["POST"])
def submit_selections_to_another_flask():
    data = request.get_json()
    selections = data.get("selection")
    print(selections)
    if selections:
        print(f"Received selections: {selections}")
        
        # Send the selections to another Flask app
        response = requests.post("http://127.0.0.1:5001/process", json={"selection": selections})
        
        if response.status_code == 200:
            json_data = response.json()
            print(f"Final result from another Flask: {json_data}")  # Debugging print
            
            # Return the final result to the frontend
            return jsonify(json_data)
        else:
            print(f"Error: Unable to reach the second Flask app. Status code: {response.status_code}")
            return jsonify({"error": "Error in submitting selections to another Flask app"}), 500
    else:
        return jsonify({"error": "No selections provided"}), 400


if __name__ == "__main__":
    app.run(debug=True,port=5000)



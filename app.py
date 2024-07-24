from flask import Flask, request, jsonify, render_template
import agent

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', {}).get('content', '')
    #name = data.get('message', {}).get('id', '')

    response = agent.handle_user_input(user_input)

    return jsonify({'message': {'content': response}})

if __name__ == '__main__':
    app.run(debug=True, port=8000)

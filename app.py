from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    grammar = data['input1']
    vocab = data['input2']

    # Example: Load and manipulate a CSV file
    df = pd.read_csv('BHS_with_Biblingo.csv')
    filtered_df = df.query('Grammar == {} and Vocab <= {}'.format(grammar, vocab))

    # Return the processed data
    return jsonify(filtered_df.to_dict(orient='records'))

if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))

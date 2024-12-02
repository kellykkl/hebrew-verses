from flask import Flask, request, jsonify
import pandas as pd
import re

app = Flask(__name__)

# Function to wrap text in <span> tags
def color_text(word, highlight, known):
    if highlight:
        return f'<span style="color:green;">{word}</span>'
    elif known:
        return f'<span style="color:blue;">{word}</span>'
    else:
        return word

def clean_hebrew_text(concatenated_text):
    # Remove 'nan' (as a string)
    cleaned_text = concatenated_text.replace("nan", "").strip()
    
    # List of Hebrew prefixes that should not have a space after them
    prefixes = ['כְ','מֵ','מִ','וָ','וְ','וַ','הֶ','הַ','הָ','לָ','לַ','לֵ','לְ','לִ','בַ','בָ','בְ','בַּ','בָּ','בְּ','בִ','בִּ','כַ','כָ','כְּ','כַּ','כָּ','וּ']
    
    # Loop through each prefix and remove space after it if present
    for prefix in prefixes:
        cleaned_text = re.sub(rf'(?<!\S)({prefix})\s+', r'\1', cleaned_text)
        cleaned_text = re.sub(rf'(<span style=.*?>{prefix}</span>)\s+', r'\1', cleaned_text)
        
    # Remove spaces after words that are followed by the maqaf (־)
    cleaned_text = re.sub(r'(\S)־\s+', r'\1־', cleaned_text)
    cleaned_text = re.sub(r'(\S)־</span>\s+', r'\1־</span>', cleaned_text)
 
    return cleaned_text


@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    grammar = data['input1']
    vocab = data['input2']

    grammar_li = grammar.split(',')
    vocab_li = vocab.split(',')

    # Remove elements in grammar_list from vocab_list
    vocab_li = [item for item in vocab_li if item not in grammar_li]

    grammar_list = [float(x) for x in grammar_li]
    vocab_list = [float(x) for x in vocab_li]

    # Example: Load and manipulate a CSV file
    df = pd.read_csv('BHS_w_biblingo.csv')
    filtered_df = df.query('Grammar.isin(@grammar_list) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')
    filtered_df_broader = df.query('(Grammar.isin(@grammar_list) or Grammar.isin(@vocab_list) or Grammar == 1.0) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')

    # Group by 'ESVLocation' and count rows matching the condition
    grouped = df.groupby('ESVLocation').size().reset_index(name='TotalCount')
    filtered_count = filtered_df_broader.groupby('ESVLocation').size().reset_index(name='FilteredCount')
    grouped_counts_broader = filtered_df_broader.groupby('ESVLocation')['Grammar'].nunique().reset_index(name='UniqueGrammarCountBroader')
    grouped_counts = filtered_df.groupby('ESVLocation')['Grammar'].nunique().reset_index(name='UniqueGrammarCount')
    result = pd.merge(grouped, filtered_count, on='ESVLocation', how='left')
    result = pd.merge(result, grouped_counts, on='ESVLocation', how='left')
    result = pd.merge(result, grouped_counts_broader, on='ESVLocation', how='left')


    result['FilteredCount'] = result['FilteredCount'].fillna(0).astype(int)
    result['UniqueGrammarCount'] = result['UniqueGrammarCount'].fillna(0).astype(int)
    result['UniqueGrammarCountBroader'] = result['UniqueGrammarCountBroader'].fillna(0).astype(int)
    result['proportion'] = result['FilteredCount']*1.0/result['TotalCount']

    result['score'] = 3.2/(len(grammar_list))*result['UniqueGrammarCount'] + \
                            (0.8/(len(vocab_list) + len(grammar_list)))*result['UniqueGrammarCountBroader'] + \
                            2.0*result['proportion'] - \
                            0.02*result['TotalCount']

    wanted = result.sort_values('score', ascending=False).head(10)


    df['BHSwordPointed'] = df['BHSwordPointed'].astype(str)
    merged = pd.merge(wanted, df, on='ESVLocation', how='left')

    merged.loc[merged.query('Grammar.isin(@grammar_list) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)').index, 'Highlight'] = True
    merged[['Highlight']] = merged[['Highlight']].fillna(value=False)

    merged.loc[merged.query('(Vocab.isin(@vocab_list) or Vocab == 1.0) and (Grammar.isin(@vocab_list) or Grammar == 1.0)').index, 'KnownBefore'] = True
    merged[['KnownBefore']] = merged[['KnownBefore']].fillna(value=False)

    merged['StyledWord'] = merged.apply(lambda row: color_text(row['BHSwordPointed'], row['Highlight'], row['KnownBefore']), axis=1)

     
    concatenated = (
        merged.sort_values(by=['ESVLocation', 'BHSwordSort'])  # Sort within each group
              .groupby('ESVLocation')['StyledWord']           # Group by ESVLocation
              .apply(lambda x: " ".join(x))                   # Concatenate strings
              .reset_index(name='Concatenated Verse')         # Create a new column
    )

    # Apply the cleaning function to the 'Concatenated Verse' column
    concatenated['Concatenated Verse'] = concatenated['Concatenated Verse'].apply(clean_hebrew_text)


    # Replace NaN with None (JSON-compliant null)
    concatenated = concatenated.where(pd.notnull(concatenated), None)

    # Return the processed data
    return jsonify(concatenated.to_dict(orient='records'))

if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))

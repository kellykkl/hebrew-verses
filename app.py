from flask import Flask, request, jsonify
import pandas as pd
import re

app = Flask(__name__)

# Function to wrap text in <span> tags
def color_text(word, highlight, known, proper):
    if highlight:
        return f'<span style="color:green;">{word}</span>'
    elif (known and not proper):
        return f'<span style="color:blue;">{word}</span>'
    elif proper:
        return f'<span style="color:gray;">{word}</span>'
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

def rolling_collect(series, window):
    result = []
    for i in range(len(series)):
        # Get the rolling window slice for a forward-looking window
        window_data = series[i: i + window]
        # Flatten the lists, remove NaN, and collect unique values
        unique_values = [item for sublist in window_data.dropna() for item in sublist]
        result.append(unique_values)
    return result

def rolling_sum_forward(series, window):
    result = []
    for i in range(len(series)):
        # Get the rolling window slice for a forward-looking window
        window_data = series[i: i + window]
        result.append(window_data.sum())
    return result

def rolling_count_forward(series, window):
    result = []
    for i in range(len(series)):
        # Get the rolling window slice for a forward-looking window
        window_data = series[i: i + window]
        result.append(window_data.count())
    return result

def grouped_rolling_collect(df, window):
    # Apply rolling_collect_unique to each group for a forward-looking window
    df['RollingUniqueGrammarBroaderList'] = rolling_collect(df['UniqueGrammarBroaderList'], window)
    df['RollingUniqueGrammarList'] = rolling_collect(df['UniqueGrammarList'], window)
    
    # Apply rolling sum for 'TotalCount' and 'FilteredCount'
    df['RollingTotalCount'] = rolling_sum_forward(df['TotalCount'], window)
    df['RollingFilteredCount'] = rolling_sum_forward(df['FilteredCount'], window)
    
    # Apply rolling count for 'ValidWindow'
    df['ValidWindow'] = rolling_count_forward(df['TotalCount'], window)
    df['ValidWindow'] = df['ValidWindow'] >= window  # Ensure at least 'window' valid entries

    return df

def extend_row_with_verses(row, window_size, df):
    """
    This function takes a row and generates new rows with the appropriate verse numbers.
    It appends (window_size - 1) additional rows with the correct verse numbers and data.
    
    Args:
    - row (pd.Series): A single row from the DataFrame.
    - window_size (int): The window size to append additional rows.
    - df (pd.DataFrame): The original DataFrame to append to.
    
    Returns:
    - pd.DataFrame: The extended DataFrame with the new rows added.
    """
    start_verse = row['verseNumber']
    new_rows = []

    # Create additional rows with the adjusted verse numbers
    for i in range(window_size):
        new_row = row.copy()
        new_row['verseNumber'] = start_verse + i  # Adjust verse number
        new_rows.append(new_row)
    
    # Convert new rows to DataFrame
    new_df = pd.DataFrame(new_rows)
    
    return new_df

# Helper function to check overlap
def check_overlap(selected, current_row, window_size=5):
    for row in selected:
        # Calculate the range of verses for the existing row
        existing_start = row['verseNumber']
        existing_end = existing_start + window_size - 1
        
        # Calculate the range of verses for the current row
        current_start = current_row['verseNumber']
        current_end = current_start + window_size - 1
        
        # Check for overlap
        if not (current_end < existing_start or current_start > existing_end):
            return True
    return False

# Function to select top non-overlapping rows
def get_top_non_overlapping(df, top_n=10, window_size=5):
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        if not check_overlap(selected, row, window_size):
            selected.append(row)
    return pd.DataFrame(selected)


@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    grammar = data['input1']
    vocab = data['input2']
    window_size = int(data['integerInput'])
    books = data['books']
    getGrammar = data['getGrammar']

    if books != "All":
        books_li = books.split(',')
        books_list = [int(x) for x in books_li]

    grammar_li = grammar.split(',')
    vocab_li = vocab.split(',')

    # Remove elements in grammar_list from vocab_list
    vocab_li = [item for item in vocab_li if item not in grammar_li]

    grammar_list = [float(x) for x in grammar_li]
    vocab_list = [float(x) for x in vocab_li]

    # Example: Load and manipulate a CSV file
    df = pd.read_csv('BHS_w_biblingo.csv')

    if books != "All":
        df = df.query('bookNumber.isin(@books_list)')



    if window_size == 1:

        grouped = df.groupby('ESVLocation').size().reset_index(name='TotalCount')

        if getGrammar:
            filtered_df = df.query('Grammar.isin(@grammar_list) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')
            filtered_df_broader = df.query('(Grammar.isin(@grammar_list) or Grammar.isin(@vocab_list) or Grammar == 1.0) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')

            # Group by 'ESVLocation' and count rows matching the condition
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
        else:

            filtered_df = df.query('Vocab.isin(@grammar_list) and (Grammar.isin(@vocab_list) or Grammar.isin(@grammar_list) or Grammar == 1.0)')
            grouped_counts = filtered_df.groupby('ESVLocation')['HebrewLexeme'].nunique().reset_index(name='UniqueVocabCount')
            result = pd.merge(grouped, grouped_counts, on='ESVLocation', how='left')
            result['score'] = 3.2/(len(grammar_list))*result['UniqueVocabCount'] 

            wanted = result.sort_values('score', ascending=False).head(10)

            df['BHSwordPointed'] = df['BHSwordPointed'].astype(str)
            merged = pd.merge(wanted, df, on='ESVLocation', how='left')

            merged.loc[merged.query('Vocab.isin(@grammar_list) and (Grammar.isin(@vocab_list) or Grammar.isin(@grammar_list) or Grammar == 1.0)').index, 'Highlight'] = True
            merged[['Highlight']] = merged[['Highlight']].fillna(value=False)

            merged.loc[merged.query('(Vocab.isin(@vocab_list) or Vocab == 1.0) and (Grammar.isin(@vocab_list) or Grammar == 1.0)').index, 'KnownBefore'] = True
            merged[['KnownBefore']] = merged[['KnownBefore']].fillna(value=False)



        merged.loc[merged.query('morphologyDetail.str.contains("proper noun")').index, 'ProperNoun'] = True
        merged[['ProperNoun']] = merged[['ProperNoun']].fillna(value=False)

        merged['StyledWord'] = merged.apply(lambda row: color_text(row['BHSwordPointed'], row['Highlight'], row['KnownBefore'], row['ProperNoun']), axis=1)

         
        concatenated = (
            merged.sort_values(by=['ESVLocation', 'BHSwordSort'])  # Sort within each group
                  .groupby(['ESVLocation','score'])['StyledWord']           # Group by ESVLocation
                  .apply(lambda x: " ".join(x))                   # Concatenate strings
                  .reset_index(name='Concatenated Verse')         # Create a new column
        )

        # Apply the cleaning function to the 'Concatenated Verse' column
        concatenated['Concatenated Verse'] = concatenated['Concatenated Verse'].apply(clean_hebrew_text)


        # Replace NaN with None (JSON-compliant null)
        concatenated = concatenated.where(pd.notnull(concatenated), None).sort_values('score',ascending=False)

    else:

        if getGrammar:

            grouped = df.groupby(['ESVLocation','verseNumber','bookNumber']).size().reset_index(name='TotalCount')

            filtered_df = df.query('Grammar.isin(@grammar_list) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')
            filtered_df_broader = df.query('(Grammar.isin(@grammar_list) or Grammar.isin(@vocab_list) or Grammar == 1.0) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)')

            # Group by 'ESVLocation' and count rows matching the condition
            filtered_count = filtered_df_broader.groupby(['ESVLocation','verseNumber','bookNumber']).size().reset_index(name='FilteredCount')

            grouped_counts_broader = (
                filtered_df_broader
                .groupby(['ESVLocation','verseNumber','bookNumber'])['Grammar']
                .agg(lambda x: list(x))
                .reset_index(name='UniqueGrammarBroaderList')
            )

            grouped_counts = (
                filtered_df
                .groupby(['ESVLocation','verseNumber','bookNumber'])['Grammar']
                .agg(lambda x: list(x))
                .reset_index(name='UniqueGrammarList')
            )

            result = pd.merge(grouped, filtered_count, on=['ESVLocation','verseNumber','bookNumber'], how='left')
            result = pd.merge(result, grouped_counts, on=['ESVLocation','verseNumber','bookNumber'], how='left')
            result = pd.merge(result, grouped_counts_broader, on=['ESVLocation','verseNumber','bookNumber'], how='left')

            # Sort by bookNumber and verseNumber, then apply grouped rolling
            result_sorted = result.sort_values(by=['bookNumber', 'verseNumber']).reset_index(drop=True)

            # Apply grouped rolling
            result_grouped = result_sorted.groupby('bookNumber').apply(grouped_rolling_collect, window=window_size).reset_index(drop=True)
            
            result_grouped['RollingUniqueGrammarBroaderCount'] = result_grouped['RollingUniqueGrammarBroaderList'].apply(lambda x: len(set(x)))
            result_grouped['RollingUniqueGrammarCount'] = result_grouped['RollingUniqueGrammarList'].apply(lambda x: len(set(x)))

            result_grouped['RollingFilteredCount'] = result_grouped['RollingFilteredCount'].fillna(0).astype(int)
            result_grouped['RollingUniqueGrammarCount'] = result_grouped['RollingUniqueGrammarCount'].fillna(0).astype(int)
            result_grouped['RollingUniqueGrammarBroaderCount'] = result_grouped['RollingUniqueGrammarBroaderCount'].fillna(0).astype(int)
            result_grouped['RollingProportion'] = result_grouped['RollingFilteredCount']*1.0/result_grouped['RollingTotalCount']

            result_grouped['score'] = 3.2/(len(grammar_list))*result_grouped['RollingUniqueGrammarCount'] + \
                                    (0.8/(len(vocab_list) + len(grammar_list)))*result_grouped['RollingUniqueGrammarBroaderCount'] + \
                                    1.5*result_grouped['RollingProportion']

            # Sort and filter the result_grouped DataFrame
            result_grouped = result_grouped.query('ValidWindow == True').sort_values('score', ascending=False)

            # Get the top 10 non-overlapping rows
            wanted = get_top_non_overlapping(result_grouped, top_n=10, window_size=window_size)

            extended_rows = []

            # Apply the function to each row in the 'wanted' DataFrame
            for _, row in wanted.iterrows():
                extended_rows.append(extend_row_with_verses(row, window_size, wanted))

            # Concatenate all extended rows into a single DataFrame
            extended_df = pd.concat(extended_rows, ignore_index=True)
            extended_df = extended_df.rename(columns={"ESVLocation": "ESVLocationStart"})

            wanted_df = extended_df[['ESVLocationStart','verseNumber','score']]

            df['BHSwordPointed'] = df['BHSwordPointed'].astype(str)
            merged = pd.merge(wanted_df, df, on='verseNumber', how='left')

            merged.loc[merged.query('Grammar.isin(@grammar_list) and (Vocab.isin(@vocab_list) or Vocab.isin(@grammar_list) or Vocab == 1.0)').index, 'Highlight'] = True
            merged[['Highlight']] = merged[['Highlight']].fillna(value=False)

            merged.loc[merged.query('(Vocab.isin(@vocab_list) or Vocab == 1.0) and (Grammar.isin(@vocab_list) or Grammar == 1.0)').index, 'KnownBefore'] = True
            merged[['KnownBefore']] = merged[['KnownBefore']].fillna(value=False)

        else:

            grouped = df.groupby(['ESVLocation','verseNumber','bookNumber']).size().reset_index(name='TotalCount')
            filtered_df = df.query('Vocab.isin(@grammar_list) and (Grammar.isin(@vocab_list) or Grammar.isin(@grammar_list) or Grammar == 1.0)')

            grouped_counts = (
                filtered_df
                .groupby(['ESVLocation','verseNumber','bookNumber'])['HebrewLexeme']
                .agg(lambda x: list(x))
                .reset_index(name='VocabList')
            )

            result = pd.merge(grouped, grouped_counts, on=['ESVLocation','verseNumber','bookNumber'], how='left')
            result_sorted = result.sort_values(by=['bookNumber', 'verseNumber']).reset_index(drop=True)
            result_sorted['RollingVocabList'] = rolling_collect(result_sorted['VocabList'], window_size)
            result_sorted['ValidWindow'] = rolling_count_forward(result_sorted['TotalCount'], window_size)
            result_sorted['ValidWindow'] = result_sorted['ValidWindow'] >= window_size  # Ensure at least 'window' valid entries

            result_sorted['RollingVocabCount'] = result_sorted['RollingVocabList'].apply(lambda x: len(set(x)))

            result_sorted['score'] = 3.2/(len(grammar_list))*result_sorted['RollingVocabCount']

            result_sorted = result_sorted.query('ValidWindow == True').sort_values('score', ascending=False)

            wanted = get_top_non_overlapping(result_sorted, top_n=10, window_size=window_size)

            extended_rows = []

            # Apply the function to each row in the 'wanted' DataFrame
            for _, row in wanted.iterrows():
                extended_rows.append(extend_row_with_verses(row, window_size, wanted))

            # Concatenate all extended rows into a single DataFrame
            extended_df = pd.concat(extended_rows, ignore_index=True)
            extended_df = extended_df.rename(columns={"ESVLocation": "ESVLocationStart"})

            wanted_df = extended_df[['ESVLocationStart','verseNumber','score']]

            df['BHSwordPointed'] = df['BHSwordPointed'].astype(str)
            merged = pd.merge(wanted_df, df, on='verseNumber', how='left')

            merged.loc[merged.query('Vocab.isin(@grammar_list) and (Grammar.isin(@vocab_list) or Grammar.isin(@grammar_list) or Grammar == 1.0)').index, 'Highlight'] = True
            merged[['Highlight']] = merged[['Highlight']].fillna(value=False)

            merged.loc[merged.query('(Vocab.isin(@vocab_list) or Vocab == 1.0) and (Grammar.isin(@vocab_list) or Grammar == 1.0)').index, 'KnownBefore'] = True
            merged[['KnownBefore']] = merged[['KnownBefore']].fillna(value=False)


        merged.loc[merged.query('morphologyDetail.str.contains("proper noun")').index, 'ProperNoun'] = True
        merged[['ProperNoun']] = merged[['ProperNoun']].fillna(value=False)

        merged['StyledWord'] = merged.apply(lambda row: color_text(row['BHSwordPointed'], row['Highlight'], row['KnownBefore'], row['ProperNoun']), axis=1)

         
        concatenated = (
            merged.sort_values(by=['ESVLocationStart', 'BHSwordSort'])  # Sort within each group
                  .groupby(['ESVLocationStart','score'])['StyledWord']           # Group by ESVLocation
                  .apply(lambda x: " ".join(x))                   # Concatenate strings
                  .reset_index(name='Concatenated Verse')         # Create a new column
        )

        # Apply the cleaning function to the 'Concatenated Verse' column
        concatenated['Concatenated Verse'] = concatenated['Concatenated Verse'].apply(clean_hebrew_text)


        # Replace NaN with None (JSON-compliant null)
        concatenated = concatenated.where(pd.notnull(concatenated), None).sort_values('score', ascending=False)

    # Return the processed data
    return jsonify(concatenated.to_dict(orient='records'))

if __name__ == '__main__':
    from os import environ
    app.run(host='0.0.0.0', port=int(environ.get('PORT', 5000)))

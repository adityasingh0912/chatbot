from flask import Flask, request, jsonify, abort
import json
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)  # This line was missing in the traceback section

# --- (Constants and Helper Functions - Remain Mostly the Same) ---
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'

def load_diamond_data(filename="diamonds.json"):
    """Loads diamond data, handles errors, and converts to lowercase."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                print(f"Error: Invalid data format in {filename}")
                return None
            # Convert string values to lowercase
            for diamond in data:
                for key, value in diamond.items():
                    if isinstance(value, str):
                        diamond[key] = value.lower()
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data: {e}")
        return None

def search_diamonds(data, query_params):
    """Filters diamonds based on query, handles various data types."""
    if not data:
        return []

    results = []
    for diamond in data:
        match = True
        for key, value in query_params.items():
            if key not in diamond:
                match = False
                break
            if isinstance(diamond[key], str):
                if str(value).lower() != diamond[key].lower():
                    match = False
                    break
            elif isinstance(diamond[key], (int, float)):
                try:
                    if float(value) != float(diamond[key]):
                        match = False
                        break
                except ValueError:
                    match = False
                    break
            else:
                if value != diamond[key]:
                    match = False
                    break
        if match:
            results.append(diamond)
    return results

def extract_constraints_from_query(user_query):
    """Extracts constraints (Carat, Color, Clarity, etc.) from the user query."""
    constraints = {}
    carat_match = re.search(r'(\d+(\.\d+)?)\s*-?\s*carat', user_query, re.IGNORECASE)
    if carat_match:
        constraints["Carat"] = float(carat_match.group(1))
    budget_match = re.search(r'under(?:\s*price)?\s*(\d+)', user_query, re.IGNORECASE)
    if budget_match:
        constraints["Budget"] = float(budget_match.group(1))
    color_match = re.search(r'\b([a-j])\b', user_query, re.IGNORECASE)
    if color_match:
        constraints["Color"] = color_match.group(1).lower()
    clarity_match = re.search(r'\b(if|vvs1|vvs2|vs1|vs2|si1|si2)\b', user_query, re.IGNORECASE)
    if clarity_match:
        constraints["Clarity"] = clarity_match.group(1).lower()
    cut_match = re.search(r'\b(excellent|ideal|very good|good)\b', user_query, re.IGNORECASE)
    if cut_match:
        constraints["Cut"] = cut_match.group(1).lower()
    symmetry_match = re.search(r'\b(excellent|very good|good)\b', user_query, re.IGNORECASE)
    if symmetry_match:
        constraints["Symmetry"] = symmetry_match.group(1).lower()
    polish_match = re.search(r'\b(excellent|ideal|very good|good)\b', user_query, re.IGNORECASE)
    if polish_match:
        constraints["Polish"] = polish_match.group(1).lower()
    style_match = re.search(r'\b(labgrown|natural)\b', user_query, re.IGNORECASE)
    if style_match:
        constraints["Style"] = style_match.group(1).lower()
    shape_match = re.search(r'\b(round|princess|emerald|asscher|cushion|marquise|radiant|oval|pear|heart|square radiant)\b', user_query, re.IGNORECASE)
    if shape_match:
        constraints["Shape"] = shape_match.group(1).lower()
    return constraints

# --- Hybrid Search (FAISS + Filtering + Ranking) - Modified for JSON output ---
def hybrid_search(user_query, df, faiss_index, model, top_k=200):
    """Hybrid search, returns DataFrame, adds 'recommendation' flag."""
    constraints = extract_constraints_from_query(user_query)

    if "Style" in constraints:
        df = df[df['Style'] == constraints["Style"]]
        if df.empty:
            return pd.DataFrame()

    if "Budget" in constraints:
        user_budget = constraints["Budget"]
        df = df[df["Price"] <= user_budget]
        if df.empty:
            return pd.DataFrame()

    if "Carat" in constraints:
        tolerance = 0.01 if constraints.get("Style", "").lower() == "labgrown" else 0.05
        df_carat = df[
            (df['Carat'] >= constraints["Carat"] - tolerance) &
            (df['Carat'] <= constraints["Carat"] + tolerance)
        ]
        if df_carat.empty:
            relaxed_tolerance = tolerance * 2
            df_carat = df[
                (df['Carat'] >= constraints["Carat"] - relaxed_tolerance) &
                (df['Carat'] <= constraints["Carat"] + relaxed_tolerance)
            ]
        if not df_carat.empty:
            subset_indices = df_carat.index.tolist()
            all_embeddings = np.load(EMBEDDING_FILE_PATH)
            subset_embeddings = all_embeddings[subset_indices]
            temp_index = faiss.IndexFlatL2(all_embeddings.shape[1])
            temp_index.add(subset_embeddings)
            new_top_k = min(top_k, len(df_carat))
            query_embedding = model.encode(user_query, convert_to_numpy=True)
            D, I = temp_index.search(np.array([query_embedding]), new_top_k)
            valid_indices = [i for i in I[0] if 0 <= i < len(df_carat)]
            valid_D = D[0][:len(valid_indices)]
            results_df = df_carat.iloc[valid_indices].copy()
            results_df['distance'] = valid_D
        else:
            query_embedding = model.encode(user_query, convert_to_numpy=True)
            new_top_k = min(top_k, df.shape[0])
            D, I = faiss_index.search(np.array([query_embedding]), new_top_k)
            valid_indices = [i for i in I[0] if 0 <= i < df.shape[0]]
            valid_D = D[0][:len(valid_indices)]
            results_df = df.iloc[valid_indices].copy()
            results_df['distance'] = valid_D
    else:
        query_embedding = model.encode(user_query, convert_to_numpy=True)
        new_top_k = min(top_k, df.shape[0])
        D, I = faiss_index.search(np.array([query_embedding]), new_top_k)
        valid_indices = [i for i in I[0] if 0 <= i < df.shape[0]]
        valid_D = D[0][:len(valid_indices)]
        results_df = df.iloc[valid_indices].copy()
        results_df['distance'] = valid_D

    def compute_score(row):
        score = row['distance']
        if "Carat" in constraints:
            score += 1000 * abs(row["Carat"] - constraints["Carat"])
        if "Budget" in constraints:
            user_budget = constraints["Budget"]
            score += 0.05 * abs(row["Price"] - user_budget)
        else:
            try:
                price = float(row["Price"])
            except ValueError:
                price = 0
            score += 0.1 * price

        for attr, penalty in [("Clarity", 50), ("Color", 50)]:
            if attr in constraints and row[attr] != constraints[attr]:
                score += penalty
        for attr, penalty in [("Cut", 20), ("Symmetry", 20), ("Polish", 20)]:
            if attr in constraints and row[attr] != constraints[attr]:
                score += penalty
        return score

    results_df['score'] = results_df.apply(compute_score, axis=1)
    results_df = results_df.sort_values(by='score', ascending=True)

    # Add 'recommendation' flag (simplified logic for demonstration)
    #  In a real application, you'd likely have a more sophisticated
    #  recommendation system.
    results_df['recommendation'] = [True] * 3 + [False] * (len(results_df) - 3)  # Recommend top 3

    return results_df.head(5).reset_index(drop=True) # Return top 5



# --- Groq Integration - Modified for JSON preparation ---
def generate_groq_response(user_query, relevant_data, client):
    """
    Use FAISS retrieval to enhance the prompt before sending it to Groq.
    Sends a prompt to Groq's LLM to generate a shop assistant–style response using the provided diamond details.
    """
    prompt = f"""You are a friendly and knowledgeable shop assistant at a diamond store.
    Your goal is to help the customer find diamonds that best match their query.
    Speak in a warm, engaging, and professional tone, and use only the provided diamond details.
    Include all of the following details for each diamond option: Style, Carat, Price, Clarity, Color, Cut, Shape, Lab, Polish, and Symmetry.

    User Query: {user_query}

    Here are some diamond details that might be relevant:
    {relevant_data}

    Please provide a detailed, numbered recommendation (1 to 5) for the customer.
    In particular, if the diamond prices are significantly below the customer's specified budget, prioritize recommending the higher-priced options within the budget as these typically represent diamonds with enhanced quality and value.
    For each diamond option, include a full explanation of its attributes and why it might be a good choice.

    Please assist the customer by providing the diamond details (carat, clarity, color, cut, shape, price, and style) along with some helpful suggestions.
    Tell them what diamond would be a better choice and why.
    Now, generate a helpful and informative response while maintaining a professional, formal tone.
    Please only use the above information to generate your response. If no relevant diamonds are found, politely inform the user.

    Example Conversations:
    Q: I need a 0.5-carat diamond with a G color.
    A: We have multiple 0.5-carat diamonds in G color with different clarity grades. For example, one with VS1 clarity is priced at $2,500.

    Q: What's the best cut for a solitaire ring?
    A: The best cut for a solitaire ring is an Ideal or Excellent cut, as it maximizes brilliance.

    Now, based on our diamond inventory, here are the most relevant options:
    {relevant_data}

    Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "system", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=750
    )

    response_text = chat_completion.choices[0].message.content
    diamonds = extract_diamond_details(response_text)

    return diamonds

def extract_diamond_details(response_text):
    # Use regular expressions to extract diamond details from the response text
    diamond_pattern = re.compile(
        r"(?P<Style>\b\w+\b) diamond with "
        r"(?P<Carat>\d+\.\d+) carat, "
        r"(?P<Clarity>\b\w+\b) clarity, "
        r"(?P<Color>\b[A-Z]+\b) color, "
        r"(?P<Cut>\b\w+\b) cut, "
        r"(?P<Shape>\b\w+\b) shape, "
        r"priced at \$(?P<Price>\d+(\.\d+)?), "
        r"(?P<Lab>\b\w+\b) lab, "
        r"(?P<Polish>\b\w+\b) polish, "
        r"(?P<Symmetry>\b\w+\b) symmetry"
    )

    diamonds = []
    for match in diamond_pattern.finditer(response_text):
        diamond = {
            "Style": match.group("Style"),
            "Carat": match.group("Carat"),
            "Clarity": match.group("Clarity"),
            "Color": match.group("Color"),
            "Cut": match.group("Cut"),
            "Shape": match.group("Shape"),
            "Price": match.group("Price"),
            "Lab": match.group("Lab"),
            "Polish": match.group("Polish"),
            "Symmetry": match.group("Symmetry")
        }
        diamonds.append(diamond)

    return diamonds

# --- Main Chatbot Logic - Modified for JSON Output ---
def diamond_chatbot(user_query, df, faiss_index, model, client):
    # 1. Quick check for "hi" or "hello"
    if user_query.strip().lower() in ["hi", "hello"]:
        print("Hello! I'm your diamond assistant. How can I help you find the perfect diamond today?")
        return

    # 2. Extract constraints from the user query
    constraints = extract_constraints_from_query(user_query)

    # 3. If constraints are empty, handle that gracefully
    if not constraints:
        print("Hello! I'm your diamond assistant. Please let me know your preferred carat, clarity, color, "
              "cut, or budget so I can help you find the perfect diamond.")
        return

    # 4. Otherwise, proceed with your existing logic
    results_df = hybrid_search(user_query, df, faiss_index, model, top_k=200)
    if results_df.empty:
        print("No matching diamonds found. Please try a different query.")
        return

    top_5 = results_df.head(5)
    relevant_data = "\n".join(top_5['combined_text'].tolist())
    diamonds = generate_groq_response(user_query, relevant_data, client)

    # Print structured data as JSON
    import json
    print(json.dumps(diamonds, indent=2))

def load_data_and_index(embedding_file, faiss_index_file, dataframe_file, model_path):
    """Loads pre-existing data, embeddings, index, and model."""
    df = pd.read_csv(dataframe_file)
    df["Carat"] = pd.to_numeric(df["Carat"], errors="coerce")
    # Convert relevant columns to lowercase
    for col in ['Style', 'Clarity', 'Color', 'Cut', 'Shape', 'Lab', 'Polish', 'Symmetry']:
        if col in df.columns:  # Check if column exists
            df[col] = df[col].astype(str).str.lower()
    embeddings = np.load(embedding_file)
    index = faiss.read_index(faiss_index_file)
    model = SentenceTransformer(model_path)
    print("Loaded data, embeddings, FAISS index, and model from disk.")
    return df, embeddings, index, model

# --- Flask API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()

        if not user_query:
            return jsonify({
                'response': "I'm your diamond assistant. How can I help you find the perfect diamond today?"
            })

        # Capture printed output from diamond_chatbot
        import json
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Call the chatbot logic
        diamond_chatbot(user_query, df, faiss_index, model, client)

        # Restore stdout and get response
        sys.stdout = old_stdout
        response = mystdout.getvalue().strip()

        # If no response was generated, provide a default
        if not response:
            response = json.dumps([{
                "Style": "Unknown",
                "Carat": "Unknown",
                "Clarity": "Unknown",
                "Color": "Unknown",
                "Cut": "Unknown",
                "Shape": "Unknown",
                "Price": "Unknown",
                "Lab": "Unknown",
                "Polish": "Unknown",
                "Symmetry": "Unknown"
            }])

        return jsonify({'response': json.loads(response)})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': "I apologize, but I encountered an error. Please try your request again."
        }), 500

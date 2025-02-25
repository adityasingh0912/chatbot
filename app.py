from flask import Flask, render_template, request, jsonify
import io
import sys
from chatbot1 import diamond_chatbot, load_data_and_index, extract_constraints_from_query
from groq import Groq
from dotenv import load_dotenv
import os
import re
import json

def convert_markdown_to_html(text):
    """
    Convert markdown bold (text) into HTML <strong>text</strong>.
    """
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

app = Flask(__name__)

# File paths
EMBEDDING_FILE_PATH = 'diamond_embeddings.npy'
FAISS_INDEX_FILE = 'diamond_faiss_index.faiss'
DATAFRAME_FILE = 'diamond_dataframe.csv'
MODEL_PATH = 'sentence_transformer_model'

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq()

# Load data, embeddings, FAISS index, and model at startup
try:
    df, embeddings, faiss_index, model = load_data_and_index(
        EMBEDDING_FILE_PATH,
        FAISS_INDEX_FILE,
        DATAFRAME_FILE,
        MODEL_PATH
    )
    print("Successfully loaded diamond data and models")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

def generate_expert_analysis(user_query, diamond_data):
    """
    Generate expert analysis using Groq.
    """
    # Convert diamond data to string format
    diamond_str = "\n".join([str(diamond) for diamond in diamond_data])
    
    prompt = f"""
    You are a diamond expert with years of experience in the industry.
    Based on the user query and the diamonds found, provide a brief 2-3 line expert recommendation.
    Focus on what makes these particular diamonds a good match for the customer's needs.
    Be concise but insightful.
    
    User Query: {user_query}
    
    Diamonds Found:
    {diamond_str}
    
    Your expert analysis (2-3 lines only):
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=150
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating expert analysis: {e}")
        return "These diamonds match your criteria and offer excellent value. Consider factors like cut quality and color which significantly impact a diamond's brilliance."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('message', '').strip()
        
        if not user_query:
            return jsonify({
                'response': "I'm your diamond assistant. How can I help you find the perfect diamond today?"
            })
        
        # Extract constraints from user query
        constraints = extract_constraints_from_query(user_query)
        
        # Check if style is missing and it's a diamond search query (not just a greeting)
        if "Style" not in constraints and not user_query.lower() in ["hi", "hello"] and len(constraints) > 0:
            # If style is missing, ask the user to specify
            return jsonify({
                'response': "Would you prefer a lab-grown or natural diamond? Lab-grown diamonds are eco-friendly and more affordable, while natural diamonds are mined from the earth and traditionally valued.",
                'needs_style': True
            })
        
        # Capture printed output from diamond_chatbot
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        
        # Call the chatbot logic
        diamond_chatbot(user_query, df, faiss_index, model, client)
        
        # Restore stdout and get response
        sys.stdout = old_stdout
        response = mystdout.getvalue().strip()
        
        # If no response was generated, provide a default
        if not response:
            response = "I'm having trouble understanding your request. Could you please provide more details about the diamond you're looking for?"
        
        # Extract the diamond data JSON if it exists
        diamond_data = None
        diamond_data_match = re.search(r'<diamond-data>([\s\S]*?)</diamond-data>', response)
        if diamond_data_match:
            try:
                diamond_data = json.loads(diamond_data_match.group(1))
            except:
                pass
        
        # Generate expert analysis if diamond data exists
        expert_analysis = ""
        if diamond_data and isinstance(diamond_data, list) and len(diamond_data) > 0:
            expert_analysis = generate_expert_analysis(user_query, diamond_data)
            
            # Insert the expert analysis before the closing </diamond-data> tag
            response = response.replace('</diamond-data>', f'</diamond-data>\n\n<expert-analysis>{expert_analysis}</expert-analysis>')
        
        # Convert markdown bold (text) to HTML <strong>text</strong>
        response_html = convert_markdown_to_html(response)
        
        return jsonify({
            'response': response_html,
            'expert_analysis': expert_analysis
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': "I apologize, but I encountered an error. Please try your request again."
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5500)
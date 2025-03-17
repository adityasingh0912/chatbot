# 💎 Gemma: AI Chatbot for Diamond Marketplace  

Welcome to **Gemma**, an AI-powered chatbot designed to assist users in finding the perfect diamond based on their preferences. This chatbot integrates advanced **natural language processing (NLP)** and **machine learning** techniques to provide personalized recommendations.  

---

## 🚀 Features  

✔ **Interactive Chatbot** – Ask questions about diamonds, and get tailored recommendations.  
✔ **AI-Powered Search** – Uses a **FAISS-based hybrid search** for accurate results.  
✔ **Expert Analysis** – Provides professional insights on selected diamonds.  
✔ **User-Friendly Interface** – A clean and intuitive web-based UI for seamless interaction.   

---

## 📂 Project Structure  

```
Gemma/
│── app.py                    # Flask application entry point
│── chatbot.py                 # Chatbot logic and recommendation engine
│── templates/
│   ├── index.html             # Web-based chat interface
│── diamonds.csv               # Raw diamond dataset
│── diamond_dataframe.csv       # Processed diamond data
│── requirements.txt            # Required dependencies
│── README.md                   # Project documentation
│── .gitignore                  # Git ignored files
```

---

## 🛠 Installation & Setup  

### 1️⃣ Clone the Repository  
```sh
git clone https://github.com/yourusername/Gemma.git
cd Gemma
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)  
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Application
Create a `.env` file in the root directory and add the following:  
```sh
GROQ_API_KEY=your_groq_api_key_here
```
Replace `your_groq_api_key_here` with your actual API key.
### 5️⃣ Run the Application  
```sh
python app.py
```

The chatbot will be available at:  
🔗 `http://127.0.0.1:5500/`  

---

## 📝 Usage Instructions  

1. **Open the Web Interface** – Visit the running URL in your browser.  
2. **Ask Questions** – Type queries like "Find me a 1-carat round diamond" in the chat.  
3. **Get Recommendations** – The chatbot will return the best-matching diamonds along with expert insights.

---

## 🤖 How It Works  

- **Data Processing**: Loads and processes diamond data from `diamonds.csv`.  
- **Embedding Generation**: Converts text descriptions into vector embeddings using `SentenceTransformer`.  
- **FAISS-Based Search**: Uses FAISS for efficient similarity search.  
- **AI-Powered Responses**: Integrates with Groq AI to generate natural language responses.

---

## ⚡ Key Technologies Used  

- **Flask**: Web framework for Python  
- **FAISS**: Vector search library for efficient similarity searches  
- **SentenceTransformer**: NLP model for text embeddings  
- **Groq API**: AI-based chatbot response generation


  
its good work...

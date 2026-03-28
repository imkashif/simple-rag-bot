import os
import sqlite3
import json
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
load_dotenv()


# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your-telegram-bot-token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

#print (TELEGRAM_TOKEN, OPENAI_API_KEY, OPENAI_MODEL)

# Initialize OpenAI client
#openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
print("OpenAI client initialized.")

# Initialize SentenceTransformer model
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer model loaded.")

def setup_db():
    conn = sqlite3.connect("knowledge.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chunks 
                 (id INTEGER PRIMARY KEY, text TEXT, embedding TEXT)''')
    conn.commit()
    return conn

def populate_db(conn):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM chunks")
    if c.fetchone()[0] > 0:
        return # DB already populated

    folder_path = "knowledge_base"
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created a new folder named '{folder_path}'. Please put your .txt or .pdf files in there and run again.")
        return

    all_chunks = []

    # Loop through every file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # --- Handle Standard Text Files ---
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                chunks = [chunk.strip() for chunk in content.split("\n\n") if len(chunk.strip()) > 50]
                all_chunks.extend(chunks)
                print(f"Loaded: {filename}")
                
        # --- Handle PDF Files ---
        elif filename.endswith(".pdf"):
            with open(file_path, "rb") as f: # 'rb' stands for read-binary
                pdf_reader = PyPDF2.PdfReader(f)
                pdf_text = ""
                
                # Loop through every page in the PDF
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pdf_text += extracted + "\n\n"
                        
                # Split the extracted PDF text into paragraphs
                chunks = [chunk.strip() for chunk in pdf_text.split("\n\n") if len(chunk.strip()) > 50]
                all_chunks.extend(chunks)
                print(f"Loaded: {filename}")

    if not all_chunks:
        print(f"No valid text or PDF files found in the '{folder_path}' folder.")
        return

    # Turn all the gathered chunks into vectors and save them
    print(f"\nFound {len(all_chunks)} paragraphs across your files. Embedding now...")
    for chunk in all_chunks:
        emb = embedder.encode(chunk).tolist()
        c.execute("INSERT INTO chunks (text, embedding) VALUES (?, ?)", 
                  (chunk, json.dumps(emb)))
    
    conn.commit()
    print("Database populated successfully!\n")

# Actually run the setup functions
db_conn = setup_db()
populate_db(db_conn)


def retrieve_top_k(conn, query, k=2):
    # 1. Turn the user's question into numbers
    query_emb = embedder.encode(query)
    
    # 2. Grab all the saved documents from the database
    c = conn.cursor()
    c.execute("SELECT text, embedding FROM chunks")
    rows = c.fetchall()
    
    scored_chunks = []
    for row in rows:
        text, emb_str = row
        doc_emb = np.array(json.loads(emb_str))
        
        # 3. The Math: Compare the question's numbers to the document's numbers
        similarity = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored_chunks.append((similarity, text))
        
    # 4. Sort the results so the highest scores are at the top, and return the top 'k'
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:k]

#Initialize database connection for use in handlers
db_conn = setup_db()
populate_db(db_conn)

print("===================================================================")
print("Bot is ready to receive commands.")
print("===================================================================")

while True:
    user_input = input("Enter a question (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    top_chunks = retrieve_top_k(db_conn, user_input, k=2)
    print("\nTop relevant chunks:")

    for score, chunk in top_chunks:
        print(f"Score: {score:.4f} - Text: {chunk[:100]}...")  # Print first 100 chars of chunk
    print("\n")

    if not top_chunks:
        print("No relevant information found in the database.")
        continue

    # Prepare the prompt for OpenAI
    prompt = f"Answer the question based on the following information:\n\n"
    for score, chunk in top_chunks:
        prompt += f"{chunk}\n\n"
    prompt += f"Question: {user_input}\nAnswer:"
    # Call OpenAI to get the answer
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    print("OpenAI Response:")
    print(response.choices[0].message.content.strip())

    

    

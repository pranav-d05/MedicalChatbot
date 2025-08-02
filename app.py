from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

embeddings = download_embeddings()
index_name = 'medical-chatbot'
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)

# MEMORY: Store previous messages
memory = ConversationBufferMemory(return_messages=True)

# Prompt with memory variables
prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    ('placeholder', '{chat_history}'),
    ('human', '{input}')
])

# Chain that handles prompt + chat history + retrieval
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    print("User:", msg)

    # Get chat history as string
    chat_history = memory.load_memory_variables({})["history"]

    # Run chain with history
    response = rag_chain.invoke({
        'input': msg,
        'chat_history': chat_history
    })

    # Save turn to memory
    memory.save_context({'input': msg}, {'output': response['answer']})

    print("Bot:", response['answer'])
    return str(response['answer'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

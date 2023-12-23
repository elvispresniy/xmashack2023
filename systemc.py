from config import PINECONE_API_KEY_SERVICES, OPENAI_API_KEY, PINECONE_API_KEY_BOOK
import messagesc

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

import pinecone

# LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Embeddings initialization
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Vectorstore initialization for services
pinecone.init(      
	api_key=PINECONE_API_KEY_SERVICES,      
	environment='gcp-starter'
)      
index = pinecone.Index('uslugi')

vectorstore_services = Pinecone.from_existing_index("uslugi", embedding=embeddings)

# Vectorstore initialization for book
pinecone.init(      
  api_key=PINECONE_API_KEY_BOOK,      
  environment='asia-southeast1-gcp-free'      
)      
index = pinecone.Index('hackathon')

vectorstore_book= Pinecone.from_existing_index("hackathon", embedding=embeddings)

def new_pipeline(query, chat_history):

    # Similarity search for advices
    docs_raw = vectorstore_book.similarity_search(query, k=3)
    docs = '\n'.join([doc.page_content for doc in docs_raw])

    # Similarity search for services
    docs_raw_services = vectorstore_services.similarity_search(query, k=2)
    docs_services = '\n'.join([doc.page_content for doc in docs_raw_services])

    # Structure a conversation query
    query_args = {
        'documents': docs,
        'services': docs_services,
        'history': chat_history,
        'input': query
    }
    query_prompt = messagesc.templates['conversation_prompt'].format(**query_args)
    print(f'query_prompt:\n{query_prompt}\n\n')

    # Inference the model
    result = llm.predict(query_prompt)

    chat_history_new = f'''
    {chat_history[-600:]}
    Клиент: {query}
    Ассистент: {result[:40]}...'''
    return result, chat_history_new


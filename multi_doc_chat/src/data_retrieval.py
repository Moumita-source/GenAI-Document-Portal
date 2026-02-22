import os
import sys
from multi_doc_chat.configuration.mongodb_connection import MongoDBClient
from multi_doc_chat.constants import DATABASE_NAME, COLLECTION_NAME, SCHEMA_FILE_PATH, INDEX_NAME, CHUNK_TEXT, EMBEDDING
from multi_doc_chat.exception import MyException
from multi_doc_chat.logger import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
from multi_doc_chat.utils.main_utils import read_yaml_file
from multi_doc_chat.prompts.prompt_library import RAG_ANSWER_PROMPT

from dotenv import load_dotenv 
load_dotenv()

class DataRetrieval:
    def __init__(self):
        """
        Initializes the MongoDB client for retrieval
        """
        try:
            # Setting the client
            self.mongo_client = MongoDBClient(database_name= DATABASE_NAME)
            # Setting the database
            self.db = self.mongo_client.client[DATABASE_NAME]
            # Setting the collection
            self.chunks_collection = self.db[COLLECTION_NAME]
            
            # Read the config file
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
            # Embeddings model
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model= self._schema_config["embedding_model"]["model_name"],
                api_key = os.getenv("GOOGLE_API_KEY"),
                output_dimensionality=768
            )
            
            # Vector store setup
            self.vector_store = MongoDBAtlasVectorSearch(
                collection= self.chunks_collection,
                embedding= self.embedding_model,
                index_name= INDEX_NAME,
                text_key= CHUNK_TEXT,
                embedding_key= EMBEDDING 
            )
            
        except Exception as e:
            raise MyException(e, sys) from e   
        
    def retrieve_relevant_chunks(self, query: str, session_id: str):
        """
        Retrieve top-k relevant document chunks for a given query,
        filtered to a specific session_id
        Returns : List of dictionaries with chunk info (text, metadata, score)
        """      
        try:
            logging.info(f"Retrieving chunks for query: {query} | session : {session_id}")
            
            # Step 1. Embed the user query 
            query_vector = self.embedding_model.embed_query(query)
            
            # Build aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": INDEX_NAME,
                        "path": EMBEDDING,
                        "queryVector": query_vector,
                        "numCandidates": 4 * 10,
                        "limit": 4,
                        "filter": {"session_id": {"$eq": session_id}}
                        
                    }
                },
                {
                    "$project": {
                        "chunk_text": 1,
                        "filename": 1,
                        "chunk_index": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {"score" : {"$gte": 0.70}}
                }
            ]
            
            # Execute aggregation
            cursor = self.chunks_collection.aggregate(pipeline= pipeline)
            results = list(cursor)
            
            if not results:
                logging.info("No relevant chunks found")
                return []
            
            # Convert to simple dicts 
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "chunk_text": doc.get("chunk_text", ""),
                    "similarity_score": doc.get("metadata", {})
                })
                
            logging.info(f"Retrieved {len(formatted_results)} relevant chunks")
            return formatted_results
        
        except Exception as e:
            raise MyException(e, sys) from e  
        
    def get_answer(self, query: str, session_id: str):
        """
        1. Retrieve relevant chunks
        2. Build context
        3. Call Gemini to generate grounded answer
        """      
        try:
            logging.info(f"Generating answer for query: {query} | session: {session_id}")
            
            relevant_chunks = self.retrieve_relevant_chunks(query=query, session_id= session_id)
            
            if not relevant_chunks:
                return "I couldn't find any relevant information in the uploaded documents."
            
            # Build context and collect sources
            context_parts = []
            
            for chunk in relevant_chunks:
                context_parts.append(chunk["chunk_text"])
                
            context = "\n\n".join(context_parts)
            
            prompt_text = RAG_ANSWER_PROMPT
            prompt = ChatPromptTemplate.from_template(prompt_text)
            
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model= self._schema_config["llm"]["google"]["model_name"],
                google_api_key = os.getenv("GOOGLE_API_KEY"),
                temperature = 0.7,
                max_output_tokens = 600
            )
            
            # Build Simple chain
            chain = (
                {"context": lambda _: context, "question": lambda x: x}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer = chain.invoke(query)
            
            return answer
        
        except Exception as e:
            raise MyException(e, sys) from e
            
    def initiate_data_retrieval(self, session_id: str, query: str) -> str:
        """
        This component does the data retrieval.
        """            
        try:
            return self.get_answer(query= query, session_id= session_id)
        except Exception as e:
            raise MyException(e, sys) from e
            
        
            
            
import os
import sys
import uuid
import tempfile
from multi_doc_chat.configuration.mongodb_connection import MongoDBClient
from multi_doc_chat.constants import DATABASE_NAME,COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SCHEMA_FILE_PATH, API_KEY
from multi_doc_chat.exception import MyException
from multi_doc_chat.logger import logging
from multi_doc_chat.utils.main_utils import read_yaml_file
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from datetime import datetime, timezone
from fastapi import UploadFile
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Optional

from dotenv import load_dotenv 
load_dotenv()

@dataclass
class IngestionArtifact:
    session_id : Optional[str]
    message: str

class DataIngestion:
    def __init__(self, files: List[UploadFile]):
        """
        Initializes the mongo db client
        """
        try:
            # Setting the client
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)
            # Setting the database
            self.db = self.mongo_client.client[DATABASE_NAME]
            # Setting the collection
            self.chunks_collection = self.db[COLLECTION_NAME]
            
            self.files = files
            self.uploaded_at = datetime.now(timezone.utc)
            
            # Read the config file
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) 
    
    @staticmethod    
    def create_session_uuid() -> str:
        """
        Generates a new random UUID (version 4) as a string.
        Returns:
        str: A 36-character UUID like '123e4567-e89b-12d3-a456-426614174000'
        """  
        try:
            return str(uuid.uuid4())  
        except Exception as e:
            raise MyException(e, sys) from e
    
    async def process_file(self, file_content: bytes, file_name: str, session_id: str):
        """
        1. Create a langchain document from the file content
        2. Split the document into chunks using text splitter
        3. Get the embedded vector for each chunk
        4. Store in the database mongodb
        """
        try:
            # Create a langchain document
            document = Document(
                page_content= file_content,
                metadata={
                    "filename": file_name,
                    "session_id": session_id,
                    "uploaded_at": self.uploaded_at.isoformat(),
                    "source": file_name
                    }) 
        
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size= CHUNK_SIZE,
                chunk_overlap= CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True, # nice for traceability
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            
            text_chunks = text_splitter.split_documents([document])  
            
            # Generate embeddings
            embeddings_model = GoogleGenerativeAIEmbeddings(
                model= self._schema_config["embedding_model"]["model_name"],
                api_key = os.getenv("GOOGLE_API_KEY")
            ) 
            
            # Saving to mongodb
            chunk_docs_for_db = []
            for idx, chunk in enumerate(text_chunks):
               chunk_doc = {
                   "session_id": session_id,
                   "filename": file_name,
                   "chunk_index": idx,
                   "chunk_text": chunk.page_content,
                   "metadata": chunk.metadata,
                   "embedding": embeddings_model.embed_documents([chunk.page_content])[0],
                   "created_at": self.uploaded_at,
                   "start_index": chunk.metadata.get("start_index"),
                   "end_index": chunk.metadata.get("start_index", 0) + len(chunk.page_content)
                   }
               chunk_docs_for_db.append(chunk_doc)
            
            if chunk_docs_for_db:
               self.chunks_collection.insert_many(chunk_docs_for_db)  
               
        except Exception as e:
            raise MyException(e, sys)      
        
        
    async def initiate_data_ingestion(self) -> IngestionArtifact:
        """
        This component does the data ingestion.
        If any file is empty, cannot be decoded, or fails to save in MongoDB,
        ingestion stops immediately and returns None with an error message. 
        """  
        logging.info("Entered initiate_data_ingestion method of data_ingestion class")
        try:
            logging.info("Starting data ingestion")
            
            logging.info("Set the session id")
            session_id = self.create_session_uuid()
            logging.info("Successfully created a session id")
            
            logging.info("Starting ingestion over each file")
            for file in self.files:
                filename = file.filename
                
                logging.info(f"Reading bytes for {filename}")
                content_bytes = await file.read()
                if not content_bytes:
                    logging.info(f"{filename} is empty. Stopping ingestion")
                    return IngestionArtifact(session_id= None, message= f"{filename} is empty. No ingestion performed.")
                
                try:
                    # Save to a temporary file
                    suffix = ".pdf" if filename.lower().endswith(".pdf") else (
                        ".docx" if filename.lower().endswith(".docx") else ".txt") 
                    with tempfile.NamedTemporaryFile(delete= False, suffix= suffix) as tmp:
                        tmp.write(content_bytes)
                        tmp_path = tmp.name
                        
                    # Choose loader based on file type
                    if filename.lower().endswith(".pdf"):
                        loader = PyPDFLoader(tmp_path)
                    elif filename.lower().endswith(".docx"):
                        loader = Docx2txtLoader(tmp_path)    
                    else:
                        loader = TextLoader(tmp_path)
                        
                    documents = loader.load()
                    text_content = "\n".join([doc.page_content for doc in documents])            
                
                except Exception as e:
                    logging.error(f"Failed to parse {filename}")
                    return IngestionArtifact(session_id= None, message= f"failed to parse {filename}")    
                
                try:
                    await self.process_file(text_content, file_name= filename, session_id= session_id) 
                except Exception as e:
                    logging.error(f"Failed to save {filename} in MondoDB. Stopping ingestion.")
                    return IngestionArtifact(session_id= None, message= f"Error saving {filename} in MongoDB: {str(e)}")
            
            logging.info("All files ingested successfully.")
            return IngestionArtifact(session_id= session_id, message= "All files ingested successfully")
                
            
        except Exception as e:
            raise MyException(e, sys) from e      
        
           

# def test_ingestion_pipeline():
#     ingestion = DataIngestion()
    
# if __name__ == "__main__":
#     test_ingestion_pipeline()    
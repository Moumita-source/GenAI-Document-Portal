import os

SCHEMA_FILE_PATH = os.path.join("config", "config.yaml")


DATABASE_NAME = "DocumentStorage"
COLLECTION_NAME = "DocumentStorage-Data"
MONGODB_URL_KEY = "MONGODB_URL"
INDEX_NAME = "vector_index"
API_KEY = "GOOGLE_API_KEY"

# Collections columns
SESSION_ID = "session_id"
FILENAME = "filename"
CHUNK_INDEX = "chunk_index"
CHUNK_TEXT = "chunk_text"
META_DATA = "metadata"
EMBEDDING = "embedding"
CREATED_AT = "created_at"
START_INDEX = "start_index"
END_INDEX = "end_index"

CHUNK_SIZE = 800    # characters
CHUNK_OVERLAP = 100 # overlapping characters
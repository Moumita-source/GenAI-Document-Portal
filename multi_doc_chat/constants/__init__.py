import os

SCHEMA_FILE_PATH = os.path.join("config", "config.yaml")


DATABASE_NAME = "DocumentStorage"
COLLECTION_NAME = "DocumentStorage-Data"
MONGODB_URL_KEY = "MONGODB_URL"
API_KEY = "GOOGLE_API_KEY"

CHUNK_SIZE = 800    # characters
CHUNK_OVERLAP = 100 # overlapping characters
# gen-ai-knowledgebase

### Project Setup Guide

Welcome to the **RAG Knowledgebase** project\! This guide will walk you through the steps to get the application up and running on your local machine.

-----

### Prerequisites

Before you begin, ensure you have the following installed:

  * **Python 3.8+**: The project is built with Python.
  * **pip**: Python's package installer.
  * **Git**: For cloning the repository.
  * **Docker** (optional): For running the Milvus vector database locally.

-----

### 1\. Clone the Repository

First, clone the project from its Git repository to your local machine:

```bash
git clone <repository_url>
cd rag_knowledgebase
```

-----

### 2\. Set Up the Environment

This project uses a `.env` file to manage environment-specific variables, including API keys and configuration settings.

#### A. Create the `.env` file

A template `.env.template` is provided to show you the required variables.

1.  Copy the template file to create your local `.env` file:

    ```bash
    cp .env.template .env
    ```

2.  Open the newly created **`.env`** file in a text editor.

#### B. Configure Environment Variables

Fill in the values for each variable in your `.env` file. Do not share this file with anyone or commit it to your Git repository.

  * **`SECRET_KEY`**: A unique, random string used by Flask for session management. You can generate one using `os.urandom(24)`.
  * **`FLASK_ENV`**: For local development, keep this as `development`.
  * **`GEMINI_KEY`**: If you plan to use Google's Gemini API, provide your API key here.
  * **`MILVUS_URI`** and **`MILVUS_TOKEN`**: If you are using Zilliz Cloud, find these connection details in your Zilliz Cloud dashboard.
  * **`SUPABASE_URL`** and **`SUPABASE_KEY`**: If you are using Supabase for storage, get these from your Supabase project settings.
  * **`SUPABASE_BUCKET_NAME`**: The name of your storage bucket in Supabase.
  * **`DEFAULT_MODEL`**: The name of the AI model you want to use for generating responses (e.g., `gemini-pro`).
  * **`ENABLE_OCR`**, **`EXTRACT_TABLES`**, **`EXTRACT_IMAGES`**: Set these to `true` or `false` based on your document processing needs.

**Example `.env` file with values:**

```ini
# Flask Configuration
SECRET_KEY=a_very_secret_and_random_string
FLASK_ENV=development

# OpenAI API Key (optional, for response generation)
GEMINI_KEY=YOUR_GEMINI_API_KEY

# Vector Database (Milvus/Zilliz Configuration)
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=document_embeddings

# Connection details from your Zilliz Cloud dashboard
MILVUS_URI=YOUR_ZILLIZ_URI
MILVUS_TOKEN=YOUR_ZILLIZ_TOKEN

# Supabase Configuration
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_KEY=YOUR_SUPABASE_KEY
SUPABASE_BUCKET_NAME=rag_files

# Storage Configuration
USE_SUPABASE_STORAGE=true
STORE_ORIGINAL_FILES=true

# AI/LLM Configuration
DEFAULT_MODEL=gemini-pro

# DocLing Processing Configuration
ENABLE_OCR=true
EXTRACT_TABLES=true
EXTRACT_IMAGES=true
```

-----
Create config.py file in the app.py root directory
**Example `config.py` file with values:**

```ini
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask/FastAPI settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-default-secret-key'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

    # Milvus/Zilliz settings
    MILVUS_HOST = os.environ.get('MILVUS_HOST', 'localhost')
    MILVUS_PORT = os.environ.get('MILVUS_PORT', '19530')
    MILVUS_URI = os.environ.get('MILVUS_URI')
    MILVUS_TOKEN = os.environ.get('MILVUS_TOKEN')
    COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'document_embeddings')
    DIMENSION = 384

    # Supabase settings
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
    SUPABASE_BUCKET_NAME = os.environ.get('SUPABASE_BUCKET_NAME', 'uploads')

    # Storage settings
    USE_SUPABASE_STORAGE = os.environ.get('USE_SUPABASE_STORAGE', 'False').lower() == 'true'
    STORE_ORIGINAL_FILES = os.environ.get('STORE_ORIGINAL_FILES', 'True').lower() == 'true'
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'html', 'rtf'}

    # Text processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # AI/LLM settings
    GEMINI_KEY = os.environ.get('GEMINI_KEY')
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL', 'gpt-3.5-turbo')

    # Embedding model
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

    # DocLing settings
    ENABLE_OCR = os.environ.get('ENABLE_OCR', 'True').lower() == 'true'
    EXTRACT_TABLES = os.environ.get('EXTRACT_TABLES', 'True').lower() == 'true'
    EXTRACT_IMAGES = os.environ.get('EXTRACT_IMAGES', 'True').lower() == 'true'
```

-----

### 3\. Install Dependencies

Create a virtual environment and install the required Python packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt
```

-----

### 4\. Run the Application

Now you can start the Flask application:

```bash
flask run
```

The application should now be running on `http://127.0.0.1:8000`. You can access the API endpoints or the front-end interface from there.

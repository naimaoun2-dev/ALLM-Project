# LLM Chatbot with RAG (Streamlit + LangChain)

A powerful chatbot application built with Streamlit and LangChain that uses RAG (Retrieval-Augmented Generation) with Chroma DB for knowledge retrieval.

## Features

1. **RAG System**: Search Chroma DB for information or use internet search (placeholder)
2. **Pre-populated Country Database**: Database comes with country information (name, flag colors, capital, population, continent)
3. **Info Type Tracking**: All information is tagged with `info_type` which can be:
   - `db`: Information from the pre-populated database
   - `user`: Information provided by users
   - `internet`: Information retrieved from internet (placeholder)
4. **User Document Input**: If the database doesn't have information, users can provide it and it's saved with metadata `info_type: user`
5. **Restaurant Lookup (Overpass API)**: Ask for restaurants near any city and the app queries OpenStreetMap's Overpass API for live results
6. **Source Indication**: When information is retrieved, the system indicates the info type (db/user/internet)
7. **Download Functionality**: Export chat history as PDF or Excel
8. **Admin Panel**: Password-protected admin panel with 3 login attempts and 30-second lockout
9. **Beautiful Outputs**: Formatted and beautified responses

## Setup

### 1. Install Dependencies: python version :3.13; langchain version: 0.3.27

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Check `.env` and fill in your values

Edit `.env` and add:
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `ADMIN_PASSWORD`: Password for admin panel (optional, defaults to "admin123")
- `MODEL_NAME`: LLM model to use (default: gemini-pro)
- `EMBEDDING_MODEL`: Embedding model (default: models/embedding-001)

### 3. Populate Database with Country Information

Run the population script to fill the database with country data:

```bash
python populate_db.py
```

This will add information about 20 countries including their names, flag colors, capitals, populations, and continents. All entries will have `info_type: db`.

### 4. Run the Application: don't forget to add your gemini key in the .env file

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### Chat Interface
- Type your questions in the chat input
- The system will search the Chroma DB for relevant information
- If no information is found, you'll be prompted to add it
- Ask for restaurants near a city (e.g., "restaurants in Madrid") to fetch live listings via Overpass API

### Adding Documents
1. Go to the sidebar
2. Enter document text in the "Add Document" section
3. Optionally add a title
4. Click "Save Document"
5. The document will be stored with metadata `info_type: user`

### Admin Panel
1. Click on the sidebar
2. Enter the admin password (configured in `.env`)
3. After 3 failed attempts, the account is locked for 30 seconds
4. Admin can view and delete user-provided documents

### Download Chat History
- Use the download buttons in the sidebar to export chat history as:
  - **PDF**: Formatted PDF with chat messages
  - **Excel**: Spreadsheet with all chat data

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── rag_system.py          # RAG system with Chroma DB
├── admin_panel.py         # Admin panel with authentication
├── download_handler.py    # PDF and Excel export functionality
├── country_data.py        # Country data for database population
├── populate_db.py         # Script to populate database with country info
├── requirements.txt       # Python dependencies
├── rag_utils.py            # Tools
├── .env                   # Your environment variables (not in git)
└── chroma_db/             # Chroma DB storage directory (created automatically)
```

## Technologies

- **Streamlit**: Web application framework
- **LangChain**: LLM application framework
- **Chroma DB**: Vector database for embeddings
- **Google Gemini**: LLM and embeddings (via langchain-google-genai)
- **ReportLab**: PDF generation
- **Pandas/OpenPyXL**: Excel generation

## Notes

- The Chroma DB is stored locally in the `chroma_db` directory
- All documents have an `info_type` metadata field: `db`, `user`, or `internet`
- Pre-populated country data has `info_type: db`
- User-provided documents have `info_type: user`
- The system automatically chunks documents for better retrieval
- Internet search is a placeholder (would use `info_type: internet`)

## License

This project is for educational purposes.


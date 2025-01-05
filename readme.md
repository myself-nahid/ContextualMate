# GenZ Contextual Chatbot

This project implements a contextual chatbot capable of answering user queries using data scraped from the GenZMarketing website. The chatbot leverages Streamlit, LangChain, FAISS, HuggingFace Embeddings, and ChatGroq for data retrieval and contextual understanding.

## Features

- Web-based Interface: Built using Streamlit for a user-friendly experience.

- Data Preprocessing: Handles CSV data with error handling.

- Vector Embeddings: Utilizes HuggingFace's all-MiniLM-L6-v2 for text vectorization.

- Vector Database: FAISS is used for efficient similarity searches.

- Language Model Integration: ChatGroq LLM for generating context-based responses.

- Document Splitting: Recursive character splitting for better vectorization and retrieval.

## Project Structure
```
├── main.py  # Streamlit App and Core Logic
├── cleaned_genzmarketing_data.csv  # Preprocessed CSV Data
├── .env  # Environment Variables
├── requirements.txt  # Dependencies
└── README.md  # Project Documentation
```
## Requirements

- Python 3.8+

- Libraries: streamlit, langchain, faiss-cpu, huggingface_hub, pandas

## Setup Instructions

1. Clone the Repository:
```
git clone https://github.com/myself-nahid/ContextualMate.git
cd contextual-chatbot
```
2. Install Dependencies:
```
pip install -r requirements.txt
```
3. Set Environment Variables:
Create a .env file with the following:
```
GROQ_API_KEY=your_api_key
HF_TOKEN=your_huggingface_token
```
4. Run the Application:
```
streamlit run app.py
```
## Usage

- Click the Document Embedding button to create vector embeddings.

- Enter a query in the input box and receive a context-based answer.

## Key Components

- Data Preprocessing: Cleans the CSV file before vectorization.

- Vector Storage: FAISS for efficient document storage and retrieval.

- LLM Integration: ChatGroq for generating responses.

- Embeddings: HuggingFace's all-MiniLM-L6-v2 for vectorization.

## Troubleshooting

- Ensure the .env file is correctly set up.

- Run pip install -r requirements.txt if modules are missing.

## License

- This project is licensed under the MIT License.

## Acknowledgments

- HuggingFace for the embedding model.

- LangChain for the retriever framework.

- FAISS for vector storage.
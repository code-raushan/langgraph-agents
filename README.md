# LangGraph RAG System with Ollama and Groq

This repository contains a Retrieval-Augmented Generation (RAG) system built using LangGraph, leveraging the power of Llama 3.2:3b model from Ollama and Groq for enhanced question-answering capabilities.

## Features

- Document retrieval from specified blog URLs
- Vector store creation using HuggingFace Transformers embeddings
- Document grading for relevance
- Answer generation using Llama 3.2:3b model
- Answer grading for quality assurance
- Flexible model switching between Ollama and Groq

## System Overview

The RAG system is implemented as a state graph with the following main components:

1. Document Retrieval: Fetches relevant documents from a pre-built vector store.
2. Model Creation: Initializes the Llama 3.2:3b model using Ollama.
3. Document Grading: Evaluates the relevance of retrieved documents.
4. Answer Generation: Produces an answer based on relevant documents.
5. Answer Grading: Assesses the quality of the generated answer.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Set up Ollama with the Llama 3.2:3b model
4. (Optional) Set up Groq API key if using Groq

## Usage

1. Start the server:
   ```
   npm start
   ```
2. Send a POST request to `http://localhost:4321/ask` with a JSON body containing the question:
   ```json
   {
     "question": "Your question here"
   }
   ```

## Configuration

- The system uses Ollama by default. To switch to Groq, uncomment the relevant sections in the `createModel` and `createJsonResponseModel` methods.
- Modify the `urls` array in the `buildVectorStore` method to change the source of documents.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


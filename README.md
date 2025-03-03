# Book Summary Retrieval API VIDEO TUTORIAL - https://www.youtube.com/watch?v=jdeQsFsABzs&ab_channel=DjordjeSpasojevic

This project provides an API for retrieving book summaries from a vector database using semantic similarity search. The API is built with FastAPI and utilizes OpenAI's embeddings to perform semantic searches on a ChromaDB vector database.

## Features

- Retrieve book summaries based on semantic similarity to a given query.
- Uses OpenAI's `text-embedding-ada-002` model for generating embeddings.
- FastAPI is used to create a RESTful API endpoint for querying the database.

## Project Structure

- `src/app.py`: Contains the FastAPI application and endpoint for retrieving book summaries.
- `src/query_vector_db.py`: Module for querying the vector database using semantic similarity.
- `src/populate_vector_db.py`: Script to populate the vector database (entry point).

## Setup Instructions

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) should be installed on your system.
- An OpenAI API key is required to generate embeddings. Set this in your environment variables.

### Creating a Conda Environment

1. **Create a new Conda environment:**

   ```bash
   conda create --name book_summary_env python=3.9
   ```

2. **Activate the environment:**

   ```bash
   conda activate book_summary_env
   ```

### Installing Required Libraries

1. **Install the required libraries from `requirements.txt`:**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that your `requirements.txt` file includes all necessary packages such as `fastapi`, `openai`, `chromadb`, and any other dependencies.

### Environment Variables

- Create a `.env` file in the root directory and add your OpenAI API key:

  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```

## Usage

1. **Run the FastAPI application:**

   Navigate to the `src` directory and execute:

   ```bash
   uvicorn app:app --reload
   ```

   This will start the FastAPI server on `http://127.0.0.1:8000`.

2. **Access the API:**

   - Use the `/retrieve` endpoint to query the vector database. Send a POST request with a JSON body containing the query string.

   Example request body:

   ```json
   {
     "query": "What is the summary of 'The Great Gatsby'?"
   }
   ```

3. **Populate the Vector Database:**

   If you need to populate the vector database, run the `populate_vector_db.py` script:

   ```bash
   python populate_vector_db.py
   ```

## Accessing the API from a TypeScript Frontend

To interact with the `/retrieve` endpoint from a TypeScript frontend application, you can use the `fetch` API or a library like `axios` to make HTTP requests. Below is an example using `fetch`.

### Example using Fetch API

1. **Setup**: Ensure your TypeScript project is set up to use the `fetch` API. If you're using a library like `axios`, make sure it's installed and imported.

2. **Code Example**:

   ```typescript
   async function getBookSummary(query: string) {
     const response = await fetch('http://127.0.0.1:8000/retrieve', {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json',
       },
       body: JSON.stringify({ query }),
     });

     if (!response.ok) {
       throw new Error('Network response was not ok');
     }

     const data = await response.json();
     return data;
   }

   // Usage
   getBookSummary("What is the summary of 'The Great Gatsby'?")
     .then(summary => console.log(summary))
     .catch(error => console.error('Error fetching summary:', error));
   ```

3. **Handling the Response**: The response from the API will be in JSON format. You can access the book summary from the `data` object returned by the `fetch` call.

This example demonstrates how to send a POST request with a JSON body containing the query string and handle the response. Adjust the URL and error handling as needed for your application.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

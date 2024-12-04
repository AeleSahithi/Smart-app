# Smart Apps with LangChain and Google Cloud AI

![cover](https://github.com/user-attachments/assets/5e76828d-c442-419f-9f09-69c65a1d9ed3)

## Introduction
In today's world, information isn't just confined to text; we interact with various data types — images, tables, videos, and more. Traditional language models focus primarily on textual data, leaving valuable insights from non-textual content untapped. Multi-modal Retrieval-Augmented Generation (RAG) systems bridge this gap by combining external knowledge retrieval with advanced generation techniques, ensuring richer, contextually accurate responses.

This blog explores building a multi-modal RAG system using Google Cloud's Vertex AI and LangChain to handle both text and image data.

---

## System Design
### High-Level Architecture:
1. **Document Loading**: Processes text and image data.
2. **Text Summarization**: Summarizes text for efficient retrieval using Gemini Pro.
3. **Image Summarization**: Extracts relevant data from images like charts and graphs.
4. **Multi-Vector Retrieval**: Stores text and image summaries in a vector database for similarity searches.
5. **Multi-Modal Answer Synthesis**: Synthesizes final responses using both text and image data.
Here is a high-level architecture diagram of the system:


---

## Prerequisites
- **Google Cloud Account**: For Vertex AI and related services.
- **Python 3.7+**: For running scripts.
- **Google Cloud SDK**: Installed and authenticated.
- **Libraries**: Install using the following command:
  ```bash
  pip install langchain langchain-google-vertexai chromadb
# Step-by-Step Instructions

## Step 1: Install and Import Dependencies
Start by installing the necessary packages. We'll use LangChain, Google Vertex AI, and Chroma DB for vector retrieval:

```bash
!pip install -U --quiet langchain langchain_google_vertexai chromadb
!pip install --quiet "unstructured[all-docs]" pypdf pillow pydanti
```
Next, authenticate with Google Cloud

```bash
from google.colab import auth
auth.authenticate_user()
```
## Step 2: Prepare and Load Data
We need a collection of documents with both text and images to build our RAG system. For this example, we'll use a sample ZIP file that contains PDF text and image data.

```bash
import zipfile
import requests
# Download and extract data
data_url = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/cj.zip"
result = requests.get(data_url)
filename = "cj.zip"
with open(filename, "wb") as file:
   file.write(result.content)
with zipfile.ZipFile(filename, "r") as zip_ref:
   zip_ref.extractall()
```

## Step 3: Generate Text Summaries
Now, let's generate summaries for the text data using Google's Gemini Pro model. This model is ideal for extracting concise, relevant summaries.

```bash
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
# Initialize the VertexAI model
model = VertexAI(temperature=0, model_name="gemini-pro", max_output_tokens=1024)
# Define a summarization function
def generate_text_summaries(texts):
    summaries = []
    for text in texts:
        prompt = f"Summarize the following text for quick retrieval: {text}"
        summary = model.generate([prompt])
        summaries.append(summary)
    return summaries
texts = ["Your document text here..."]
text_summaries = generate_text_summaries(texts)
```
the output of text summary looks something like this
![text output](https://github.com/user-attachments/assets/ad876764-7d4b-4142-a4f9-17623c6d1514)


## Step 4:Generate Image Summaries
For images, we'll first encode them in base64, then pass them through Gemini Pro Vision to generate concise image summaries.

```bash
import base64
from PIL import Image
# Function to encode images in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
# Summarize images
def summarize_images(image_paths):
    image_summaries = []
    for image_path in image_paths:
        encoded_image = encode_image(image_path)
        prompt = "Summarize this image for retrieval: "  # Provide a suitable prompt
        summary = model.generate([prompt, encoded_image])
        image_summaries.append(summary)
    return image_summaries
image_paths = ["path/to/image.jpg"]
image_summaries = summarize_images(image_paths)
```
and image summaries look like this when you provide the path to an image and a prompt
![imag esummaries](https://github.com/user-attachments/assets/a527d638-1c14-4977-8305-5c136831f615)


## Step 5:Create Multi-Vector Retrieval
Now, let's use ChromaDB to create a multi-vector retrieval system. We'll store both the text and image summaries for efficient retrieval.

```bash
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma

# Create the multi-vector retriever
vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=VertexAIEmbeddings(model_name="textembedding-gecko"))
retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=InMemoryStore())

# Add summaries to the vectorstore
retriever.vectorstore.add_documents(text_summaries + image_summaries)
```
## Step 6:Build Multi-Modal RAG Chain
Create a RAG chain that can handle both text and image queries, retrieve relevant documents, and synthesize answers.

```bash
from langchain_core.messages import HumanMessage
def multi_modal_rag_chain(retriever):
    model = VertexAI(temperature=0, model_name="gemini-pro-vision", max_output_tokens=1024)
    # RAG chain
    chain = retriever | model
    return chain
# Create RAG chain
rag_chain = multi_modal_rag_chain(retriever)
```

## Step 7:Test the System
Finally, test the system by running a query and seeing how the model retrieves and synthesizes the answer from both text and image data.

```bash
query = "What are the EV/NTM and NTM revenue growth for MongoDB, Cloudflare, and Datadog?"
result = rag_chain.invoke(query)
print(result)query = "What are the EV/NTM and NTM revenue growth for MongoDB, Cloudflare, and Datadog?"
result = rag_chain.invoke(query)
print(result)
```

## Result 
By the end, we'll have a powerful multi-modal RAG system capable of handling complex queries involving both text and images. For example, given a question about the financial performance of companies, the system will retrieve and summarize relevant charts and reports, and provide a comprehensive answer and the output may look something similar to this.
![output](https://github.com/user-attachments/assets/f1d9a334-c070-4bab-9de7-90def9deb730)

and you we can also plot the document or image summaries which results something like this based on the document or image you provide to the model.
![plot](https://github.com/user-attachments/assets/a7ebc9a5-30a3-4e76-a9e4-1caca4a597b6)

The overall representation of the model is shown below.


## What's Next?
Now that we've built a multi-modal RAG system, the possibilities are endless! We can extend the system to include video analysis, integrate more advanced models, or apply it to other domains such as healthcare or legal document review.

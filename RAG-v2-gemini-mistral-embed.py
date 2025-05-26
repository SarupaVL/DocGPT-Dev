import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from PyPDF2 import PdfReader
import re
import json
import string
from collections import Counter
import io
import nltk
from nltk.corpus import stopwords
import pytesseract
from PIL import Image
import fitz
import numpy as np
import faiss

import os
from mistralai import Mistral
from time import sleep
import random

# Add debug messages to track execution
print("Starting application initialization...")

# Download necessary NLTK data right at the beginning
try:
    print("Checking for NLTK data...")
    nltk.data.find('corpora/stopwords')
    print("NLTK data found!")
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    print("NLTK data downloaded!")

# API Key configuration with better error handling
GOOGLE_API_KEY = "REPLACEEEEE"
try:
    print("Configuring Google API...")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("Google API configured!")
except Exception as e:
    print(f"Error configuring Google API: {str(e)}")
    st.error(f"Error configuring Google API: {str(e)}")

# Initialize Mistral client globally (will be set when user provides API key)
mistral_client = None

system_prompt = """You are a document assistant. You can read and understand the document uploaded by user to answer the question asked by user.
Read and understand the content/context fully inorder to make the correct answer. Don't make any answer of your own. Suppose if you don't have a content or the content is empty, then convey to user that you can't answer that question because its hard to find the detail in the document nicely.
Make your **response more {response_format}**

IMPORTANT: At the end of your answer, include a confidence score from 1-10 where:
- 1-3: Low confidence (limited or no relevant information found in document)
- 4-6: Medium confidence (some relevant information, but may be incomplete)
- 7-10: High confidence (clear and complete information found in document)

Format your confidence score like this: [Confidence: X/10]"""

ADDN_QUERY_PROMPT = """You are follow-up query generated. The user will share a query that they are searching in a document (like pdf, docs etc.). Your job is to generate 2 or 3 more queries with same context. Something like follow-up queries (includes what could user will ask after this query relatively) and related queries (related to the query in different tone).
Use following JSON schema for response:
{
    "follow_up_query": [
        "follow_up_query_1",
        "follow_up_query_2"
    ],
    "related_queries": [
        "related_query_1",
        "related_query_2"
    ]
}

NOTE:
- If your are not sure about the context of the user query like you don't have any idea about the query, then just use user query as follow-up and related queries, DON'T use any random stuff before you knowing about the context fully.
- Make sure your response is parsable using json.loads in python."""

def initialize_mistral_client(api_key):
    """Initialize Mistral client with provided API key."""
    global mistral_client
    try:
        mistral_client = Mistral(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing Mistral client: {str(e)}")
        return False

def get_mistral_embedding(text, max_retries=3, base_delay=1):
    """Get embedding using Mistral API with rate limiting and retry logic."""
    global mistral_client
    if mistral_client is None:
        raise ValueError("Mistral client not initialized. Please provide API key.")
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to avoid hitting rate limits
            if attempt > 0:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit, waiting {delay:.2f} seconds before retry {attempt + 1}")
                sleep(delay)
            
            embeddings_batch_response = mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return embeddings_batch_response.data[0].embedding
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                    print(f"Rate limit exceeded, waiting {delay:.2f} seconds before retry {attempt + 1}")
                    sleep(delay)
                    continue
                else:
                    print(f"Rate limit exceeded after {max_retries} attempts")
                    st.error("⚠️ Mistral API rate limit exceeded. Please wait a moment and try again.")
                    raise e
            else:
                print(f"Error getting Mistral embedding: {error_str}")
                st.error(f"Error getting Mistral embedding: {error_str}")
                raise e
    
    raise Exception("Failed to get embedding after all retries")

def preprocess_text(text):
    """Simple text preprocessing function."""
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    except Exception as e:
        print(f"Error in preprocess_text: {str(e)}")
        return text.lower().split()  # Fallback

def remove_duplicates(keywords):
    """Remove duplicate keywords using string similarity."""
    try:
        # Pre-process keywords
        processed_keywords = []
        for kw in keywords:
            if not kw or not isinstance(kw, str):
                continue  # Skip non-string or empty keywords
            tokens = set(preprocess_text(kw))
            processed_keywords.append((kw, tokens))
        
        # Remove duplicates based on token overlap
        unique_keywords = []
        for i, (kw, tokens) in enumerate(processed_keywords):
            is_unique = True
            for j, (_, other_tokens) in enumerate(processed_keywords[:i]):
                # If more than 70% of tokens overlap, consider it a duplicate
                if tokens and other_tokens:
                    intersection = tokens.intersection(other_tokens)
                    union = tokens.union(other_tokens)
                    if union:  # Avoid division by zero
                        overlap = len(intersection) / len(union)
                        if overlap > 0.7:
                            is_unique = False
                            break
            if is_unique:
                unique_keywords.append(kw)
        
        return unique_keywords
    except Exception as e:
        print(f"Error in remove_duplicates: {str(e)}")
        return list(set([k for k in keywords if k and isinstance(k, str)]))  # Fallback

def check_content_relevance(content, query, keyword):
    """Check if content is relevant to the query without using LLM."""
    try:
        if not content or not query or not keyword:
            return False

        # Simple relevance check - both query terms and keyword should be present
        query_tokens = set(preprocess_text(query))
        content_tokens = set(preprocess_text(content))
        
        # Check if keyword is present
        keyword_presence = any(k in content.lower() for k in keyword.lower().split())
        
        # Calculate overlap with query terms
        if query_tokens:
            query_overlap = len(query_tokens.intersection(content_tokens)) / len(query_tokens)
        else:
            query_overlap = 0
        
        # Content is relevant if it contains the keyword and has sufficient query term overlap
        return keyword_presence and query_overlap > 0.3
    except Exception as e:
        print(f"Error in check_content_relevance: {str(e)}")
        return True  # Default to including content in case of error
    
def extract_text_with_ocr(pdf_file):
    """
    Extract text from scanned PDF using OCR via PyMuPDF + Tesseract.
    """
    try:
        print("Extracting text using OCR...")
        pdf_bytes = pdf_file.getvalue()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        content = ""
        num_pages = 0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)  # High-res for better OCR
            img = Image.open(io.BytesIO(pix.tobytes("png")))  # Convert to PIL image
            text = pytesseract.image_to_string(img)
            content += text + "\n"
            num_pages += 1

            if num_pages % 5 == 0:
                print(f"OCR processed {num_pages} pages...")

        print(f"OCR completed: {num_pages} pages processed")
        return content, num_pages
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        raise e

class MistralVectorStore:
    """Custom vector store using Mistral embeddings and FAISS."""
    
    def __init__(self):
        self.texts = []
        self.embeddings = None
        self.index = None
    
    def add_texts(self, texts):
        """Add texts to the vector store with rate limiting."""
        global mistral_client
        if mistral_client is None:
            raise ValueError("Mistral client not initialized. Please provide API key.")
            
        print(f"Creating embeddings for {len(texts)} text chunks using Mistral...")
        st.info(f"🔄 Creating {len(texts)} embeddings with rate limiting to avoid API limits...")
        
        self.texts = texts
        
        # Get embeddings for all texts with rate limiting
        embeddings_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate delay between requests to stay under rate limits
        # Mistral free tier: ~5 requests per second, we'll be conservative with 2 per second
        request_delay = 0.5  # 500ms between requests
        
        for i, text in enumerate(texts):
            status_text.text(f"🔍 Processing embedding {i+1}/{len(texts)} (Rate limited for API stability)")
            progress_bar.progress((i + 1) / len(texts))
            
            try:
                # Add delay between requests (except for the first one)
                if i > 0:
                    sleep(request_delay)
                
                embedding = get_mistral_embedding(text)
                embeddings_list.append(embedding)
                
            except Exception as e:
                st.error(f"❌ Error creating embedding for chunk {i+1}: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                raise e
        
        progress_bar.empty()
        status_text.empty()
        
        # Convert to numpy array
        self.embeddings = np.array(embeddings_list)
        
        # Create FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        
        st.success(f"✅ Successfully created {len(embeddings_list)} embeddings!")
        print("Mistral vector store created successfully!")
    
    def similarity_search(self, query, k=4):
        """Search for similar documents."""
        if self.index is None:
            raise ValueError("Vector store not initialized")
        
        # Get embedding for query using Mistral
        query_embedding = np.array([get_mistral_embedding(query)])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return Document-like objects for compatibility
        class Document:
            def __init__(self, page_content):
                self.page_content = page_content
        
        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(Document(self.texts[idx]))
        
        return results

class RAG_v2_gemini:
    def __init__(self, system_prompt: str, chunk_size: int, temperature: float, verbose: bool) -> None:
        try:
            print("Initializing RAG_v2_gemini...")
            self.system_prompt = system_prompt
            self.chunk_size = chunk_size
            self.generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            print("Setting up Gemini models...")
            self.core_gemini = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config,
                system_instruction=self.system_prompt,
            )
            self.core_chat = self.core_gemini.start_chat()
            self.utils_gemini = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=self.generation_config,
            )
            
            # Initialize Mistral vector store
            print("Setting up Mistral vector store...")
            self.vector_store = None
            self.verbose = verbose
            print("RAG_v2_gemini initialized successfully!")
        except Exception as e:
            print(f"Error initializing RAG_v2_gemini: {str(e)}")
            st.error(f"Error initializing RAG system: {str(e)}")
            raise e

    def load_vectorestore(self, context):
        try:
            global mistral_client
            if mistral_client is None:
                st.error("Mistral client not initialized. Please provide your Mistral API key.")
                return
                
            print("Loading vector store with Mistral embeddings...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
            text_chunks = text_splitter.split_text(context)
            print(f"Text split into {len(text_chunks)} chunks")
            
            if not text_chunks:
                raise ValueError("No valid text chunks extracted from document.")

            # Use Mistral vector store
            self.vector_store = MistralVectorStore()
            self.vector_store.add_texts(text_chunks)
            print("Vector store loaded successfully with Mistral embeddings!")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            st.error(f"Error loading vector store: {str(e)}")
            raise e

    def extract_keywords_from_query(self, query):
        """Extract keywords from query without using LLM."""
        try:
            if not query:
                return []
                
            # Simple extraction - remove stopwords and keep meaningful terms
            tokens = preprocess_text(query)
            # Get most frequent terms
            counter = Counter(tokens)
            # Return top keywords (or all if less than 3)
            top_keywords = [word for word, _ in counter.most_common(3)]
            # Always include the full query as a "keyword" for retrieval
            if query and query.strip():
                return [query] + top_keywords
            return top_keywords
        except Exception as e:
            print(f"Error extracting keywords: {str(e)}")
            return [query]  # Fallback to just using the original query

    def get_answer(self, question):
        try:
            global mistral_client
            if mistral_client is None:
                return "Please provide your Mistral API key to use embeddings. [Confidence: 0/10]"
                
            print(f"Processing question: {question}")
            start_time = time.time()
            logs = {
                "user_prompt": question
            }

            # 1. Follow-up & related query generation (still using LLM as this is valuable)
            print("Generating follow-up queries...")
            try:
                res = self.utils_gemini.generate_content(f"{ADDN_QUERY_PROMPT}\n\nHere is the query: {question}").candidates[0].content.parts[0].text
                pattern = re.compile(r'```json\s*(\{.*\})\s*```', re.DOTALL)
                match = pattern.search(res)
                res = match.group(1) if match else res
                additional_queries = json.loads(res)
                follow_ups = additional_queries.get('follow_up_query', [])
                related_queries = additional_queries.get('related_queries', [])
                print("Follow-up queries generated successfully!")
            except Exception as e:
                print(f"Error generating follow-up queries: {str(e)}")
                follow_ups = [question]
                related_queries = []
                
            query_list = follow_ups + related_queries
            query_list.append(question)
            logs['query_list'] = query_list
            
            # 2. Convert queries to keywords WITHOUT using LLM
            print("Extracting keywords...")
            key_words = []
            for query in query_list:
                # Extract keywords using our function instead of LLM
                extracted_keywords = self.extract_keywords_from_query(query)
                key_words.extend(extracted_keywords)
            logs['keywords'] = key_words
            
            # 3. Remove duplicates WITHOUT using LLM
            print("Removing duplicate keywords...")
            ref_key_words = remove_duplicates(key_words)
            print(f"Refined keywords: {ref_key_words}")
            st.write("**Search Keywords:**", ref_key_words)
            logs['refined_keywords'] = ref_key_words

            # Check if vector store is initialized
            if not self.vector_store:
                print("Vector store not initialized!")
                return "Please upload and process documents first. [Confidence: 0/10]"

            # 4. Get relevant contents using Mistral embeddings
            print("Searching for relevant content using Mistral embeddings...")
            relevent_contents = []
            for ref_key in ref_key_words:
                try:
                    related_docs = self.vector_store.similarity_search(ref_key)
                    # Convert the search results to string format
                    related_docs_str = "\n".join([doc.page_content for doc in related_docs])
                    relevent_contents.append(related_docs_str)
                except Exception as e:
                    print(f"Error in similarity search for {ref_key}: {str(e)}")
                    st.error(f"Error in similarity search for {ref_key}: {str(e)}")
            logs['relevent_contents'] = relevent_contents

            # 5. Use all relevant content
            print("Processing relevant content...")
            ref_relevent_content = relevent_contents
            logs['refined_relevent_content'] = ref_relevent_content
            print(f"Considering {len(ref_relevent_content)} contents.")
            
            # If no relevant content found, inform the user
            if not ref_relevent_content:
                print("No relevant content found.")
                response = "I couldn't find relevant information in the document to answer your question. Could you please rephrase or ask something else related to the document content? [Confidence: 1/10]"
                end_time = time.time()
                time_took = end_time - start_time
                print(f"Time took: {time_took:.2f}s")
                logs['final_answer'] = response
                logs['time_took'] = time_took
                st.info(f"⏱️ Time took: {time_took:.2f}s")
                return response
            
            # Use LLM to generate final answer
            print("Generating final answer...")
            try:
                res = self.core_chat.send_message(f"Here is the content: {str(ref_relevent_content)}\n\nQuestion: {question}")
                response = res.candidates[0].content.parts[0].text.strip()
                print("Answer generated successfully!")
            except Exception as e:
                print(f"Error generating answer: {str(e)}")
                response = f"I encountered an error while generating the answer: {str(e)}. Please try again. [Confidence: 0/10]"
            
            # If response doesn't include confidence score, add a default one
            if "[Confidence:" not in response:
                # Determine confidence based on amount of relevant content
                if len(ref_relevent_content) > 3:
                    confidence = 8
                elif len(ref_relevent_content) > 1:
                    confidence = 6
                else:
                    confidence = 4
                response += f"\n\n[Confidence: {confidence}/10]"
                
            end_time = time.time()
            time_took = end_time - start_time
            print(f"Answer processing completed in {time_took:.2f}s")
            logs['final_answer'] = response
            logs['time_took'] = time_took
            st.info(f"⏱️ Time took: {time_took:.2f}s | 🔍 Using Mistral embeddings")
            return response
        except Exception as e:
            print(f"Critical error in get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}. Please try again. [Confidence: 0/10]"

def main():
    try:
        print("Starting Streamlit app...")
        st.set_page_config(
            page_title="RAG-V2 Document Assistant with Mistral Embeddings",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📄 RAG-V2 with Mistral Embeddings")
        st.markdown("Upload documents and ask questions about their content using Mistral embeddings.")
        
        # Check session state
        if "agent" not in st.session_state:
            print("Creating agent session state")
            st.session_state.agent = None

        if "history" not in st.session_state:
            print("Creating history session state")
            st.session_state.history = []
            
        # Display chat history
        print("Displaying chat history")
        for history in st.session_state.history:
            with st.chat_message(history["role"]):
                st.markdown(history["text"])
        
        # Create sidebar
        with st.sidebar:
            st.title("🔧 Configuration")
            
            # Mistral API Key input
            st.subheader("🔑 Mistral API Key")
            mistral_api_key = st.text_input(
                "Enter your Mistral API Key:", 
                type="password",
                help="Required for embeddings. Get your API key from https://console.mistral.ai/"
            )
            
            # Rate limiting information
            st.info("ℹ️ **Rate Limiting**: The app automatically adds delays between API calls to avoid rate limits. Document processing may take longer but will be more reliable.")
            
            if mistral_api_key:
                if initialize_mistral_client(mistral_api_key):
                    st.success("✅ Mistral client initialized!")
                else:
                    st.error("❌ Failed to initialize Mistral client")
            else:
                st.warning("⚠️ Please provide your Mistral API key to use embeddings")
            
            st.divider()
            
            st.subheader("⚙️ Model Settings")
            temperature = st.slider(
                label="Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.75, 
                step=0.05, 
                help="Helps to generate more creative content when the temperature is high."
            )
            
            chunk_size = st.slider(
                label="Chunk size", 
                min_value=300, 
                max_value=1000, 
                value=700, 
                step=50, 
                help="Set the character size for document chunks. Larger chunks provide more context but slow down processing."
            )
            
            verbose = st.checkbox(
                "Show debug info", 
                value=False,
                help="If enabled, additional debugging information will be displayed."
            )
            
            response_format = st.selectbox(
                "Response format", 
                options=["concise", "detailed"],
                help="Choose whether answers should be brief or comprehensive."
            )
            
            _set = st.button("🚀 Initialize RAG System", disabled=not mistral_api_key)
            
            if _set:
                if not mistral_api_key:
                    st.error("Please provide your Mistral API key first!")
                else:
                    try:
                        print("Initializing RAG system...")
                        with st.spinner("Initializing RAG system..."):
                            st.session_state.agent = RAG_v2_gemini(
                                system_prompt=system_prompt.format(response_format=response_format),
                                chunk_size=chunk_size,
                                temperature=temperature,
                                verbose=verbose
                            )
                        st.success("🎉 RAG system ready with Mistral embeddings!")
                        print("RAG system initialization successful")
                    except Exception as e:
                        print(f"Error initializing RAG system: {str(e)}")
                        st.error(f"Error initializing RAG system: {str(e)}")
            
            st.divider()
            st.subheader("📄 Document Upload")

            extraction_method = st.radio(
                "PDF Text Extraction Method",
                options=["Parse", "OCR"],
                index=0,
                help="Parse: Normal text extraction (faster, works for digital PDFs)\nOCR: Optical Character Recognition (slower, better for scanned documents)"
            )

            files = st.file_uploader(
                "Upload your files", 
                accept_multiple_files=True, 
                type=["pdf", "txt"],
                help="Upload PDF or text files to analyze"
            )
            
            process = st.button("📊 Process Documents", disabled=not (mistral_api_key and st.session_state.agent))
            
            if process and files:
                if not mistral_api_key:
                    st.error("Please provide your Mistral API key first!")
                elif st.session_state.agent is not None:
                    try:
                        with st.spinner('🔄 Loading your files and creating embeddings... This may take a while...'):
                            print(f"Processing {len(files)} files...")
                            total_content = ''
                            num_pages = 0
                            
                            for file in files:
                                try:
                                    file_type = file.type
                                    print(f"Processing file: {file.name}, type: {file_type}")
                                    
                                    if file_type == "application/pdf":
                                        if extraction_method == "Parse":
                                            # Regular PDF text extraction
                                            pdf_reader = PdfReader(file)
                                            content = ''
                                            file_pages = len(pdf_reader.pages)
                                            for page in pdf_reader.pages:
                                                content += page.extract_text()
                                            num_pages += file_pages
                                            print(f"Extracted {file_pages} pages from PDF using parsing")
                                        else:  # OCR extraction
                                            # Show a progress message for OCR which is slower
                                            with st.status(f"OCR processing {file.name}... This might take longer", expanded=True) as status:
                                                content, file_pages = extract_text_with_ocr(file)
                                                num_pages += file_pages
                                                status.update(label=f"Completed OCR for {file.name}: {file_pages} pages")
                                            print(f"Extracted {file_pages} pages from PDF using OCR")

                                    elif file_type == "text/plain":
                                        content = file.read()
                                        content = content.decode("utf-8")
                                        print(f"Processed text file: {len(content)} characters")
                                    
                                    else:
                                        st.warning(f"Unsupported file type: {file_type}")
                                        continue

                                    total_content += content
                                except Exception as e:
                                    print(f"Error processing file {file.name}: {str(e)}")
                                    st.error(f"Error processing file {file.name}: {str(e)}")
                                                        
                            if total_content:
                                print(f"Content extracted (first 500 chars):\n{total_content[:500]}")
                                print(f"Loading vector store with {len(total_content)} characters...")
                                st.session_state.agent.load_vectorestore(total_content)
                                print("Vector store loaded successfully")
                                st.success(f"✅ Documents processed with Mistral embeddings: {len(files)} files, {num_pages} PDF pages")
                            else:
                                st.warning("No content extracted from files")
                    except Exception as e:
                        print(f"Error during document processing: {str(e)}")
                        st.error(f"Error during document processing: {str(e)}")
                else:
                    st.info("Please initialize the RAG system first")
                    print("RAG system not initialized")

        # Chat interface
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not mistral_api_key:
                st.error("Please provide your Mistral API key in the sidebar!")
            elif st.session_state.agent is not None:
                try:
                    print(f"User prompt: {prompt}")
                    st.session_state.history.append({"role": "user", "text": prompt})
                    
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        with st.spinner("🔍 Generating answer using Mistral embeddings..."):
                            response = st.session_state.agent.get_answer(prompt)
                            
                        message_placeholder.markdown(response)
                        
                    st.session_state.history.append({"role": "assistant", "text": response})
                    print("Response generated successfully")
                except Exception as e:
                    print(f"Error processing user input: {str(e)}")
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.info("Please initialize the RAG system first")
                print("RAG system not initialized")
                
        print("Streamlit app initialized successfully")
    except Exception as e:
        print(f"Critical error in main: {str(e)}")
        st.error(f"Application error: {str(e)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        st.error(f"Fatal application error: {str(e)}")

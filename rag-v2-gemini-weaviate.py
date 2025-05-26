import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

import weaviate
from weaviate.auth import AuthApiKey
import weaviate.classes as wvc

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
API_KEY = "AIzaSyA-9fewNOpA7lJqRri2F8Ce4e6VRIKzHaU"
try:
    print("Configuring Google API...")
    genai.configure(api_key=API_KEY)
    print("Google API configured!")
except Exception as e:
    print(f"Error configuring Google API: {str(e)}")
    st.error(f"Error configuring Google API: {str(e)}")

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

class RAG_v2_weaviate:
    def __init__(self, system_prompt: str, chunk_size: int, temperature: float, verbose: bool, 
                 weaviate_url: str = None, weaviate_api_key: str = None) -> None:
        try:
            print("Initializing RAG_v2_weaviate...")
            import os
            os.environ["GOOGLE_API_KEY"] = API_KEY
            self.system_prompt = system_prompt
            self.chunk_size = chunk_size
            self.verbose = verbose
            self.index_name = "DocumentChunk"
            
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
            
            print("Setting up embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=os.environ["GOOGLE_API_KEY"]
            )
            
            # Initialize Weaviate client
            print("Initializing Weaviate client...")
            self.weaviate_client = self._initialize_weaviate_client(weaviate_url, weaviate_api_key)
            self.vector_store = None
            
            print("RAG_v2_weaviate initialized successfully!")
        except Exception as e:
            print(f"Error initializing RAG_v2_weaviate: {str(e)}")
            st.error(f"Error initializing RAG system: {str(e)}")
            raise e

    def _initialize_weaviate_client(self, weaviate_url: str = None, weaviate_api_key: str = None):
        """Initialize Weaviate client with proper configuration for v4.14.4."""
        try:
            if weaviate_url and weaviate_api_key:
                # Weaviate Cloud Services configuration (v4+ syntax)
                print(f"Connecting to Weaviate Cloud: {weaviate_url}")
                
                import weaviate
                
                # For Weaviate client v4.14.4
                client = weaviate.connect_to_wcs(
                    cluster_url=weaviate_url,
                    auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
                    headers={
                        "X-Google-Api-Key": API_KEY  # For Google embeddings
                    }
                )
                
                # Test connection - v4 uses is_connected() instead of is_ready()
                if client.is_connected():
                    print("✅ Connected to Weaviate Cloud Services!")
                    return client
                else:
                    raise Exception("❌ Weaviate Cloud connection failed - cluster not connected")
                    
            else:
                # Try local Weaviate instance as fallback
                print("Trying local Weaviate instance...")
                client = weaviate.connect_to_local()
                if client.is_connected():
                    print("✅ Connected to local Weaviate instance!")
                    return client
                else:
                    raise Exception("❌ No Weaviate instance available")
                        
        except Exception as e:
            print(f"❌ Weaviate connection error: {str(e)}")
            raise Exception(f"Could not connect to Weaviate: {str(e)}")

    def load_vectorestore(self, context, source_name="document"):
        try:
            print("Loading vector store...")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=50
            )
            text_chunks = text_splitter.split_text(context)
            print(f"Text split into {len(text_chunks)} chunks")
            
            if not text_chunks:
                raise ValueError("No valid text chunks extracted from document.")

            # Create documents for LangChain
            from langchain.schema import Document
            documents = []
            for i, chunk in enumerate(text_chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "chunk_id": i
                    }
                )
                documents.append(doc)

            # Initialize WeaviateVectorStore and add documents
            print("Creating Weaviate vector store...")
            self.vector_store = WeaviateVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.weaviate_client,
                index_name=self.index_name,
                text_key="content"
            )
            
            print(f"✅ Vector store loaded successfully with {len(text_chunks)} chunks!")
            
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
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
            print(f"Processing question: {question}")
            start_time = time.time()
            logs = {
                "user_prompt": question
            }

            # 1. Follow-up & related query generation
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
            
            # 2. Extract keywords
            print("Extracting keywords...")
            key_words = []
            for query in query_list:
                extracted_keywords = self.extract_keywords_from_query(query)
                key_words.extend(extracted_keywords)
            logs['keywords'] = key_words

            # 3. Remove duplicates
            print("Removing duplicate keywords...")
            ref_key_words = remove_duplicates(key_words)
            print(f"Refined keywords: {ref_key_words}")
            st.write(f"🔍 **Search Keywords**: {ref_key_words}")
            logs['refined_keywords'] = ref_key_words

            # Check if vector store is initialized
            if not self.vector_store:
                print("Vector store not initialized!")
                return "Please upload and process documents first. [Confidence: 0/10]"

            # 4. Search for relevant content
            print("Searching for relevant content...")
            relevant_contents = []
            for ref_key in ref_key_words:
                try:
                    # Use similarity search
                    related_docs = self.vector_store.similarity_search(
                        query=ref_key,
                        k=4  # Number of similar documents to retrieve
                    )
                    # Extract content from documents
                    if related_docs:
                        related_docs_str = "\n".join([doc.page_content for doc in related_docs])
                        relevant_contents.append(related_docs_str)
                except Exception as e:
                    print(f"Error in similarity search for {ref_key}: {str(e)}")
                       
            logs['relevant_contents'] = relevant_contents

            # 5. Filter relevant content
            print("Filtering relevant content...")
            ref_relevant_content = []
            for content, keyword in zip(relevant_contents, ref_key_words):
                try:
                    ref_relevant_content.append(content)
                except Exception as e:
                    print(f"Error checking relevance: {str(e)}")

            logs['refined_relevant_content'] = ref_relevant_content
            print(f"Considering {len(ref_relevant_content)} contents.")
            
            # If no relevant content found, inform the user
            if not ref_relevant_content:
                print("No relevant content found.")
                response = "I couldn't find relevant information in the document to answer your question. Could you please rephrase or ask something else related to the document content? [Confidence: 1/10]"
                end_time = time.time()
                time_took = end_time - start_time
                print(f"Time took: {time_took:.2f}s")
                logs['final_answer'] = response
                logs['time_took'] = time_took
                st.info(f"Time took: {time_took:.2f}s")
                return response
            
            # 6. Generate final answer
            print("Generating final answer...")
            try:
                res = self.core_chat.send_message(f"Here is the content: {str(ref_relevant_content)}\n\nQuestion: {question}")
                response = res.candidates[0].content.parts[0].text.strip()
                print("Answer generated successfully!")
            except Exception as e:
                print(f"Error generating answer: {str(e)}")
                response = f"I encountered an error while generating the answer: {str(e)}. Please try again. [Confidence: 0/10]"
            
            # Add confidence score if not present
            if "[Confidence:" not in response:
                if len(ref_relevant_content) > 3:
                    confidence = 8
                elif len(ref_relevant_content) > 1:
                    confidence = 6
                else:
                    confidence = 4
                response += f"\n\n[Confidence: {confidence}/10]"
            
            end_time = time.time()
            time_took = end_time - start_time
            print(f"✅ Answer processing completed in {time_took:.2f}s")
            logs['final_answer'] = response
            logs['time_took'] = time_took
            st.info(f"Time took: {time_took:.2f}s")
            return response
        except Exception as e:
            print(f"❌ Critical error in get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}. Please try again. [Confidence: 0/10]"

def main():
    try:
        print("Starting Streamlit app...")
        st.set_page_config(
            page_title="RAG-V2 Document Assistant with Weaviate Cloud",
            page_icon="🌐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🌐 RAG-V2 with Weaviate Cloud")
        st.markdown("Upload documents and ask questions using **Weaviate Cloud Services** vector database.")
        
        # Check session state
        if "agent" not in st.session_state:
            st.session_state.agent = None

        if "history" not in st.session_state:
            st.session_state.history = []
            
        # Display chat history
        for history in st.session_state.history:
            with st.chat_message(history["role"]):
                st.markdown(history["text"])
        
        # Create sidebar
        with st.sidebar:
            st.title("Configuration")
            
            # Weaviate Cloud Configuration (moved to top)
            st.subheader("🌐 Weaviate Cloud Setup")
            st.markdown("Get your credentials from [Weaviate Console](https://console.weaviate.cloud/)")
            
            weaviate_url = st.text_input(
                "Cluster URL",
                value="",
                placeholder="https://your-cluster-abc123.weaviate.network",
                help="Your Weaviate Cloud Services cluster URL"
            )
            
            weaviate_api_key = st.text_input(
                "API Key",
                value="",
                type="password",
                placeholder="Your WCS API key",
                help="Your Weaviate Cloud Services API key"
            )
            
            # Connection status indicator
            if weaviate_url and weaviate_api_key:
                st.success("✅ Credentials provided")
            else:
                st.warning("⚠️ Please provide Weaviate credentials")
            
            st.divider()
            
            # Model Configuration
            st.subheader("🤖 Model Settings")
            temperature = st.slider(
                label="Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.75, 
                step=0.05, 
                help="Higher values make output more creative"
            )
            
            chunk_size = st.slider(
                label="Chunk size", 
                min_value=300, 
                max_value=1000, 
                value=700, 
                step=50, 
                help="Size of document chunks for processing"
            )
            
            verbose = st.checkbox(
                "Show debug info", 
                value=False,
                help="Display additional debugging information"
            )
            
            response_format = st.selectbox(
                "Response format", 
                options=["concise", "detailed"],
                help="Choose response style"
            )
            
            st.divider()
            
            # Initialize button
            _set = st.button("Initialize RAG System", type="primary")
            
            if _set:
                if not weaviate_url or not weaviate_api_key:
                    st.error("❌ Please provide both Weaviate URL and API Key")
                else:
                    try:
                        with st.spinner("🔄 Initializing RAG system with Weaviate Cloud..."):
                            st.session_state.agent = RAG_v2_weaviate(
                                system_prompt=system_prompt.format(response_format=response_format),
                                chunk_size=chunk_size,
                                temperature=temperature,
                                verbose=verbose,
                                weaviate_url=weaviate_url,
                                weaviate_api_key=weaviate_api_key
                            )
                        st.success("✅ RAG system ready with Weaviate Cloud!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.divider()
            
            # Document Upload
            st.subheader("📄 Document Upload")

            extraction_method = st.radio(
                "PDF Extraction Method",
                options=["Parse", "OCR"],
                index=0,
                help="Parse: Fast, for digital PDFs | OCR: Slow, for scanned documents"
            )

            files = st.file_uploader(
                "Upload files", 
                accept_multiple_files=True, 
                type=["pdf", "txt"],
                help="Upload PDF or text files"
            )
            
            process = st.button("Process Documents")
            
            if process and files:
                if st.session_state.agent is not None:
                    try:
                        with st.spinner('Processing documents and storing in Weaviate Cloud...'):
                            total_content = ''
                            num_pages = 0
                            
                            for file in files:
                                try:
                                    file_type = file.type
                                    
                                    if file_type == "application/pdf":
                                        if extraction_method == "Parse":
                                            pdf_reader = PdfReader(file)
                                            content = ''
                                            file_pages = len(pdf_reader.pages)
                                            for page in pdf_reader.pages:
                                                content += page.extract_text()
                                            num_pages += file_pages
                                        else:  # OCR
                                            with st.status(f"🔍 OCR processing {file.name}...", expanded=True) as status:
                                                content, file_pages = extract_text_with_ocr(file)
                                                num_pages += file_pages
                                                status.update(label=f"✅ OCR completed: {file_pages} pages")

                                    elif file_type == "text/plain":
                                        content = file.read().decode("utf-8")
                                    
                                    else:
                                        st.warning(f"⚠️ Unsupported file type: {file_type}")
                                        continue

                                    total_content += content
                                except Exception as e:
                                    st.error(f"❌ Error processing {file.name}: {str(e)}")
                                                        
                            if total_content:
                                st.session_state.agent.load_vectorestore(total_content, f"upload_{int(time.time())}")
                                st.success(f"✅ Documents stored in Weaviate Cloud: {len(files)} files, {num_pages} pages")
                            else:
                                st.warning("⚠️ No content extracted from files")
                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")
                else:
                    st.info("Please initialize the RAG system first")

        # Chat interface
        if prompt := st.chat_input("Ask a question about your documents..."):
            if st.session_state.agent is not None:
                try:
                    st.session_state.history.append({"role": "user", "text": prompt})
                    
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        with st.spinner("Generating answer..."):
                            response = st.session_state.agent.get_answer(prompt)
                            
                        message_placeholder.markdown(response)
                        
                    st.session_state.history.append({"role": "assistant", "text": response})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.info("💡 Please initialize the RAG system first")
                
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"❌ Fatal error: {str(e)}")

import time
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from PyPDF2 import PdfReader
import re
import json
import requests

# Google GenAI setup for embeddings
genai.configure(api_key="AIzaSyA-9fewNOpA7lJqRri2F8Ce4e6VRIKzHaU") 

# OpenRouter API configuration
OPENROUTER_API_KEY = ""  # Will be set via UI
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Headers for console output formatting
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

system_prompt = """You are a document assistant. You can read and understand the document uploaded by user to answer the question asked by user.
Read and understand the content/context fully inorder to make the correct answer. Don't make any answer of your own. Suppose if you don't have a content or the content is empty, then convey to user that you can't answer that question because its hard to find the detail in the document nicely.
Make your *response more {response_format}*"""

ADDN_QUERY_PROMPT = """You are follow-up query generated. The user will share a query that they are searching in a document (like pdf, docs etc.). Your job is to generate 2 or 3 more queries with same context. Something like follow-up queries (includes what could user will ask after this query relatively) and related queries (related to the query in different tone).
Use following JSON schema for response:
{
    "follow_up_query": [
        "follow_up_query_1",
        "follow_up_query_2",
    ],
    "related_queries": [
        "related_query_1",
        "related_query_2",
    ]
}

NOTE:
- If your are not sure about the context of the user query like you don't have any idea about the query, then just use user query as follow-up and related queries, DON'T use any random stuff before you knowing about the context fully.
- Make sure your response is parsable using json.loads in python."""

class RAG_v2_openrouter:
    def _init_(self, system_prompt: str, chunk_size: int, temperature: float, openrouter_api_key: str, verbose: bool) -> None:
        import os
        os.environ["GOOGLE_API_KEY"] = "AIzaSyA-9fewNOpA7lJqRri2F8Ce4e6VRIKzHaU"
        self.system_prompt = system_prompt
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.openrouter_api_key = openrouter_api_key
        self.verbose = verbose
        
        # Google Embeddings setup
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.vector_store = None
        
        # We'll track conversation for the OpenRouter chat
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        
        print(f"{OKGREEN}RAG system initialized with OpenRouter GPT-4o and Google Embeddings{ENDC}")
        if self.verbose:
            st.write("System initialized with the following configuration:")
            st.write({
                "Temperature": self.temperature,
                "Chunk Size": self.chunk_size,
                "Verbose": self.verbose,
                "System Prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt
            })

    def load_vectorestore(self, context):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=0)
        text = text_splitter.split_text(context)
        self.vector_store = FAISS.from_texts(text, self.embeddings)

    def call_openrouter_api(self, messages):
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8501",  # Replace with your actual domain
            "X-Title": "RAG-v2 App"  # Your app's name
        }
        
        data = {
            "model": "openai/gpt-4o",  # Using GPT-4o model
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            response_json = response.json()
            if "error" in response_json:
                print(f"API Error: {response_json['error']}")
                # Return a formatted error response that won't break our code
                return {
                    "choices": [{
                        "message": {
                            "content": f"Error processing request: {response_json.get('error', {}).get('message', 'Unknown error')}"
                        }
                    }]
                }
                
            return response_json
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            # Return a formatted error response
            return {
                "choices": [{
                    "message": {
                        "content": f"Error connecting to OpenRouter API: {str(e)}"
                    }
                }]
            }
        except ValueError as e:  # JSON parsing error
            print(f"JSON parsing error: {e}")
            # Return a formatted error response
            return {
                "choices": [{
                    "message": {
                        "content": "Error parsing response from OpenRouter API"
                    }
                }]
            }

    def get_answer(self, question):
        start_time = time.time()
        logs = {
            "user_prompt": question
        }

        try:
            # 1. Follow-up & related query generation
            query_generation_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{ADDN_QUERY_PROMPT}\n\nHere is the query: {question}"}
            ]
            
            res = self.call_openrouter_api(query_generation_messages)
            response_text = res["choices"][0]["message"]["content"]
            
            try:
                # Extract JSON from response
                pattern = re.compile(r'\{.*\}', re.DOTALL)
                match = pattern.search(response_text)
                if match:
                    res_json = match.group(0)
                    try:
                        additional_queries = json.loads(res_json)
                        follow_ups = additional_queries.get('follow_up_query', [])
                        related_queries = additional_queries.get('related_queries', [])
                        query_list = follow_ups + related_queries
                        query_list.append(question)
                    except json.JSONDecodeError:
                        # Fallback if JSON parsing fails
                        query_list = [question]
                else:
                    query_list = [question]
            except Exception as e:
                print(f"{FAIL}Error in query generation: {str(e)}{ENDC}")
                # Fallback to just using the original question
                query_list = [question]
                
            logs['query_list'] = query_list
            
            # 2. Convert queries to keywords
            key_words = []
            for query in query_list:
                keyword_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"""Given a question by user, you have to convert it into nice one simple form or like keyword. Just convert the question into simpler form that looks easier. Don't include any other text, just the query as response.\nFor example:\nWhat all are the related works they considered? -> Related works.\n\nNow here is the query: {query}"""}
                ]
                
                res = self.call_openrouter_api(keyword_messages)
                key_words.append(res["choices"][0]["message"]["content"])
                
            logs['keywords'] = key_words
            
            # 3. Remove duplicates
            refine_keywords_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""The following LIST of key-words may contains some duplicates in terms of spelling, but all the key-words are all in same category or in the same field. If is contains any duplicates you have to remove and return the correct list. If the list is already doesn't have any duplicates, then just return the list. Make sure all the key-words are comma seperated in your response.\nHere is the list: {key_words}"""}
            ]
            
            res = self.call_openrouter_api(refine_keywords_messages)
            refined_key_words = res["choices"][0]["message"]["content"]
            
            ref_key_words = []
            for i in refined_key_words.split(','):
                ref_key_words.append(i.strip())
                
            print(f"{OKBLUE}Refined keywords: {ref_key_words}{ENDC}")
            st.write(ref_key_words)
            logs['refined_keywords'] = ref_key_words

            # Check if vector store is initialized
            if self.vector_store is None:
                return "Please upload and process documents before asking questions."

            # 4. Get relevant contents
            relevent_contents = []
            for ref_key in ref_key_words:
                related_docs = str(self.vector_store.similarity_search(ref_key))
                relevent_contents.append(related_docs)
            logs['relevent_contents'] = relevent_contents

            # 5. Remove unwanted contents
            ref_relevent_content = []
            for i in range(len(relevent_contents)):
                relevance_check_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"CONTENT:\n{relevent_contents[i]}\n\nUnderstand the above content fully. QUESTION: {question}\n\nKEY_WORD: {ref_key_words[i]}.\nCheck if we can able to answer the question by reading the content or Does the content have some context to answer the question.\nAlso check the KEY_WORD is present in the content. Response with only yes if all conditions satisfied or just no."}
                ]
                
                res = self.call_openrouter_api(relevance_check_messages)
                is_relevent = res["choices"][0]["message"]["content"].strip().lower()
                
                if is_relevent == 'yes':
                    ref_relevent_content.append(relevent_contents[i])
                else:
                    print(f"{WARNING}Rejected content {i}{ENDC}")

            logs['refined_relevent_content'] = ref_relevent_content
            print(f"{OKGREEN}Considering only {len(ref_relevent_content)} contents.{ENDC}")
            
            # If no relevant content found
            if len(ref_relevent_content) == 0:
                return "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or upload more relevant documents."
            
            # Update conversation history for this query and get final answer
            final_query_message = f"Here is the content: {str(ref_relevent_content)}\n\nQuestion: {question}"
            self.conversation_history.append({"role": "user", "content": final_query_message})
            
            res = self.call_openrouter_api(self.conversation_history)
            response = res["choices"][0]["message"]["content"].strip()
            
            # Add assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                # Keep system message and last 4 exchanges (8 messages)
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-8:]
            
            end_time = time.time()
            time_took = end_time - start_time
            print(f"{OKBLUE}Time taken: {time_took:.2f}s{ENDC}")
            logs['final_answer'] = response
            logs['time_took'] = time_took
            
            st.info(f"Time took: {time_took:.2f}s")
            
            if self.verbose:
                st.write("Debug logs:", logs)
                
            return response
            
        except Exception as e:
            import traceback
            print(f"{FAIL}Error in get_answer: {str(e)}{ENDC}")
            print(traceback.format_exc())
            return f"An error occurred while processing your question: {str(e)}"

def main():
    st.title("Rag-V2 with OpenRouter GPT-4o")
    
    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "history" not in st.session_state:
        st.session_state.history = []
    
    for history in st.session_state.history:
        with st.chat_message(history["role"]):
            st.markdown(history["text"])
    
    with st.sidebar:
        st.title("Set the model Conf.")
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password")
        temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.75, step=0.05, help="Helps to generate content creative every time when the temperature is high.")
        chunk_size = st.slider(label="Chunk size", min_value=300, max_value=1000, value=700, step=50, help="Set the chars size in a chunk that'll be extracted from the uploaded docs and provide to LLM to generate answer. Larger the size more the detail but slower the generation speed.")
        verbose = st.checkbox("verbose", help="If verbose enabled, the logging will be displayed in the UI.")
        response_format = st.selectbox("Response format", options=["concise", "detailed"])
        
        _set = st.button("Set")
        
        if _set:
            if openrouter_api_key:
                st.session_state.agent = RAG_v2_openrouter(
                    system_prompt=system_prompt.format(response_format=response_format),
                    chunk_size=chunk_size,
                    temperature=temperature,
                    openrouter_api_key=openrouter_api_key,
                    verbose=verbose
                )
                st.success("Rag ready!!")
            else:
                st.error("Please provide an OpenRouter API key")
        
        st.divider()

        files = st.file_uploader("Upload your files", accept_multiple_files=True, type=["pdf", "txt"])
        process = st.button("Process")
        if process and files:
            if st.session_state.agent is not None:
                with st.spinner('loading your file. This may take a while...'):
                    total_content = ''
                    num_pages = 0
                    for file in files:
                        file_type = file.type
                        if file_type == "application/pdf":
                            pdf_reader = PdfReader(file)
                            content = ''
                            for page in pdf_reader.pages:
                                num_pages += 1
                                content += page.extract_text()

                        if file_type == "text/plain":
                            content = file.read()
                            content = content.decode("utf-8")

                        total_content += content

                    st.session_state.agent.load_vectorestore(total_content)
                st.success("Documents loaded.")
            else:
                st.info("Set the model Conf. first...")

    if prompt := st.chat_input("Enter your message..."):
        if st.session_state.agent is not None:
            st.session_state.history.append({"role": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response = st.session_state.agent.get_answer(prompt)
                message_placeholder.markdown(response)
            st.session_state.history.append({"role": "assistant", "text": response})
        else:
            st.info("Set the model Conf. first.")

if _name_ == '_main_':
    main()

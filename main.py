import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
import dotenv
import asyncio

# Load environment variables
dotenv.load_dotenv()

# Initialize components
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Load prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Create the RAG chain
retriever = vector_store.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit App
st.title("ChatURL")
st.logo(".streamlit/assets/logo.png")

# Initialize session state
if "content_indexed" not in st.session_state:
    st.session_state["content_indexed"] = False

# Sidebar for URL Input
with st.sidebar:
    st.header("Load Content")

    # Initialize session state to track loaded URLs
    if "loaded_urls" not in st.session_state:
        st.session_state["loaded_urls"] = []  # A list to store loaded URLs

    # Input for URL
    url = st.text_input("Enter a URL to analyze:", "https://example.com/")

    if st.button("Load URL"):
        if url:  # Ensure the URL is not empty
            # Check if the URL has already been loaded
            if url in st.session_state["loaded_urls"]:
                st.warning("This URL has already been loaded.")
            else:
                # Load and chunk contents of the URL
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(
                            class_=("post-content", "post-title", "post-header")
                        )
                    ),
                )
                try:
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    all_splits = text_splitter.split_documents(docs)

                    # Add new chunks to the vector store
                    _ = vector_store.add_documents(documents=all_splits)

                    # Track the URL as loaded
                    st.session_state["loaded_urls"].append(url)

                    st.success(f"Content from the URL '{url}' has been successfully added!")
                except Exception as e:
                    st.error(f"Error loading content from the URL: {str(e)}")
        else:
            st.error("Please enter a valid URL.")

    # Display the list of loaded URLs
    st.write("### Loaded URLs:")
    if st.session_state["loaded_urls"]:
        for loaded_url in st.session_state["loaded_urls"]:
            st.write(f"- {loaded_url}")
    else:
        st.info("No URLs have been loaded yet.")

# Question-Answering Interface with Chatbot-like UI
if st.session_state.get("loaded_urls"):  # Check if any URLs have been loaded

    # Initialize session state for chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # Stores chat messages as a list of dicts

    # Display chat messages in the UI
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):  # 'user' or 'assistant'
            st.write(message["content"])  # Display the content of the message

    # User input
    if user_input := st.chat_input("Ask a question about the content:"):
        # Add user input to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Display user message in the chat
        with st.chat_message("user"):
            st.write(user_input)

        # Generate assistant's response
        async def stream_response():
            with st.chat_message("assistant"):  # Display assistant's chat message
                placeholder = st.empty()  # Placeholder for updating the response incrementally
                response_text = ""  # Accumulate the assistant's response incrementally

                # Use the retrieval chain to get the response
                response = await retrieval_chain.ainvoke({"input": user_input})
                response_text = response["answer"]
                placeholder.write(response_text)

                # Add the assistant's response to chat history
                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

        # Run the async function
        asyncio.run(stream_response())

else:
    st.info("Please load URL in the sidebar before asking questions.")

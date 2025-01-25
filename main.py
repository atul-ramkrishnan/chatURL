import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
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

# Define state schema
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

async def generate(state: State):
    """
    Generates a response to the user's query using the LLM.

    This function is asynchronous because it streams tokens from the LLM as they 
    are generated. The async behavior allows real-time, non-blocking processing 
    of the streamed response, enabling incremental updates to the user interface.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = await llm.ainvoke(messages)  # Asynchronous invocation for streaming
    return {"answer": response.content}

# Compile workflow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit App
st.title("URL-Based Q&A System")

# Initialize session state
if "content_indexed" not in st.session_state:
    st.session_state["content_indexed"] = False

# URL Input
url = st.text_input("Enter a URL to analyze:", "https://calpaterson.com/agile.html")
if st.button("Load and Index Content"):
    # Load and chunk contents of the blog
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
        
        # Index chunks
        vector_store.delete()  # Clear any previous data
        _ = vector_store.add_documents(documents=all_splits)
        st.session_state["content_indexed"] = True  # Mark content as indexed
        st.success("Content loaded and indexed successfully!")
    except Exception as e:
        st.error(f"Error loading content: {str(e)}")

# Question-Answering Interface with Chatbot-like UI
if st.session_state["content_indexed"]:  # Check if content has been indexed

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

                # Stream response token by token using `stream_mode="messages"`
                async for msg, metadata in graph.astream({"question": user_input}, stream_mode="messages"):
                    if msg.content:  # Ensure there's content to add
                        response_text += msg.content
                        placeholder.write(response_text)  # Update the UI incrementally

                # Finalize display after the stream is complete
                placeholder.write(response_text)

                # Add the assistant's response to chat history
                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

        # Run the async function
        asyncio.run(stream_response())

else:
    st.info("Please load and index content from a URL before asking questions.")
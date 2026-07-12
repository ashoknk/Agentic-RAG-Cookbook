"""==============================================================================
ARCHITECTURAL COMPARISON: 02b_RAGChainSample vs. 02c_LCEL_NewDoc
------------------------------------------------------------------------------
Both files build fully functional RAG pipelines, but they use entirely 
different underlying paradigms to assemble the components and handle the lifecycle.

KEY ARCHITECTURAL DIFFERENCES & IMPROVEMENTS IN 02c:

1. CLASSIC CHAINS VS. LCEL DECORATIVE SYNTAX (LangChain Expression Language)
   - 02b (Classic Chains): Uses legacy wrapper functions like `create_retrieval_chain` 
     and `create_stuff_documents_chain` which encapsulate the pipeline under the hood. 
     The data flow is handled automatically but is less transparent.
   - 02c (LCEL Pipes): Explicitly constructs the pipeline step-by-step using the Linux 
     pipe operator (`|`). This approach is more transparent, modular, and customizable, 
     making it the modern standard for advanced LangChain development.

2. RUNNABLEPASSTHROUGH & INPUT ROUTING
   - 02c utilizes `RunnablePassthrough()` to perfectly control the dictionary keys. 
     When a question string is passed to `rag_chain.invoke(question)`:
       a) The question string bypasses formatting and is sent as-is into the 
          'question' key via `RunnablePassthrough()`.
       b) The same question string is simultaneously routed through the retriever, 
          and the resulting documents are fed into `format_docs` to build the 'context' key.

3. STROUTPUTPARSER FOR CLEANER OUTPUTS
   - 02b returns a complex dictionary containing keys like `['answer']` and `['context']`.
   - 02c appends `| StrOutputParser()` at the end of the execution stream. This instantly 
     intercepts the raw AI message chunk, extracts only the text body, and streams back 
     a clean Python string directly, eliminating dictionary parsing boilerplate.

4. STATIC RETRIEVAL VS. DYNAMIC VECTOR STORE UPDATING
   - 02b is read-only; it connects to a predefined, static vector index and runs queries.
   - 02c implements a mutable database lifestyle. It introduces an interactive 
     `add_new_documents()` execution sequence that processes a raw string token on the fly, 
     chunks it with `text_splitter`, and dynamically updates the persistent ChromaDB collection
     using `vectorstore.add_documents()`
   - 02c performs an immediate validation check by running queries like "What are the 
     key concepts in reinforcement learning?" both before and after the database update.
==============================================================================
"""

import os
import warnings
from dotenv import load_dotenv

from langchain.chat_models.base import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

warnings.filterwarnings('ignore')
load_dotenv()

DATA_DIR = 'data'
PERSIST_DIRECTORY = './chroma_db'
COLLECTION_NAME = 'rag_collection'  # collection_name acts as a unique namespace that groups and stores your specific embeddings


## Load existing Chromadb with Open AI embeddings from disk
def create_vector_store(persist_directory: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the existing vectorstore from disk without .from_documents()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    print(f'Loaded existing vectorstore with {vectorstore._collection.count()} vectors.')
    print(f'Data persisted to: {persist_directory}')
    return vectorstore

## Convert vector store to retriever
def build_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={'k': 3})

# ==========================================
# Part1 : Create a custom prompt using RunnablePassthrough, retriever and format_docs()
# ==========================================
# Create RAG Chain Alternative - Using LCEL (LangChain Expression Language)    

# ChatPromptTemplate.from_template 
#    - EXPECTS: A single, raw, continuous text string.
#    - BEHAVIOR: It automatically creates a default "Human" message role under 
#      the hood or treats the input as a single instruction block. 
#    - BEST FOR: Simple, linear prompts or quick LCEL chains where you just want 
#      to drop variables like {context} and {question} into a static block of text

# `RunnablePassthrough()` to perfectly control the dictionary keys. The question string bypasses formatting and 
# is sent as-is into the 'question' key via `RunnablePassthrough()`.

# The same question string is simultaneously routed through the retriever, 
# and the resulting documents are fed into `format_docs` to build the 'context' key.

# appends `| StrOutputParser()` at the end of the execution stream. 
# This instantly intercepts the raw AI message chunk, extracts only the text body

def build_rag_chain(retriever):
    llm=init_chat_model("openai:gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        '''Use the following context to answer the question.
            If you don't know the answer based on the context, say you don't know.
            Provide specific details from the context to support your answer.

            Context:{context}
            Question: {question}
            Answer:''' )

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain = (
        {
            'context': retriever | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain



def query_rag_lcel(question: str, rag_chain, retriever):
    print('-' * 50)
    print(f'Question: {question}')
    print('-' * 50)

    answer = rag_chain.invoke(question)
    print(f'Answer: {answer}')

    #Understand the source using retriever
    docs = retriever.invoke(question)
    print('\nSource Documents:')
    for index, doc in enumerate(docs, start=1):
        print(f'\n--- Source {index} ---')
        print(doc.page_content[:200] + '...')
    return answer


# ==========================================
### Part2 : Add New Documents To Existing Vector Store ###
# ==========================================
#  Processes a raw string token on the fly, chunks it with `text_splitter`, 
#  and dynamically updates the persistent ChromaDB collection using `vectorstore.add_documents()`
def add_new_documents(vectorstore, text_splitter):
    new_document = '''
Reinforcement Learning in Detail

Reinforcement learning (RL) is a type of machine learning where an agent learns to make
decisions by interacting with an environment. The agent receives rewards or penalties
based on its actions and learns to maximize cumulative reward over time. Key concepts
in RL include: states, actions, rewards, policies, and value functions. Popular RL
algorithms include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and
Actor-Critic methods. RL has been successfully applied to game playing (like AlphaGo),
robotics, and autonomous systems.
'''

    new_doc = Document(
        page_content=new_document,
        metadata={'source': 'manual_addition', 'topic': 'reinforcement_learning'},
    )
    new_chunks = text_splitter.split_documents([new_doc])
    # NOTE Just for testing
    # print(f'Total vectors before: {vectorstore._collection.count()}.')
    vectorstore.add_documents(new_chunks)

    print(f'Added {len(new_chunks)} new chunks to the vector store.')
    print(f'Total vectors now: {vectorstore._collection.count()}.')
    return new_chunks


def main():
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    # documents = load_documents(DATA_DIR)
    # text_splitter, chunks = make_text_chunks(documents)

    # We still need text_splitter isolated to chunk manual additions later on
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=[' '])

    # Connect directly to your existing disk storage
    vectorstore = create_vector_store(PERSIST_DIRECTORY)

    # vectorstore = create_vector_store(chunks, PERSIST_DIRECTORY)
    retriever = build_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)

    print('===================== Testing LCEL Chain: =====================')
    query_rag_lcel('What are the key concepts in reinforcement learning?', rag_chain, retriever)
    query_rag_lcel('What is machine learning?', rag_chain, retriever)
    query_rag_lcel('What is deep learning?', rag_chain, retriever)

    print('\n===================== Add New Documents =====================')
    add_new_documents(vectorstore, text_splitter)

    print('===================== Query Updated Vector Store =====================')
    query_rag_lcel(
        'What are the key concepts in reinforcement learning?',
        rag_chain,
        retriever,
    )


if __name__ == '__main__':
    main()

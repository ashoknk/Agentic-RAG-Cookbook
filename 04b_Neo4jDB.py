"""
================================================================================
This script builds directly upon the previous GraphRAG foundation (`04a_Neo4jDB.py`) 
by introducing **Custom Prompt Engineering**. While the earlier script relied on 
LangChain's default behavior, this version explicitly overrides the final answer 
generation phase to ensure responses are highly readable, authoritative, and 
strictly grounded in the returned database context.

WHAT'S NEW IN THIS VERSION (04b):
- **Prompt Customization (`PromptTemplate`)**: Defines a rigid system persona (`cypher_qa_template`) 
  instructing the LLM to treat raw Neo4j graph results as authoritative.
- **Strict Grounding**: Instructs the LLM *not* to hallucinate outside info, using 
  only the provided database `{context}` to answer the user's `{question}`.
- **Chain Integration**: Injects the `qa_prompt` directly into `GraphCypherQAChain.from_llm()`.

THE LOGICAL FLOW:
1. DATA SEEDING: Execute Cypher queries via CSV to build the Movie/Person/Genre graph schema.
2. PROMPT ENGINEERING: Construct a structured QA template ensuring human-friendly phrasing.
3. CHAIN CONFIGURATION: Instantiate the `GraphCypherQAChain` using the custom prompt injection.
4. TESTS & EVALUATION: Execute multiple test queries spanning multi-hop traversals, 
   aggregations, and filters, observing how the LLM formats raw database responses.
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# Filter by the warning message text instead of the class category
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*LangChainPendingDeprecationWarning.*")

from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain 
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
graph=Neo4jGraph(url=NEO4J_URI,username=NEO4J_USERNAME,password=NEO4J_PASSWORD)


## Dataset Movie
movie_query="""
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/ashoknk/ashnaiku_dataset/main/movies_small.csv' as row

MERGE(m:Movie{id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

# print(movie_query)
graph.query(movie_query)
graph.refresh_schema()
# print(graph.schema)
groq_api_key=os.getenv("GROQ_API_KEY")


llm=ChatGroq(groq_api_key=groq_api_key,model_name="openai/gpt-oss-20b")
print(llm)

#TODO : when you set this to True, it's best practice to ensure your Neo4j database user (NEO4J_USERNAME) only has read-only permissions if you are just querying data, especially if this app will ever be exposed to end-users.

# chain = GraphCypherQAChain.from_llm(
#     graph=graph, 
#     llm=llm, 
#     verbose=True, 
#     allow_dangerous_requests=True  # Add this line
# )

# 1. Define a clear instruction template for the final answer phase
cypher_qa_template = """You are an assistant that helps to form readable and human-understandable answers.
The information part contains the raw database results that you MUST use to construct the response. 
Treat the provided information as authoritative and complete.

Information: {context}
Question: {question}

Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template=cypher_qa_template
)

# 2. Add the qa_prompt to your existing chain definition
chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    verbose=True, 
    allow_dangerous_requests=True,
    qa_prompt=qa_prompt  # 👈 Add this line here
)


print("===============================")
query1 = "Who was the director of the movie GoldenEye?"
response = chain.invoke({"query": query1})
# print("Response:", response)
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")


print("===============================")
query2 = "Who were the actors of the movie Dead Man Walking?"
response = chain.invoke({"query": query2})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")


# print("===============================")
query3 = "How many artists are there in movie Jury Duty?"
response = chain.invoke({"query": query3})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")


print("===============================")
query4 = "How many movies has Nicole Kidman acted in?"
response = chain.invoke({"query": query4})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")


print("===============================")
# Tests traversing: Person -> ACTED_IN -> Movie <- ACTED_IN <- Person
query_5 = "Who are the actors that have co-starred with Tom Hanks in any movie?"
response = chain.invoke({"query": query_5})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")

print("===============================")
# Tests linking a director to actors via shared movie nodes
query_6 = "Which actors have appeared in movies directed by Martin Scorsese?"
response = chain.invoke({"query": query_6})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")

print("===============================")
# Tests string matching and mathematical sorting logic (m.imdbRating)
query_7 = "What is the highest-rated movie in the database, and what is its rating?"
response = chain.invoke({"query": query_7})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")

print("===============================")
# Tests pattern filtering combined with a localized result limit
query_8 = "List the top 3 Drama movies based on their IMDb rating."
response = chain.invoke({"query": query_8})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")

print("===============================")
# Tests node counting grouped by property attributes
query_9 = "How many total movies are categorized under the Romance genre?"
response = chain.invoke({"query": query_9})
print(f"Query:    {response['query']}")
print(f"Response: {response['result']}")


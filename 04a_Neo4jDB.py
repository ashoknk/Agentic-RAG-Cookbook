"""
================================================================================
This script demonstrates how to integrate a Graph Database (Neo4j) with a Large 
Language Model (LLM via Groq) using LangChain. It showcases a modern "GraphRAG" 
pattern where natural language questions are translated into Cypher database 
queries automatically by an LLM.

PREREQUISITES:
- A running Neo4j database instance.
- Environment variables configured in a `.env` file:  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and GROQ_API_KEY.

THE LOGICAL FLOW:
1. DATA SEEDING: Run a Cypher command (`LOAD CSV`) to ingest a small dataset of 
   movies, actors, directors, and genres into the graph database.
2. GRAPH SCHEMA: Neo4j's schema is refreshed and fed into LangChain so the LLM 
   understands the database structure.
3. CHAIN CONFIGURATION: A `GraphCypherQAChain` is built using the Neo4j graph 
   and a Groq-hosted LLM.
4. EXECUTION & EVALUATION: A series of natural language queries are executed to 
   test everything from simple properties to multi-hop relationship traversals 
   and aggregations.

DATABASE SCHEMA CREATED:
------------------------
Nodes:  (:Movie), (:Person), (:Genre)
Edges:  (:Person)-[:DIRECTED]->(:Movie)
        (:Person)-[:ACTED_IN]->(:Movie)
        (:Movie)-[:IN_GENRE]->(:Genre)
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# Filter by the warning message text instead of the class category
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*LangChainPendingDeprecationWarning.*")

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain  
from langchain_groq import ChatGroq

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

chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    verbose=True, 
    allow_dangerous_requests=True  # Add this line
)
# print("Chain:", chain)

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


"""
================================================================================
 EXTERNAL TOOL INTEGRATION & PUBLIC DATA SCRAPING HUB
================================================================================
Demonstrates how to plug LangChain into open-source, freemium, and advanced 
    hybrid information engines to bypass training data cutoffs and hallucinations.

1. Initializes free/freemium live-web scraping tools (DuckDuckGo, Google, Tavily).
2. Sets up targeted technical & academic lookup engines (StackExchange, WolframAlpha, Merriam-Webster).

Role in Agentic RAG:
    These external integrations act as the "hands and eyes" (Tools/Function Calls) 
    that autonomous agents can dynamically invoke when local vector memory is insufficient.
================================================================================
"""

import os
# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

import warnings
import logging
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_tavily import TavilySearch

from langchain_community.document_loaders import PubMedLoader

from langchain_community.tools import StackExchangeTool
from langchain_community.utilities import StackExchangeAPIWrapper

from langchain_community.tools import WolframAlphaQueryRun
from langchain_community.utilities import WolframAlphaAPIWrapper

from langchain_community.tools import MerriamWebsterQueryRun
from langchain_community.utilities import MerriamWebsterAPIWrapper

#NOTE Some of the below clases did not work for me
# from langchain_community.retrievers import WebResearchRetriever
# from langchain_community.utilities import GoogleSearchAPIWrapper
# from langchain_community.tools import GoogleSearchResults
# åTODO change later if above classes are not found 
# from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults


# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma


# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["MERRIAM_WEBSTER_API_KEY"] = os.getenv("MERRIAM_WEBSTER_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"

# ======= 🌟 Completely Free (No API Keys Required) =======
# 1. DuckDuckGo Search

# Initialize DuckDuckGo tool
# region="wt-wt" is a parameter that tells the search engine to return global, non-region-specific results
# time="y" parameter restricts your search results to the past year.
api_wrapper_ddg = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=2)
ddg = DuckDuckGoSearchRun(api_wrapper=api_wrapper_ddg)
print(ddg.name)

#  ======= ⚖️ Freemium (Free Tiers Available) =======
# 2. Tavily Search
tavily_tool = TavilySearch(k=2)
print(tavily_tool.name)

# 3. Google Search
# Initialize Google tool
# api_wrapper_google = GoogleSearchAPIWrapper(k=2)
# google = GoogleSearchResults(api_wrapper=api_wrapper_google)
# print(google.name)

# TODO get API key for Bing 
# 4. Bing Search
# from langchain_community.tools import BingSearchResults
# from langchain_community.utilities import BingSearchAPIWrapper

# # Initialize Bing tool
# api_wrapper_bing = BingSearchAPIWrapper(k=2)
# bing = BingSearchResults(api_wrapper=api_wrapper_bing)
# print(bing.name)


# 5. Stack Exchange
# Initialize Stack Exchange tool
api_wrapper_stack = StackExchangeAPIWrapper(max_results=2)
stack = StackExchangeTool(api_wrapper=api_wrapper_stack)
print(stack.name)

# ====💰 Paid / Advanced Hybrid Tools====

# 6. Wolfram Alpha (Computational Knowledge)
# Initialize Wolfram tool
api_wrapper_wolfram = WolframAlphaAPIWrapper()
wolfram = WolframAlphaQueryRun(api_wrapper=api_wrapper_wolfram)
print(wolfram.name)


# 7. Merriam-Webster (Dictionary & Thesaurus)
# Initialize Merriam-Webster tool
api_wrapper_mw = MerriamWebsterAPIWrapper()
mw = MerriamWebsterQueryRun(api_wrapper=api_wrapper_mw)
print(mw.name)


# 8. Web Research Retriever (Scraper + Vectorstore Hub)
# Initialize Web Research Retriever components
# It requires a search wrapper, vector database, and an LLM to handle parsing
# Initialize wrappers and embeddings
# search_wrapper = GoogleSearchAPIWrapper(k=2)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)

# # Initialize your vectorstore cleanly
# vectorstore = Chroma(embedding_function=embeddings)

# llm = ChatOpenAI(temperature=0)

# web_research = WebResearchRetriever.from_llm(
#     vectorstore=vectorstore,
#     llm=llm,
#     search=search_wrapper,
#     allow_dangerous_requests=True 
# )

# print(web_research.__class__.__name__)


# 9. PubMedLoader (Medical & Life Sciences Research)
# Initialize the PubMedLoader with a search query and a limit on maximum documents
pubmed_loader = PubMedLoader(query="Agentic RAG in healthcare", load_max_docs=2)
# Print the class name to verify it instantiates successfully
print(pubmed_loader.__class__.__name__)

# 10. RedditLoader / TwitterLoader (Social Sentiment & Public Feeds)
# For queries evaluating dynamic consumer feedback, trending tech discussions, or security alerts floating around public forums, community community-driven loaders are ideal options.
from langchain_community.document_loaders import RedditPostsLoader
# Pulls textual threads from specified subreddits or query terms

reddit_loader = RedditPostsLoader(
    client_id="YOUR_REDDIT_CLIENT_ID",          # Required
    client_secret="YOUR_REDDIT_CLIENT_SECRET",  # Required
    user_agent="Agentic-RAG-Cookbook/1.0",       # Required
    search_queries=["LangGraph", "AgenticRAG"],
    mode="search",
    number_posts=5                              
)

print(reddit_loader.__class__.__name__)

# Test 1. Using DuckDuckGoSearchRun to search
# NOTE These functions are not used in the current code but can be used for future reference or testing purposes.
# def live_web_search(query: str) -> str:
#     print("🔍 Searching the Live Web...")
#     search = DuckDuckGoSearchRun()
#     return search.run(query)



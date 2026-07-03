

import os
# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "MyAgenticRAGApp/1.0 (contact: ashoknkjp@gmail.com)"

import warnings
import logging
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_community.tools.tavily_search import TavilySearchResults
# TODO from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)


from langchain_community.tools import StackExchangeTool
from langchain_community.utilities import StackExchangeAPIWrapper

from langchain_community.tools import WolframAlphaQueryRun
from langchain_community.utilities import WolframAlphaAPIWrapper

from langchain_community.tools import MerriamWebsterQueryRun
from langchain_community.utilities import MerriamWebsterAPIWrapper

from langchain_community.retrievers import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchResults

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# NOTE: this works too but not all cases from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["MERRIAM_WEBSTER_API_KEY"] = os.getenv("MERRIAM_WEBSTER_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"


# ======= 🌟 Completely Free (No API Keys Required) =======
# 1. DuckDuckGo Search

# Initialize DuckDuckGo tool
api_wrapper_ddg = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="y", max_results=2)
ddg = DuckDuckGoSearchRun(api_wrapper=api_wrapper_ddg)
print(ddg.name)

#  ======= ⚖️ Freemium (Free Tiers Available) =======
# 3. Tavily Search
tavily = TavilySearchResults()
print(tavily.name)


# 4. Google Search
# # Initialize Google tool
# NOTE  this works too 
# from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchResults but not in all cases

# Initialize Google tool
api_wrapper_google = GoogleSearchAPIWrapper(k=2)
google = GoogleSearchResults(api_wrapper=api_wrapper_google)
print(google.name)

# TODO get API key for Bing 
# 5. Bing Search
# from langchain_community.tools import BingSearchResults
# from langchain_community.utilities import BingSearchAPIWrapper

# # Initialize Bing tool
# api_wrapper_bing = BingSearchAPIWrapper(k=2)
# bing = BingSearchResults(api_wrapper=api_wrapper_bing)
# print(bing.name)


# 6. Stack Exchange
# Initialize Stack Exchange tool
api_wrapper_stack = StackExchangeAPIWrapper(max_results=2)
stack = StackExchangeTool(api_wrapper=api_wrapper_stack)
print(stack.name)

# ====💰 Paid / Advanced Hybrid Tools====

# 7. Wolfram Alpha (Computational Knowledge)
# Initialize Wolfram tool
api_wrapper_wolfram = WolframAlphaAPIWrapper()
wolfram = WolframAlphaQueryRun(api_wrapper=api_wrapper_wolfram)
print(wolfram.name)


# 8. Merriam-Webster (Dictionary & Thesaurus)
# Initialize Merriam-Webster tool
api_wrapper_mw = MerriamWebsterAPIWrapper()
mw = MerriamWebsterQueryRun(api_wrapper=api_wrapper_mw)
print(mw.name)


# 9. Web Research Retriever (Scraper + Vectorstore Hub)
# Initialize Web Research Retriever components
# It requires a search wrapper, vector database, and an LLM to handle parsing
# Initialize wrappers and embeddings
search_wrapper = GoogleSearchAPIWrapper(k=2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)

# Initialize your vectorstore cleanly
vectorstore = Chroma(embedding_function=embeddings)

llm = ChatOpenAI(temperature=0)

# FIX: Added allow_dangerous_requests=True
web_research = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search_wrapper,
    allow_dangerous_requests=True 
)

print(web_research.__class__.__name__)


# ========NOT TESTED =======

# PubMedLoader (Medical & Life Sciences Research)
# If you want to pull scientific data similar to ArXiv but focused completely on medicine, biology, and healthcare tracking notes, use the PubMed loader.

# Setup: pip install xmltodict

from langchain_community.document_loaders import PubMedLoader

def pubmed_search(query: str) -> str:
    print("🧬 Searching PubMed...")
    loader = PubMedLoader(query, load_max_docs=2)
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs) or "No medical articles found."

# DuckDuckGoSearchRun (Real-time Live Web Search)
# When Wikipedia or ArXiv details are out of date, you can use a zero-cost, zero-configuration web search wrapper to pull snippets directly from the live internet.

# Setup: pip install duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun

def live_web_search(query: str) -> str:
    print("🔍 Searching the Live Web...")
    search = DuckDuckGoSearchRun()
    return search.run(query)

# RedditLoader / TwitterLoader (Social Sentiment & Public Feeds)
# For queries evaluating dynamic consumer feedback, trending tech discussions, or security alerts floating around public forums, community community-driven loaders are ideal options.

# Setup: pip install praw (for Reddit API)

from langchain_community.document_loaders import RedditLoader

# Pulls textual threads from specified subreddits or query terms
loader = RedditLoader(
    search_queries=["LangGraph", "AgenticRAG"],
    mode="search",
    limit=5
)        
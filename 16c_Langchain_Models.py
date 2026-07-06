import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Keys Setup
# ==============================================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]
    
question1 = "How does a microwave cook food so fast? "
question2 = "Why do mirrors reflect our image?"
question3 = "How does a touch screen on a phone work?"
append = " Please explain in 3 sentences."

# ==============================================================================
# 2. CHATOPENAI MODEL INTEGRATION
# ==============================================================================
# NOTE: The model `openai:gpt-4o-mini` does not exist or you do not have access to it
print("--- 1. Initializing OpenAI Model ---")
# Universal initialization pattern
# OPENAI_MODEL = "openai:gpt-5.4-mini"
OPENAI_MODEL = "gpt-5.4-mini"
model = init_chat_model(OPENAI_MODEL)
response = model.invoke(question1 + append)
print(f"Universal Pattern Answer:\n{response.content}\n")

# Alternative direct ChatOpenAI class invocation
model_openai = ChatOpenAI(model=OPENAI_MODEL)
response_openai = model_openai.invoke(question1 + append)
print(f"Direct Class Answer:\n{response_openai.content}\n")
print("-" * 60)

# ==============================================================================
# 3. GOOGLE GEMINI MODEL INTEGRATION
# ==============================================================================
print("--- 2. Initializing Google Gemini Model ---")
# Universal initialization pattern
INIT_GOOGLE_MODEL = "google_genai:gemini-2.5-flash"
model_gemini = init_chat_model(INIT_GOOGLE_MODEL)
response_gemini = model_gemini.invoke(question2 + append)
print(f"Universal Pattern Answer:\n{response_gemini.content}\n")

# Alternative Direct Google Generative AI class instantiation
GOOGLE_MODEL = "gemini-2.5-flash"
model_google_direct = ChatGoogleGenerativeAI(model=GOOGLE_MODEL)
response_google_direct = model_google_direct.invoke(question2 + append)
print(f"Direct Class Answer:\n{response_google_direct.content}\n")
print("-" * 60)

# ==============================================================================
# 4. GROQ MODEL INTEGRATION
# ==============================================================================
print("--- 3. Initializing Groq Model ---")
# Universal initialization pattern
INIT_GROQ_MODEL = "groq:llama-3.1-8b-instant"
model_groq = init_chat_model(INIT_GROQ_MODEL)
response_groq = model_groq.invoke(question3 + append)
print(f"Universal Pattern Answer:\n{response_groq.content}\n")

# Direct Groq mapping initialization alternative
GROQ_MODEL = "llama-3.1-8b-instant"
model_groq_direct = ChatGroq(model=GROQ_MODEL)
response_groq_direct = model_groq_direct.invoke(question3 + append)
print(f"Direct Class Answer:\n{response_groq_direct.content}\n")
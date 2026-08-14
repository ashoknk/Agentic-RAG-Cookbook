import React, { useState, useEffect } from 'react';
import { Search, Loader2, BookOpen, ChevronDown, ChevronUp, History, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

interface DocumentSource {
  page_content: string;
  metadata: {
    source?: string;
    [key: string]: any;
  };
}

interface SearchResult {
  answer: string;
  time: number;
  retrieved_docs: DocumentSource[];
}

interface HistoryItem {
  id: string;
  question: string;
  answer: string;
  time: number;
}

const API_BASE_URL = 'http://localhost:8000'; // FastAPI default port

export default function App() {
  const [isInitializing, setIsInitializing] = useState(true);
  const [initError, setInitError] = useState<string | null>(null);
  const [numChunks, setNumChunks] = useState(0);
  
  const [question, setQuestion] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [showSources, setShowSources] = useState(false);
  
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    const saved = localStorage.getItem('rag_search_history');
    return saved ? JSON.parse(saved) : [];
  });

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('rag_search_history', JSON.stringify(history));
  }, [history]);

  // Initialize RAG System on Mount
  useEffect(() => {
    initializeSystem();
  }, []);

  const initializeSystem = async () => {
    setIsInitializing(true);
    setInitError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error('Server returned an error status while initializing');
      }
      const data = await response.json();
      setNumChunks(data.num_chunks);
    } catch (err: any) {
      setInitError(err.message || 'Failed to connect to the backend server.');
    } finally {
      setIsInitializing(false);
    }
  };

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!question.trim() || isSearching) return;

    setIsSearching(true);
    setSearchError(null);
    setResult(null);
    setShowSources(false);

    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Search failed' }));
        throw new Error(errData.detail || 'Failed to fetch search results');
      }

      const data: SearchResult = await response.json();
      setResult(data);

      // Save to history (keep max 3)
      const newItem: HistoryItem = {
        id: Date.now().toString(),
        question,
        answer: data.answer,
        time: data.time
      };
      setHistory(prev => [newItem, ...prev.filter(item => item.question !== question)].slice(0, 3));
    } catch (err: any) {
      setSearchError(err.message || 'An unexpected error occurred during search.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleHistoryClick = (histQuestion: string) => {
    setQuestion(histQuestion);
    // Auto-focus search box
    const input = document.getElementById('search-input');
    if (input) {
      input.focus();
    }
  };

  const clearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans flex flex-col items-center p-4 md:p-8">
      <div className="w-full max-w-2xl flex-grow">
        
        {/* Header Section */}
        <header className="text-center mt-8 mb-10">
          <div className="inline-flex items-center justify-center p-3 bg-zinc-900 text-white rounded-2xl mb-4 shadow-sm">
            <Search className="w-8 h-8" />
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight text-zinc-900">
            RAG Document Search
          </h1>
          <p className="mt-2 text-zinc-500 font-medium">
            Ask questions about the loaded research papers and documents
          </p>
        </header>

        {/* Initialization Screen */}
        {isInitializing && (
          <div className="bg-white border border-zinc-200 rounded-2xl p-8 text-center shadow-sm flex flex-col items-center justify-center">
            <Loader2 className="w-10 h-10 animate-spin text-zinc-600 mb-4" />
            <h3 className="font-semibold text-zinc-800 text-lg">Initializing RAG System...</h3>
            <p className="text-zinc-500 text-sm mt-1 max-w-md">
              Processing research papers, indexing document chunks into a vector database, and building the LangGraph workflow. This may take 15-30 seconds.
            </p>
          </div>
        )}

        {/* Initialization Error Screen */}
        {initError && (
          <div className="bg-rose-50 border border-rose-200 rounded-2xl p-6 text-center shadow-sm flex flex-col items-center justify-center">
            <AlertCircle className="w-12 h-12 text-rose-500 mb-3" />
            <h3 className="font-semibold text-rose-800 text-lg">Initialization Failed</h3>
            <p className="text-rose-600 text-sm mt-1 max-w-md">
              {initError}
            </p>
            <p className="text-xs text-zinc-400 mt-2">
              Make sure your FastAPI server is running on <code className="bg-rose-100 px-1 py-0.5 rounded text-rose-800">{API_BASE_URL}</code>.
            </p>
            <button 
              onClick={initializeSystem}
              className="mt-4 flex items-center gap-2 bg-zinc-900 hover:bg-zinc-800 text-white text-sm font-medium px-4 py-2 rounded-lg transition"
            >
              <RefreshCw className="w-4 h-4" /> Retry Connection
            </button>
          </div>
        )}

        {/* Main Application Interface */}
        {!isInitializing && !initError && (
          <div className="space-y-6">
            
            {/* System Status / Success Toast */}
            <div className="bg-emerald-50 border border-emerald-100 rounded-xl px-4 py-3 flex items-center justify-between shadow-xs">
              <div className="flex items-center gap-2.5">
                <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                <span className="text-emerald-900 text-sm font-medium">
                  System Ready — <span className="font-semibold">{numChunks} document chunks loaded</span>
                </span>
              </div>
            </div>

            {/* Search Input Form */}
            <form onSubmit={handleSearch} className="relative group">
              <input
                id="search-input"
                type="text"
                placeholder="What would you like to know?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={isSearching}
                className="w-full bg-white border border-zinc-200 rounded-2xl pl-5 pr-32 py-4 text-zinc-900 font-medium placeholder-zinc-400 outline-none transition shadow-sm focus:border-zinc-900 focus:ring-1 focus:ring-zinc-900 disabled:opacity-70 disabled:bg-zinc-100"
              />
              <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center">
                <button
                  type="submit"
                  disabled={isSearching || !question.trim()}
                  className="bg-zinc-900 hover:bg-zinc-800 text-white font-semibold text-sm px-5 py-2.5 rounded-xl transition flex items-center gap-1.5 disabled:opacity-40 disabled:hover:bg-zinc-900"
                >
                  {isSearching ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Searching
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4" />
                      Search
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Search Error banner */}
            {searchError && (
              <div className="bg-rose-50 border border-rose-200 text-rose-800 px-4 py-3 rounded-xl flex items-center gap-2.5 text-sm">
                <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0" />
                <span>{searchError}</span>
              </div>
            )}

            {/* Skeleton / Loading indicator for searching */}
            {isSearching && (
              <div className="space-y-4 animate-pulse">
                <div className="bg-zinc-200 h-6 w-1/4 rounded-md"></div>
                <div className="bg-zinc-200 h-32 rounded-xl"></div>
              </div>
            )}

            {/* Search Results Display */}
            {result && !isSearching && (
              <div className="space-y-4">
                
                {/* Answer Card */}
                <div className="bg-zinc-900 text-white rounded-2xl p-6 shadow-md border border-zinc-800">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs tracking-wider uppercase font-bold text-zinc-400">Answer</span>
                    <span className="text-xs text-zinc-400 font-medium bg-zinc-800 px-2.5 py-1 rounded-full">
                      ⏱️ {result.time.toFixed(2)}s
                    </span>
                  </div>
                  <p className="text-zinc-100 leading-relaxed text-base whitespace-pre-wrap">
                    {result.answer}
                  </p>
                </div>

                {/* Collapsible Source Documents */}
                <div className="bg-white border border-zinc-200 rounded-2xl overflow-hidden shadow-sm">
                  <button
                    onClick={() => setShowSources(!showSources)}
                    className="w-full px-6 py-4 flex items-center justify-between bg-zinc-50 hover:bg-zinc-100/70 transition outline-none text-left"
                  >
                    <div className="flex items-center gap-2.5 text-zinc-700">
                      <BookOpen className="w-5 h-5 text-zinc-500" />
                      <span className="font-semibold text-sm">
                        Source Documents ({result.retrieved_docs.length})
                      </span>
                    </div>
                    {showSources ? (
                      <ChevronUp className="w-5 h-5 text-zinc-500" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-zinc-500" />
                    )}
                  </button>

                  {showSources && (
                    <div className="p-4 bg-zinc-50/30 border-t border-zinc-200 divide-y divide-zinc-200">
                      {result.retrieved_docs.map((doc, idx) => (
                        <div key={idx} className="py-4 first:pt-0 last:pb-0">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold text-zinc-500">
                              Document {idx + 1}
                            </span>
                            {doc.metadata?.source && (
                              <span className="text-[10px] font-mono text-zinc-400 bg-zinc-100 px-2 py-0.5 rounded max-w-[200px] truncate" title={doc.metadata.source}>
                                {doc.metadata.source.split('/').pop()}
                              </span>
                            )}
                          </div>
                          <blockquote className="text-zinc-600 text-sm leading-relaxed border-l-2 border-zinc-300 pl-3 italic whitespace-pre-line">
                            {doc.page_content.length > 300 
                              ? `${doc.page_content.substring(0, 300)}...` 
                              : doc.page_content
                            }
                          </blockquote>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            )}

            {/* History Section */}
            {history.length > 0 && (
              <div className="pt-4 border-t border-zinc-200">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2 text-zinc-500 font-semibold text-sm">
                    <History className="w-4 h-4" />
                    <span>Recent Searches</span>
                  </div>
                  <button 
                    onClick={clearHistory}
                    className="text-xs text-zinc-400 hover:text-zinc-600 transition"
                  >
                    Clear History
                  </button>
                </div>
                <div className="space-y-2.5">
                  {history.map((item) => (
                    <div 
                      key={item.id} 
                      onClick={() => handleHistoryClick(item.question)}
                      className="bg-white border border-zinc-200 hover:border-zinc-300 rounded-xl p-3.5 transition cursor-pointer shadow-2xs text-left"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <span className="text-sm font-semibold text-zinc-800 line-clamp-1 flex-grow">
                          {item.question}
                        </span>
                        <span className="text-[10px] text-zinc-400 font-mono flex-shrink-0 bg-zinc-50 px-1.5 py-0.5 rounded">
                          {item.time.toFixed(2)}s
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-1 line-clamp-1">
                        {item.answer}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}

      </div>
      
      {/* Small Minimalist Footer */}
      <footer className="w-full text-center mt-12 mb-4 py-4 border-t border-zinc-200 max-w-2xl text-[11px] text-zinc-400 font-medium">
        Powered by LangGraph, FAISS & OpenAI GPT-4o • Replacing Streamlit with React
      </footer>
    </div>
  );
}

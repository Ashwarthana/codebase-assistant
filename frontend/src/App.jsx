import React, { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [stats, setStats] = useState({ total_chunks: 0, total_files: 0, index_loaded: false });
  const [repoUrl, setRepoUrl] = useState('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState('');
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState('proposed');
  const [isAsking, setIsAsking] = useState(false);
  const [history, setHistory] = useState([]);
  const [toasts, setToasts] = useState([]);
  
  const [codePanel, setCodePanel] = useState({
    isOpen: false,
    file: '',
    start: 0,
    end: 0,
    lines: [],
    loading: false
  });

  const addToast = (message, type = 'error') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 5000);
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      addToast('Failed to fetch stats from backend.');
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const handleIndex = async () => {
    if (!repoUrl) return;
    setIsIndexing(true);
    setIndexStatus('Starting indexing job...');
    
    try {
      const res = await fetch(`${API_BASE}/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl })
      });
      
      if (!res.ok) throw new Error('Failed to start indexing');
      const { job_id } = await res.json();
      
      const poll = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/index/status/${job_id}`);
          if (statusRes.ok) {
            const { status } = await statusRes.json();
            setIndexStatus(status);
            if (status.toLowerCase().includes('done') || status.toLowerCase().includes('completed') || status.toLowerCase() === 'finished') {
              clearInterval(poll);
              setIsIndexing(false);
              setIndexStatus('Indexing complete.');
              fetchStats();
            } else if (status.toLowerCase().includes('failed') || status.toLowerCase().includes('error')) {
              clearInterval(poll);
              setIsIndexing(false);
              addToast(`Indexing failed: ${status}`);
            }
          }
        } catch (e) {
          clearInterval(poll);
          setIsIndexing(false);
          addToast('Lost connection during indexing.');
        }
      }, 1000);
      
    } catch (err) {
      addToast(err.message || 'Error starting indexing');
      setIsIndexing(false);
      setIndexStatus('');
    }
  };

  const handleAsk = async () => {
    if (!question) return;
    setIsAsking(true);
    
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode, k: 5 })
      });
      
      if (!res.ok) throw new Error('Failed to get answer');
      const data = await res.json();
      
      setHistory(prev => [{ question, ...data }, ...prev]);
      setQuestion('');
    } catch (err) {
      addToast(err.message || 'Error asking question');
    } finally {
      setIsAsking(false);
    }
  };

  const openCodePanel = async (citation) => {
    setCodePanel({ isOpen: true, file: citation.file, start: citation.start, end: citation.end, lines: [], loading: true });
    
    try {
      const fetchStart = Math.max(1, citation.start - 15);
      const fetchEnd = citation.end + 15;
      
      const res = await fetch(`${API_BASE}/file?path=${encodeURIComponent(citation.file)}&start=${fetchStart}&end=${fetchEnd}`);
      if (!res.ok) throw new Error('Failed to fetch file content');
      
      const data = await res.json();
      setCodePanel(prev => ({
        ...prev,
        lines: data.lines || [],
        fetchStart,
        loading: false
      }));
    } catch (err) {
      addToast('Error loading code context');
      setCodePanel(prev => ({ ...prev, loading: false }));
    }
  };

  const closeCodePanel = () => {
    setCodePanel(prev => ({ ...prev, isOpen: false }));
  };

  return (
    <div className="min-h-screen pb-12 flex">
      <main className="flex-1 max-w-4xl mx-auto p-6 space-y-8">
        {/* Header */}
        <header className="border-b border-gray-200 pb-6 mb-8 mt-4">
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 tracking-tight">CodeInsight</h1>
              <p className="text-gray-500 mt-1">RAG-Powered Codebase Understanding Assistant</p>
            </div>
            <div className="bg-white px-4 py-2 rounded-lg shadow-sm border border-gray-100 flex gap-4 text-sm">
              <div className="flex flex-col items-center">
                <span className="text-gray-400 font-semibold text-xs uppercase">Total Files</span>
                <span className="text-blue-600 font-bold">{stats.total_files.toLocaleString()}</span>
              </div>
              <div className="w-px bg-gray-200"></div>
              <div className="flex flex-col items-center">
                <span className="text-gray-400 font-semibold text-xs uppercase">Total Chunks</span>
                <span className="text-blue-600 font-bold">{stats.total_chunks.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </header>

        {/* Step 1 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold">1</span>
            Index Repository
          </h2>
          <div className="flex gap-3">
            <input
              type="text"
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="e.g., https://github.com/user/repo"
              value={repoUrl}
              onChange={e => setRepoUrl(e.target.value)}
              disabled={isIndexing}
            />
            <button
              onClick={handleIndex}
              disabled={isIndexing || !repoUrl}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white px-6 py-2 rounded-lg font-medium transition-colors"
            >
              {isIndexing ? 'Indexing...' : 'Index'}
            </button>
          </div>
          {isIndexing && (
            <div className="mt-4">
              <div className="text-sm text-gray-600 mb-2">{indexStatus || 'Processing...'}</div>
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div className="bg-blue-500 h-2 rounded-full w-full animate-pulse"></div>
              </div>
            </div>
          )}
        </section>

        {/* Step 2 */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold">2</span>
            Ask Questions
          </h2>
          <div className="space-y-4">
            <textarea
              className="w-full border border-gray-300 rounded-lg p-4 h-32 focus:ring-2 focus:ring-blue-500 outline-none resize-none"
              placeholder="Ask a question about the indexed codebase..."
              value={question}
              onChange={e => setQuestion(e.target.value)}
              disabled={isAsking}
            />
            
            <div className="flex justify-between items-center">
              <div className="flex bg-gray-100 p-1 rounded-lg">
                <button
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${mode === 'proposed' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setMode('proposed')}
                >
                  Proposed (Strict)
                </button>
                <button
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${mode === 'baseline' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setMode('baseline')}
                >
                  Baseline (Naive)
                </button>
              </div>
              
              <button
                onClick={handleAsk}
                disabled={isAsking || !question}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white px-8 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                {isAsking ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Thinking...
                  </>
                ) : 'Ask'}
              </button>
            </div>
          </div>
        </section>

        {/* History */}
        {history.length > 0 && (
          <div className="space-y-6">
            <h3 className="text-xl font-bold text-gray-800 border-b pb-2">Conversation History</h3>
            {history.map((item, idx) => (
              <ResponseCard key={idx} data={item} onCitationClick={openCodePanel} />
            ))}
          </div>
        )}
      </main>

      {/* Code Viewer Panel */}
      {codePanel.isOpen && (
        <div className="fixed inset-y-0 right-0 w-[600px] bg-white shadow-2xl border-l border-gray-200 flex flex-col slide-in z-40">
          <div className="flex justify-between items-center p-4 border-b border-gray-200 bg-gray-50">
            <h3 className="font-mono text-sm text-gray-700 truncate mr-4">{codePanel.file}</h3>
            <button onClick={closeCodePanel} className="text-gray-400 hover:text-gray-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
          <div className="flex-1 overflow-auto bg-[#1e1e1e] text-gray-300 p-4 relative">
            {codePanel.loading ? (
              <div className="flex items-center justify-center h-full text-gray-400">Loading context...</div>
            ) : (
              <div className="code-line relative">
                {codePanel.lines.map((line, idx) => {
                  const lineNum = codePanel.fetchStart + idx;
                  const isHighlighted = lineNum >= codePanel.start && lineNum <= codePanel.end;
                  return (
                    <div key={lineNum} className={`flex ${isHighlighted ? 'bg-[#3e3e24] text-gray-100' : ''}`}>
                      <span className="w-12 flex-shrink-0 text-right pr-4 text-gray-600 select-none">{lineNum}</span>
                      <span className="whitespace-pre flex-1">{line || ' '}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Toasts */}
      <div className="fixed bottom-4 right-4 z-50 space-y-2">
        {toasts.map(t => (
          <div key={t.id} className="bg-gray-900 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 animate-slideIn">
            <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
            <span className="text-sm font-medium">{t.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResponseCard({ data, onCitationClick }) {
  const isSupported = data.status === 'SUPPORTED';
  
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
      <div className="bg-gray-50 p-4 border-b border-gray-100">
        <p className="font-medium text-gray-900">Q: {data.question}</p>
      </div>
      
      <div className="p-6">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          <span className={`px-3 py-1 text-xs font-bold uppercase rounded-full ${isSupported ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
            {data.status}
          </span>
          
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span className="font-medium">Confidence:</span>
            <div className="w-32 bg-gray-200 rounded-full h-2">
              <div className={`h-2 rounded-full ${isSupported ? 'bg-green-500' : 'bg-amber-500'}`} style={{ width: `${Math.round(data.confidence * 100)}%` }}></div>
            </div>
            <span>{Math.round(data.confidence * 100)}%</span>
          </div>
          
          {data.overlap_score !== undefined && (
            <div className="text-sm text-gray-600 border-l pl-4 border-gray-200">
              <span className="font-medium">Overlap Score:</span> {data.overlap_score.toFixed(2)}
            </div>
          )}
          
          <div className="text-sm text-gray-500 border-l pl-4 border-gray-200">
            Mode: <span className="font-medium capitalize">{data.mode}</span>
          </div>
        </div>

        <div className="prose prose-sm max-w-none text-gray-800 mb-6">
          {data.answer.split('\n').map((line, i) => <p key={i} className="mb-2">{line}</p>)}
        </div>

        {data.citations && data.citations.length > 0 && (
          <div className="mb-6 border-t pt-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              Citations
              {data.citation_valid === false && (
                <span className="bg-red-100 text-red-700 text-xs px-2 py-0.5 rounded flex items-center gap-1">
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                  Invalid
                </span>
              )}
            </h4>
            <div className="flex flex-wrap gap-2">
              {data.citations.map((c, i) => (
                <button
                  key={i}
                  onClick={() => onCitationClick(c)}
                  className="inline-flex items-center gap-1.5 px-3 py-1 bg-gray-100 hover:bg-gray-200 border border-gray-200 rounded-md text-xs font-mono text-gray-700 transition-colors"
                >
                  <svg className="w-3.5 h-3.5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
                  {c.file.split('/').pop()} ({c.start}-{c.end})
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-2">
          {data.retrieved_files && data.retrieved_files.length > 0 && (
            <details className="group">
              <summary className="text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900 select-none flex items-center">
                <svg className="w-4 h-4 mr-1 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                Retrieved Files ({data.retrieved_files.length})
              </summary>
              <div className="mt-2 pl-5">
                <ul className="list-disc text-xs text-gray-500 space-y-1 font-mono">
                  {data.retrieved_files.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            </details>
          )}

          {data.evidence && data.evidence.length > 0 && (
            <details className="group">
              <summary className="text-sm font-medium text-gray-600 cursor-pointer hover:text-gray-900 select-none flex items-center">
                <svg className="w-4 h-4 mr-1 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" /></svg>
                Retrieved Evidence Chunks ({data.evidence.length})
              </summary>
              <div className="mt-2 space-y-2 pl-5">
                {data.evidence.map((ev, i) => (
                  <div key={i} className="bg-gray-50 border border-gray-100 rounded p-2 text-xs">
                    <div className="font-mono text-gray-700 mb-1 font-semibold">{ev.name} <span className="text-gray-400 font-normal">({ev.type})</span> - {ev.file}:{ev.start_line}</div>
                    <pre className="text-[11px] text-gray-600 overflow-x-auto p-2 bg-gray-100 rounded">{ev.code}</pre>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';

export default function App() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.5);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/experiments/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query, 
          top_k: parseInt(topK), 
          alpha: parseFloat(alpha) 
        })
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const result = await res.json();
      setData(result.comparisons);
    } catch (err) {
      console.error("Experiment failed:", err);
      setError("Failed to fetch comparison data. Make sure the backend server is running on localhost:8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '30px', fontFamily: 'sans-serif', maxWidth: '1200px', margin: '0 auto', color: '#fff' }}>
      <h2>Week 4 — Retrieval Experiments Dashboard</h2>
      
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
        <input 
          type="text" 
          placeholder="Enter search question..." 
          value={query} 
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: '350px', padding: '10px', borderRadius: '4px', border: '1px solid #444', background: '#222', color: '#fff' }}
        />
        <label>Top K: 
          <input 
            type="number" 
            value={topK} 
            onChange={(e) => setTopK(e.target.value)} 
            style={{ width: '60px', marginLeft: '5px', padding: '8px', borderRadius: '4px', border: '1px solid #444', background: '#222', color: '#fff' }} 
          />
        </label>
        <label style={{ marginLeft: '10px' }}>Alpha: 
          <input 
            type="number" 
            step="0.1" 
            min="0" 
            max="1" 
            value={alpha} 
            onChange={(e) => setAlpha(e.target.value)} 
            style={{ width: '60px', marginLeft: '5px', padding: '8px', borderRadius: '4px', border: '1px solid #444', background: '#222', color: '#fff' }} 
          />
        </label>
        <button 
          onClick={handleCompare} 
          disabled={loading}
          style={{ padding: '10px 20px', borderRadius: '4px', background: '#646cff', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}
        >
          {loading ? 'Running...' : 'Compare Modes'}
        </button>
      </div>

      {error && <p style={{ color: '#ff6b6b' }}>{error}</p>}

      {data && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
          {['dense', 'bm25', 'hybrid'].map((mode) => (
            <div key={mode} style={{ border: '1px solid #333', borderRadius: '8px', padding: '15px', background: '#1a1a1a' }}>
              <h3 style={{ textTransform: 'uppercase', margin: '0 0 10px 0', color: '#646cff' }}>{mode}</h3>
              <p><strong>Latency:</strong> {data[mode]?.execution_time_ms} ms</p>
              <hr style={{ borderColor: '#333' }} />
              {data[mode]?.results.map((doc, idx) => (
                <div key={idx} style={{ background: '#242424', padding: '10px', marginBottom: '10px', borderRadius: '4px', borderLeft: '3px solid #646cff' }}>
                  <small style={{ color: '#aaa' }}>Score: {doc.score} | Page {doc.page_number || 'N/A'}</small>
                  <p style={{ fontSize: '13px', margin: '8px 0 0 0', color: '#ddd' }}>{doc.content}</p>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
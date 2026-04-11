import { useEffect, useState } from "react"
import "./App.css"

function App() {
  const [scenario, setScenario] = useState("")
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [history, setHistory] = useState([])

  const [healthStatus, setHealthStatus] = useState("checking")
  const [healthMessage, setHealthMessage] = useState("")
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState("")

  useEffect(() => {
    checkHealth()
    fetchDocuments()
  }, [])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/health`)
      const data = await res.json()

      if (res.ok && data.status === "ok") {
        setHealthStatus("ok")
        setHealthMessage(`Backend connected (v${data.version})`)
      } else {
        setHealthStatus("degraded")
        setHealthMessage(data.reason || "Backend degraded")
      }
    } catch {
      setHealthStatus("offline")
      setHealthMessage("Backend unavailable")
    }
  }

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/documents`)
      const data = await res.json()

      if (!res.ok) {
        setDocumentsError("Could not load documents.")
        return
      }

      setDocuments(data.documents || [])
    } catch {
      setDocumentsError("Cannot load documents while backend is offline.")
    }
  }

  const handleSubmit = async () => {
    setError("")
    setResponse(null)

    if (scenario.trim().length < 10) {
      setError("Scenario must be at least 10 characters long.")
      return
    }

    setLoading(true)

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/v1/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          scenario,
          user_type: "trainee"
        })
      })

      const data = await res.json()

      if (!res.ok) {
        setError(data.message || "Something went wrong.")
      } else {
        setResponse(data)

        const newHistoryItem = {
          scenario,
          answer: data.answer
        }

        setHistory((prev) => [newHistoryItem, ...prev].slice(0, 5))
      }
    } catch {
      setError("Cannot connect to server. Is the backend running?")
    }

    setLoading(false)
  }

  return (
    <div className="page">
      <div className="main-layout">
        <div className="app">
          <h1>Psychology Trainee Assistant</h1>

          <label className="input-label">Describe your clinical scenario</label>

          <textarea
            placeholder="Describe your clinical scenario..."
            rows="6"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            disabled={loading}
          />

          <button onClick={handleSubmit} disabled={loading || !scenario.trim()}>
            {loading ? "Loading..." : "Get guidance"}
          </button>

          {error && <div className="error-box">{error}</div>}

          {response && (
            <div className="response-box">
              {!response.found_in_documents && (
                <div className="warning-box">
                  No relevant guidance found in the available documents. Please consult a supervisor.
                </div>
              )}

              <h3>Answer</h3>
              <p>{response.answer}</p>

              <p className="processing-time">
                Answer generated in {(response.processing_time_ms / 1000).toFixed(1)}s
              </p>

              <details className="sources-box">
                <summary>Sources</summary>
                {response.sources && response.sources.length > 0 ? (
                  <ul>
                    {response.sources.map((source, index) => (
                      <li key={index}>
                        <strong>{source.document}</strong><br />
                        Section: {source.section}<br />
                        Page: {source.page}<br />
                        Relevance: {(source.relevance_score * 100).toFixed(0)}%
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p>No sources available.</p>
                )}
              </details>
            </div>
          )}

          {history.length > 0 && (
            <div className="history-box">
              <h3>Last 5 Queries</h3>
              {history.map((item, index) => (
                <div key={index} className="history-item">
                  <p><strong>Scenario:</strong> {item.scenario}</p>
                  <p><strong>Answer:</strong> {item.answer}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <aside className="sidebar">
          <h2>Knowledge Base</h2>

          <div className="status-card">
            <div className="status-row">
              <span
                className={`status-dot ${
                  healthStatus === "ok"
                    ? "green"
                    : healthStatus === "degraded"
                    ? "red"
                    : "gray"
                }`}
              ></span>
              <span className="status-text">
                {healthStatus === "ok"
                  ? "Connected"
                  : healthStatus === "degraded"
                  ? "Degraded"
                  : "Offline"}
              </span>
            </div>
            <p className="status-message">{healthMessage}</p>
          </div>

          <div className="documents-card">
            <h3>Indexed Documents</h3>

            {documentsError && <p className="documents-error">{documentsError}</p>}

            {!documentsError && documents.length === 0 && (
              <p className="documents-empty">No documents loaded yet.</p>
            )}

            {documents.length > 0 && (
              <ul className="documents-list">
                {documents.map((doc, index) => (
                  <li key={index} className="document-item">
                    <strong>{doc.name}</strong>
                    <p>Sections: {doc.sections}</p>
                    <p>Chunks: {doc.chunks}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}

export default App
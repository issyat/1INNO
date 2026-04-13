import { useState } from "react"
import "./App.css"

function App() {
  const [scenario, setScenario] = useState("")
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [history, setHistory] = useState([])

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
      <div className="background-glow background-glow-1"></div>
      <div className="background-glow background-glow-2"></div>

      <div className="app-shell">
        <div className="hero-badge">Clinical Support Assistant</div>

        <h1>Psychology Trainee Assistant</h1>

        <p className="hero-subtitle">
          Get grounded guidance for difficult clinical scenarios using trusted,
          cited documents. Describe the situation clearly to receive structured support.
        </p>

        <div className="input-card">
          <label className="input-label">Describe your clinical scenario</label>

          <textarea
            placeholder="Example: My patient disclosed suicidal thoughts during the session and says they do not feel safe going home..."
            rows="7"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            disabled={loading}
          />

          <div className="action-row">
            <button onClick={handleSubmit} disabled={loading || !scenario.trim()}>
              {loading ? "Loading..." : "Get guidance"}
            </button>

            <span className="helper-text">
              Responses are based only on indexed source documents.
            </span>
          </div>
        </div>

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

            {response.sources && response.sources.length > 0 && (
              <div className="sources-box">
                <h4>Sources</h4>

                {response.sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <div className="source-header">
                      📄 {source.document}
                    </div>

                    <div className="source-details">
                      <span>Section: {source.section}</span>
                      <span>Page: {source.page}</span>
                      <span>
                        Relevance: {(source.relevance_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            <p className="processing-time">
              Answer generated in {(response.processing_time_ms / 1000).toFixed(1)}s
            </p>
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
    </div>
  )
}

export default App
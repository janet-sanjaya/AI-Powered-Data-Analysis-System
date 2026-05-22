import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [preview, setPreview] = useState([]);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [status, setStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [model, setModel] = useState("openai/gpt-4o-mini");

  const handleUpload = async () => {
    if (!file) {
      setStatus("Please select a CSV file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoadingUpload(true);
    setStatus("Uploading and cleaning file...");

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const text = await response.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error(text);
      }

      if (!response.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      setFileId(data.file_id);
      setPreview(data.preview || []);
      setStatus("File cleaned successfully.");
      setAnalysis(null);
    } catch (error) {
      console.error(error);
      setStatus(`Error: ${error.message}`);
    } finally {
      setLoadingUpload(false);
    }
  };

  const handleAnalyze = async () => {
    if (!fileId) {
      setStatus("Please upload a CSV first.");
      return;
    }

    if (!question.trim()) {
      setStatus("Please type a question first.");
      return;
    }

    setLoadingAnalyze(true);
    setStatus("Generating analysis...");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_id: fileId,
          question: question,
          model: model,
        }),
      });

      const text = await response.text();
      
      let data;
      try {
        data = JSON.parse(text);
      } catch (e) {
        throw new Error(text);
      }

      if (!response.ok) {
        throw new Error(data.detail || "Analysis failed");
      }

      setAnalysis(data);
      setStatus("Analysis complete.");
    } catch (error) {
      console.error(error);
      setStatus(`Error: ${error.message}`);
      setAnalysis(null);
    } finally {
      setLoadingAnalyze(false);
    }
  };

  const cardStyle = {
    background: "rgba(17, 24, 39, 0.92)",
    border: "1px solid rgba(148, 163, 184, 0.18)",
    borderRadius: "20px",
    padding: "22px",
    marginBottom: "22px",
    boxShadow: "0 10px 30px rgba(0,0,0,0.18)",
  };

  const titleStyle = {
    fontSize: "44px",
    fontWeight: 800,
    textAlign: "center",
    margin: "0 0 8px 0",
    marginBottom: "30px",
    letterSpacing: "-0.02em",
  };

  const subtitleStyle = {
    textAlign: "center",
    color: "#cbd5e1",
    marginBottom: "28px",
    fontSize: "16px",
  };

  const labelStyle = {
    display: "block",
    marginBottom: "10px",
    fontSize: "18px",
    fontWeight: 700,
    color: "#f8fafc",
  };

  const inputStyle = {
    width: "100%",
    padding: "13px 14px",
    borderRadius: "12px",
    border: "1px solid #334155",
    background: "#0f172a",
    color: "white",
    outline: "none",
    fontSize: "15px",
    boxSizing: "border-box",
  };

  const buttonBase = {
    border: "none",
    borderRadius: "12px",
    cursor: "pointer",
    fontWeight: 700,
    padding: "12px 18px",
    transition: "transform 0.12s ease, opacity 0.12s ease",
  };

  const primaryButton = {
    ...buttonBase,
    background: loadingUpload ? "#475569" : "#2563eb",
    color: "white",
  };

  const analyzeButton = {
    ...buttonBase,
    background: loadingAnalyze ? "#64748b" : "#10b981",
    color: "white",
    width: "100%",
  };

  const sectionTitleStyle = {
    fontSize: "22px",
    fontWeight: 800,
    margin: "0 0 14px 0",
    color: "#f8fafc",
  };

  const tableWrapper = {
    overflowX: "auto",
    borderRadius: "14px",
    border: "1px solid #334155",
  };

  const tableStyle = {
    width: "100%",
    borderCollapse: "collapse",
    minWidth: "720px",
  };

  const thStyle = {
    background: "#1f2937",
    color: "#f8fafc",
    textAlign: "left",
    padding: "10px 12px",
    borderBottom: "1px solid #334155",
    fontSize: "14px",
    position: "sticky",
    top: 0,
  };

  const tdStyle = {
    padding: "10px 12px",
    borderBottom: "1px solid #334155",
    color: "#e5e7eb",
    fontSize: "14px",
    verticalAlign: "top",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(circle at top, rgba(59,130,246,0.12), transparent 35%), linear-gradient(180deg, #0f172a 0%, #0b1220 100%)",
        color: "white",
        padding: "36px 22px 48px",
        fontFamily:
          'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <div style={{ width: "100%", padding: "0 40px" }}>
        <header style={{ marginBottom: "30px" }}>
          <h1 style={titleStyle}>AInsight</h1>
          <p style={subtitleStyle}>
            Upload a CSV file, ask a business question, and get analysis with recommendations.
          </p>
        </header>

        <div style={cardStyle}>
          <div
            style={{
              display: "flex",
              gap: "14px",
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              style={{
                color: "#cbd5e1",
                maxWidth: "100%",
              }}
            />
            <button onClick={handleUpload} disabled={loadingUpload} style={primaryButton}>
              {loadingUpload ? "Uploading..." : "Upload and Clean"}
            </button>
            {status && (
              <span
                style={{
                  color: status.toLowerCase().includes("error") ? "#fca5a5" : "#fbbf24",
                  fontWeight: 600,
                }}
              >
                {status}
              </span>
            )}
          </div>
          {fileId && (
            <p style={{ marginTop: "12px", color: "#94a3b8", fontSize: "13px" }}>
              Dataset loaded successfully.
            </p>
          )}
        </div>

        {preview.length > 0 && (
          <div style={cardStyle}>
            <h2 style={sectionTitleStyle}>Data Preview</h2>
            <div style={tableWrapper}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    {Object.keys(preview[0]).map((key) => (
                      <th key={key} style={thStyle}>
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? "#111827" : "#0f172a" }}>
                      {Object.values(row).map((val, j) => (
                        <td key={j} style={tdStyle}>
                          {String(val)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div style={cardStyle}>
          <label style={labelStyle}>Select Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              ...inputStyle,
              cursor: "pointer",
            }}
          >
            <option value="openai/gpt-4o-mini">OpenAI GPT-4o mini</option>
            <option value="anthropic/claude-4.5-sonnet">Anthropic Claude 4.5 Sonnet</option>
            <option value="google/gemini-2.5-flash">Google Gemini 2.5 Flash</option>
            <option value="meta-llama/llama-3.3-70b-instruct">Meta Llama 3.3 70B</option>
          </select>
        </div>

        <div style={cardStyle}>
          <label style={labelStyle}>Ask a Question</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Example: Which branch caused the biggest drop in September?"
            style={inputStyle}
          />
          <button
            onClick={handleAnalyze}
            disabled={loadingAnalyze}
            style={{ ...analyzeButton, marginTop: "14px" }}
          >
            {loadingAnalyze ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        {analysis && (
          <div style={cardStyle}>
            <h2 style={sectionTitleStyle}>Generated Code</h2>
            <pre
              style={{
                whiteSpace: "pre-wrap",
                background: "#0b1220",
                padding: "16px",
                borderRadius: "14px",
                border: "1px solid #334155",
                overflowX: "auto",
                color: "#e2e8f0",
                margin: "0 0 24px 0",
                lineHeight: 1.6,
              }}
            >
              {analysis.generated_code}
            </pre>

            <h2 style={sectionTitleStyle}>Answer</h2>
            {analysis.result_table && analysis.result_table.length > 0 ? (
              <div style={{ marginBottom: "24px" }}>
                <div style={tableWrapper}>
                  <table style={tableStyle}>
                    <thead>
                      <tr>
                        {Object.keys(analysis.result_table[0]).map((key) => (
                          <th key={key} style={thStyle}>
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.result_table.map((row, i) => (
                        <tr key={i} style={{ background: i % 2 === 0 ? "#111827" : "#0f172a" }}>
                          {Object.values(row).map((val, j) => (
                            <td key={j} style={tdStyle}>
                              {String(val)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p style={{ color: "#cbd5e1", marginBottom: "24px" }}>
                {analysis.answer_preview || "No answer available."}
              </p>
            )}

            <h2 style={sectionTitleStyle}>Analysis</h2>
            <p style={{ lineHeight: "1.8", color: "#e5e7eb", marginBottom: "24px" }}>
              {analysis.analysis || analysis.answer_preview || "No analysis available."}
            </p>

            <h2 style={sectionTitleStyle}>Recommendation</h2>
            <p style={{ lineHeight: "1.8", color: "#e5e7eb", marginBottom: 0 }}>
              {analysis.recommendation || "No recommendation available."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
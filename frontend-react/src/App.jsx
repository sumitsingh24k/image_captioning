import { useMemo, useState } from "react";

export default function App() {
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:8000/caption");
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [caption, setCaption] = useState("Your generated caption will appear here.");
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  const onFileChange = (event) => {
    const selected = event.target.files?.[0];
    setFile(selected || null);
    setError("");
  };

  const generateCaption = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    setError("");
    setCaption("Generating...");

    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(apiUrl.trim(), {
        method: "POST",
        body: form
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate caption.");
      }
      setCaption(data.caption || "No caption generated.");
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setCaption("Caption generation failed.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="blob blob-a" />
      <div className="blob blob-b" />
      <main className="card">
        <header>
          <p className="eyebrow">Vision + Language</p>
          <h1>VisionCaption</h1>
          <p className="subtitle">Turn any image into a natural language description.</p>
        </header>

        <section className="grid">
          <label className="field">
            <span>Backend API URL</span>
            <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} placeholder="http://127.0.0.1:8000/caption" />
          </label>
        </section>

        <section className="upload">
          <label htmlFor="imageInput" className="uploadBox">
            <input id="imageInput" type="file" accept="image/*" onChange={onFileChange} />
            <div>
              <strong>Drop an image here</strong>
              <p>or click to browse files</p>
            </div>
          </label>
          {previewUrl && <img className="preview" src={previewUrl} alt="Preview" />}
        </section>

        <button className="generateBtn" onClick={generateCaption} disabled={isLoading}>
          {isLoading ? "Generating..." : "Generate Caption"}
        </button>

        {error && <p className="error">{error}</p>}

        <section className="result">
          <p className="resultLabel">Generated Caption</p>
          <p className="resultText">{caption}</p>
        </section>
      </main>
    </div>
  );
}

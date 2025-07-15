import React, { useState } from "react";

function InputForm({ onPredict }) {  
  const [text, setText] = useState("");
  const [error, setError] = useState("");  

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) {  
      setError("⚠️ Please enter some text.");
      return;
    }
    setError("");  
    onPredict(text);  
  };

  return (
    <div className="input-container">
      <form onSubmit={handleSubmit}>
        <textarea 
          value={text} 
          onChange={(e) => setText(e.target.value)} 
          placeholder="Enter text here..."
          className="input-textarea"
        />
        <button type="submit" className="analyze-button">Analyze</button>
      </form>
      {error && <p className="error-message">{error}</p>}
    </div>
  );
}

export default InputForm;

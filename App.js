import React, { useState } from "react";
import axios from "axios";
import InputForm from "./components/InputForm";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState(null);

  const handlePrediction = async (text) => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", { text });
      console.log("API Response:", response.data); // Debugging log

      if (response.data && response.data.prediction) {
        setPrediction(response.data.prediction);

        // Ensure confidence is always displayed as a percentage
        if (response.data.confidence !== undefined) {
          setConfidence((response.data.confidence * 100).toFixed(2) + "%"); 
        } else {
          setConfidence("N/A");
        }
        
        setError(null);
      } else {
        setError("Invalid response from server.");
      }
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setError("Failed to get response from API.");
      setPrediction(null);
      setConfidence(null);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Hate Speech & Cyberbullying Detector</h1>
      <InputForm onPredict={handlePrediction} />
      
      <div style={{ marginTop: "20px" }}>
        {error && <p style={{ color: "red" }}>⚠️ {error}</p>}
        {prediction !== null && (
          <div style={{ marginTop: "10px", padding: "10px", border: "1px solid #ccc", borderRadius: "8px", display: "inline-block" }}>
            <h2>📝 Prediction: <span style={{ color: "#007bff" }}>{prediction}</span></h2>
            <h3>📊 Confidence: <span style={{ color: "#28a745" }}>{confidence}</span></h3>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

import React from "react";

function ResultDisplay({ result }) {
  return (
    <div>
      {result ? (
        <div>
          <h2>Prediction:</h2>
          <p>{result.prediction}</p>
          <h3>Confidence:</h3>
          <p>{result.confidence}</p>
        </div>
      ) : (
        <p>Enter text and click analyze to see results.</p>
      )}
    </div>
  );
}

export default ResultDisplay;

import React, { useState, useEffect, useRef } from 'react';
import './QnAGenerator.css';
import * as THREE from 'three';
import NET from 'vanta/dist/vanta.net.min';

const QnAGenerator = () => {
  const vantaRef = useRef(null);
  const [file, setFile] = useState(null);
  const [qnaPairs, setQnaPairs] = useState([]);
  const [error, setError] = useState("");

  // Vanta background effect setup
  useEffect(() => {
    const vantaEffect = NET({
      el: vantaRef.current,
      mouseControls: true,
      touchControls: true,
      gyroControls: false,
      minHeight: 200.00,
      minWidth: 200.00,
      scale: 1.00,
      scaleMobile: 1.00
    });

    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, []);

  // Handle file change when a user selects a file
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  // Handle file upload and sending data to the server
  const handleFileUpload = async () => {
    const formData = new FormData();

    if (file) {
      formData.append('file', file);  // Append the selected file if available
    }

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        // Combine questions and answers into a usable format
        const combinedQnA = data.questions.map((question, index) => ({
          question,
          answer: data.answers[index].answer,
        }));
        setQnaPairs(combinedQnA);
        setError("");
      } else {
        setError(data.error);
      }
    } catch (error) {
      setError("Error uploading file");
      console.error("Error uploading file:", error); // Log error to console
    }
  };

  return (
    <div className="qna-container">
      <div ref={vantaRef} className="vanta-background"></div>
      <div className="qna-content">
        <h1>QnA Generator</h1>
        <p>Upload a PDF file to generate questions and answers.</p>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleFileUpload}>Generate QnA</button>
        {error && <p className="error">{error}</p>}
        {qnaPairs.length > 0 && (
          <div className="qna-list">
            {qnaPairs.map((qna, index) => (
              <div key={index} className="qna-pair">
                <p className="question"><strong>Q:</strong> {qna.question}</p>
                <p className="answer"><strong>A:</strong> {qna.answer}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default QnAGenerator;

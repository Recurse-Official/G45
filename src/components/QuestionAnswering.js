import React, { useEffect, useRef, useState } from 'react';
import './QuestionAnswering.css';
import * as THREE from 'three';
import HALO from 'vanta/dist/vanta.halo.min';

const QuestionAnswering = () => {
  const vantaRef = useRef(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let vantaEffect;
    if (!vantaEffect) {
      vantaEffect = HALO({
        el: vantaRef.current,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.0,
        minWidth: 200.0,
        scale: 1.0,
        scaleMobile: 1.0,
        THREE,
      });
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy();
    };
  }, []);

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleDateChange = (event) => {
    setSelectedDate(event.target.value);
  };

  const handleFormSubmit = async (event) => {
    event.preventDefault();

    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    setError('');
    setAnswer('');
    setDates([]);

    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, date: selectedDate }),
      });
      const data = await response.json();

      if (response.ok) {
        if (data.dates) {
          setDates(data.dates);
        } else {
          setAnswer(data.answer);
        }
      } else {
        setError(data.error || 'Error querying the document');
      }
    } catch (error) {
      setError('Error communicating with backend');
      console.error(error);
    }
  };

  return (
    <div className="qna-container" ref={vantaRef}>
      <div className="qna-content">
        <h1>QnA System</h1>
        <p>Enter a question, and get an answer based on the trained context.</p>
        <form onSubmit={handleFormSubmit}>
          <div className="input-container">
            <input
              type="text"
              value={question}
              onChange={handleQuestionChange}
              placeholder="Enter your question"
              required
            />
          </div>
          {dates.length > 0 && (
            <div className="date-selector">
              <label>Select a date:</label>
              <select value={selectedDate} onChange={handleDateChange}>
                <option value="">Most Recent</option>
                {dates.map((date, index) => (
                  <option key={index} value={date}>
                    {date}
                  </option>
                ))}
              </select>
            </div>
          )}
          <button type="submit">Get Answer</button>
        </form>
        {error && <p className="error">{error}</p>}
        {answer && (
          <div className="qna-answer">
            <p>
              <strong>Answer:</strong> {answer}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuestionAnswering;

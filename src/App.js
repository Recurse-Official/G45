import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import QnAGenerator from './components/QnAGenerator';
import QuestionAnswering from './components/QuestionAnswering';
import './App.css';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/qandagenerator" element={<QnAGenerator />} />
        <Route path="/questionanswering" element={<QuestionAnswering />} />
      </Routes>
    </Router>
  );
}

export default App;

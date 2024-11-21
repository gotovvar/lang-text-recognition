import React, { useState } from 'react';
import './App.css';

const App = () => {
  const [selectedMethod, setSelectedMethod] = useState(null);
  const [files, setFiles] = useState(null);
  const [results, setResults] = useState(null);
  const [fileUrls, setFileUrls] = useState({});

  const handleMethodSelect = (method) => {
    setSelectedMethod(method);
  };

  const handleFileChange = (event) => {
    const uploadedFiles = event.target.files;
    setFiles(uploadedFiles);
    
    const newFileUrls = {};
    Array.from(uploadedFiles).forEach((file) => {
      const url = URL.createObjectURL(file);
      newFileUrls[file.name] = url;
    });
    setFileUrls(newFileUrls);
  };

  const handleSubmit = async () => {
    if (!selectedMethod || !files) {
      alert('Please select a method and upload files.');
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('method', selectedMethod);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/v0/upload-html/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Error while uploading files');
      }

      const result = await response.json();
      setResults(result.results);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const saveResultsToFile = () => {
    if (!results) {
      alert('No results to save.');
      return;
    }
  
    let resultsText = '';
    results.forEach((result) => {
      const fileName = result.filename;
      const language = result.language;
      const classicSummary = result.classic_summary;
      const keywordsSummary = result.keywords_summary;
      const neuralSummary = result.neural_summary;
  
      resultsText += `${fileName} - ${language}\n`;
      resultsText += `Классический реферат - ${classicSummary}\n`;
      resultsText += `Реферат в виде ключевых слов - ${keywordsSummary}\n\n`;
      resultsText += `Реферат при помощи нейронных сетей - ${neuralSummary}\n\n`;
    });
  
    const blob = new Blob([resultsText], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'language_detection_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <h1>Language Detection</h1>
      <p>Select a method and upload files to detect the language</p>

      <div className="button-group">
        <button onClick={() => handleMethodSelect('ngram')} className="method-button">
          N-Gram
        </button>
        <button onClick={() => handleMethodSelect('alphabet')} className="method-button">
          Alphabet
        </button>
        <button onClick={() => handleMethodSelect('neural')} className="method-button">
          Neural
        </button>
      </div>

      <div>
        <input type="file" multiple onChange={handleFileChange} className="file-input" />
      </div>

      <div>
        <button onClick={handleSubmit} className="submit-button">Submit</button>
      </div>

      {results && (
        <div className="result-area">
          <h3>Results:</h3>
          <ul>
            {results.map((result, index) => {
              const fileName = result.filename;
              const language = result.language;
              const classicSummary = result.classic_summary;
              const keywordsSummary = result.keywords_summary;
              const neuralSummary = result.neural_summary;
              const fileUrl = fileUrls[fileName];

              return (
                <li key={index}>
                  <a href={fileUrl} target="_blank" rel="noopener noreferrer">
                    {fileName}
                  </a> - {language}<br />
                  Классический реферат - {classicSummary}<br />
                  Реферат в виде ключевых слов - {keywordsSummary}<br />
                  Реферат при помощи нейронных сетей - {neuralSummary}<br />
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {results && (
        <div>
          <button onClick={saveResultsToFile} className="save-button">Save Results</button>
        </div>
      )}
    </div>
  );
};

export default App;

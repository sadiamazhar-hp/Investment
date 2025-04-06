import React, { useEffect, useState } from "react";
import axios from "axios";
import './App.css';

function App() {
  const [message, setMessage] = useState("");
  const [quote, setQuote] = useState("");
    useEffect(() => {
      axios.get("http://127.0.0.1:5000/api/data")
        .then(response => setMessage(response.data.message))
        .catch(error => console.error(error));
    }, []);
  
    const sendData = () => {
      axios.post("http://127.0.0.1:5000/api/insert", { name: "Sadia" })
        .then(response => console.log(response.data))
        .catch(error => console.error(error));
    };
    const sendDataQuote = () => {
      console.log("RUNNED");
      axios.post("http://127.0.0.1:5000/api/predict", { quote }, { timeout: 60000 })  // 60s timeout
        .then(response => setMessage(JSON.stringify(response.data, null, 2)))
        .catch(error => console.error("Error:", error));

    };
    return (
      <div>
        <h1>STOCK PREDICTIONS</h1>
        <input 
        type="text" 
        placeholder="Enter Stock Ticker"
        value={quote}
        onChange={(e) => setQuote(e.target.value)}
      />
      <button onClick={sendDataQuote}>Get Prediction</button>
      </div>
    );
  }
  

export default App;

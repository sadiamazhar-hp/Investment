import "./Home.css";
import React, { useState } from "react";
import axios from "axios";

function Home({ setPrediction }) {
    const [stockSymbol, setStockSymbol] = useState("");
    const [loading, setLoading] = useState(false);

    const handlePredict = async () => {
        if (!stockSymbol) return;

        console.log("RUNNED");
        setLoading(true); // Show loader

        axios.post("http://127.0.0.1:5000/api/predict", { quote: stockSymbol }, { timeout: 180000 })
            .then(response => {
                setLoading(false); // Hide loader
                setPrediction(response.data); // Pass result to App.js
            })
            .catch(error => {
                setLoading(false);
                console.error("Error:", error);
            });
    };

    return (
        <div className="home-container flex flex-col justify-center items-center gap-10">
            <h1>STOCK PREDICTION</h1>
            <div className="input-container">
  <input 
    type="text" 
    placeholder="Enter stock symbol..."   
    value={stockSymbol}
    onChange={(e) => setStockSymbol(e.target.value)}
  />
 <button className="custom-button full-rounded" onClick={handlePredict} disabled={loading}>
  <span>{loading ? "Predicting..." : "Predict"}</span>
  <div className="border"></div>
</button>

  {loading && <div className="loader"></div>}
</div>

        </div>
    );
}

export default Home;

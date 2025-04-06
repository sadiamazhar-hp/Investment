import React, { useState } from "react";
import Home from "./components/Home";
import Result from "./components/Result";

function App() {
  // // Dummy prediction data
  // const dummyPrediction = {
  //   adj_close: 391.26,
  //   arima_forecast: [
  //     391.0421289340138,
  //     390.5275850213388,
  //     390.9642163730209,
  //     390.99265349058055,
  //     391.0958955828464,
  //     391.0279215792963,
  //     391.0184542682432
  //   ],
  //   arima_pred: 424.13,
  //   close: 391.26,
  //   decision: "SELL",
  //   error_arima: 19.09,
  //   error_lr: 24.24,
  //   error_lstm: 11.69,
  //   forecast_set: [
  //     [396.1236886447267],
  //     [405.74837635287184],
  //     [405.8860266173832],
  //     [400.79347686849724],
  //     [405.0208821039899],
  //     [404.0574202591223],
  //     [408.402794291489]
  //   ],
  //   high: 391.74,
  //   idea: "FALL",
  //   low: 382.8,
  //   lr_pred: 396.12,
  //   lstm_7days: [
  //     385.7200927734375,
  //     388.2629089355469,
  //     387.87274169921875,
  //     387.4012451171875,
  //     388.712890625,
  //     388.7657470703125,
  //     389.1737976074219
  //   ],
  //   lstm_pred: 415.77,
  //   open: 383.22,
  //   quote: "MSFT",
  //   volume: 39571500.0
  // };
  const [prediction, setPrediction] = useState(null);

  return (
    <div  className="min-h-screen bg-gradient-to-tl from-zinc-900 to-sky-900">
      <Home setPrediction={setPrediction} />
      {prediction && <Result prediction={prediction} />}
      {/* <Result prediction={prediction} /> Pass dummy data to Result */}
    </div>
  );
}


export default App;

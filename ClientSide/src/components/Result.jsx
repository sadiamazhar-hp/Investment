import React from "react";
import "./Result.css";
import fsImage from "./LR_MSFT.png";  // Adjust the path if needed
import { generateNextSevenDays } from "./HelperFunction";
import LineGraphComponent from "./Graph";
function Result({ prediction }) {
    const daysLabels = generateNextSevenDays(); // Calling the function
    return (
        <div className="result-container p-8">
            <h1>PREDICTION RESULT</h1>
            <h2>{prediction.quote}</h2>
            <div className="quote_summary text-white p-4 space-x-4">
                <span><strong>Adj Close:</strong> {prediction.adj_close}</span>
                <span><strong>Close:</strong> {prediction.close}</span>
                <span><strong>High:</strong> {prediction.high}</span>
                <span><strong>Low:</strong> {prediction.low}</span>
                <span><strong>Open:</strong> {prediction.open}</span>
            </div>
    {/*------------------------------------LINEAR REGRESSION------------------------------------------ */}
    <div className="p-6">
    <h3 className="text-xl font-bold mb-4 text-center">LINEAR REGRESSION</h3>
    <div className="Model">
        <div className="Model_Section_1 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            <div className="flex flex-col items-center">
            <h3 className="text-lg font-bold mb-2">Past Data Prediction</h3>
            {/* <img className="Model_Sample_Image border " src={fsImage} alt="Sample Image" /> */}
            <img className="Model_Sample_Image" src={`http://localhost:5000/static/LR_${prediction.quote}.png`} alt="Prediction Chart" />
            </div>
            <div className="flex flex-col gap-4">
            <div className="border p-3 w-48 text-center">
                <strong>Tomorrow's Prediction:</strong> {prediction.lr_pred}
            </div>
            <div className="border p-3 w-48 text-center">
                <strong>Error:</strong> {prediction.error_lr}
            </div>
            </div>

        </div>
        <div className="Model_Section_2 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            
            <div className="NextDays_Prediction p-6">
            <h3 className="text-lg font-bold mb-2">Next 7 Days Prediction</h3>
            <div className="Prediction">
                {daysLabels.map((label, index) => (
                <div key={index} className="day-label">
                    <span className="label">{label} :</span> {prediction.forecast_set[index] !== undefined ? parseFloat(prediction.forecast_set[index]).toFixed(2) : 'No data'}
                </div>
                ))}
            </div>





            </div>
            <div className="flex flex-col items-center">
            <LineGraphComponent forecastSet={prediction.forecast_set} />
            </div>

        </div>
        
    </div>
    </div>

   {/*------------------------------------ LSTM------------------------------------------ */}
    <div className="p-6">
    <h3 className="text-xl font-bold mb-4 text-center">LSTM MODEL</h3>
    <div className="Model">
        <div className="Model_Section_1 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            <div className="flex flex-col items-center">
            <h3 className="text-lg font-bold mb-2">Past Data Prediction</h3>
            <img className="Model_Sample_Image" src={`http://localhost:5000/static/LSTM_${prediction.quote}.png`} alt="Prediction Chart" />
            </div>
            <div className="flex flex-col gap-4">
            <div className="border p-3 w-48 text-center">
                <strong>Tomorrow's Prediction:</strong> {prediction.lstm_pred}
            </div>
            <div className="border p-3 w-48 text-center">
                <strong>Error:</strong> {prediction.error_lstm}
            </div>
            </div>

        </div>
        <div className="Model_Section_2 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            
            <div className="NextDays_Prediction p-6">
            <h3 className="text-lg font-bold mb-2">Next 7 Days Prediction</h3>
            <div className="Prediction">
  {daysLabels.map((label, index) => (
    <div key={index} className="day-label">
      <span className="label">{label} :</span>{' '}
      {prediction.lstm_7days[index] !== undefined
        ? parseFloat(prediction.lstm_7days[index]).toFixed(2)
        : 'No data'}
    </div>
  ))}
</div>






            </div>
            <div className="flex flex-col items-center">
            <LineGraphComponent forecastSet={prediction.lstm_7days.map(item => [item])} />

            </div>

        </div>
        
    </div>
    </div>
       {/*------------------------------------ARIMA MODEL------------------------------------------ */}
       <div className="p-6">
    <h3 className="text-xl font-bold mb-4 text-center">ARIMA MODEL</h3>
    <div className="Model">
        <div className="Model_Section_1 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            <div className="flex flex-col items-center">
            <h3 className="text-lg font-bold mb-2">Past Data Prediction</h3>
            <img className="Model_Sample_Image"  src={`http://localhost:5000/static/ARIMA_${prediction.quote}.png`} alt="Prediction Chart" /> 
            </div>
            <div className="flex flex-col gap-4">
            <div className="border p-3 w-48 text-center">
                <strong>Tomorrow's Prediction:</strong> {prediction.arima_pred}
            </div>
            <div className="border p-3 w-48 text-center">
                <strong>Error:</strong> {prediction.error_arima}
            </div>
            </div>

        </div>
        <div className="Model_Section_2 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg">
            
            <div className="NextDays_Prediction p-6">
            <h3 className="text-lg font-bold mb-2">Next 7 Days Prediction</h3>
            <div className="Prediction">
  {daysLabels.map((label, index) => (
    <div key={index} className="day-label">
      <span className="label">{label} :</span>{' '}
      {prediction.arima_forecast[index] !== undefined
        ? parseFloat(prediction.arima_forecast[index]).toFixed(2)
        : 'No data'}
    </div>
  ))}
</div>






            </div>
            <div className="flex flex-col items-center">
            <LineGraphComponent forecastSet={prediction.arima_forecast.map(item => [item])} />
            </div>

        </div>
        
    </div>
    <div className="Model_Section_3 flex flex-row items-center gap-6 p-4 border rounded-lg shadow-lg mt-6">
        <div class="flex decision_idea gap-4 pl-4">
            <div class="w-64 flex-1 ">IDEA</div>
            <div class="w-64 flex-1 ">{prediction.idea}</div>
            <div class="w-64 flex-1 ">DECISION</div>
            <div class="w-64 flex-1 ">{prediction.decision}</div>
        </div>
        </div>
    </div>


            {/* <img src={`http://localhost:5000/static/LSTM_${prediction.quote}.png`} alt="Prediction Chart" />
            <img src={`http://localhost:5000/static/LR_${prediction.quote}.png`} alt="Prediction Chart" />
            <img src={`http://localhost:5000/static/ARIMA_${prediction.quote}.png`} alt="Prediction Chart" /> */}

            {/* <pre>{prediction ? JSON.stringify(prediction, null, 2) : "No data received"}</pre> */}
        </div>
    );
}


export default Result;

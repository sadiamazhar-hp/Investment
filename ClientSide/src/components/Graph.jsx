import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const LineGraphComponent = ({ forecastSet }) => {
  // Extracting values from the forecast_set to create an array of numbers
  const data = forecastSet.map(item => item[0]);

  // Chart.js data format
  const chartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], // Days of the week
    datasets: [
      {
        label: 'Forecast for the Week',
        data: data, // The array of numbers passed as a prop
        borderColor: 'rgba(75, 192, 192, 1)', // Line color
        backgroundColor: 'rgba(75, 192, 192, 0.2)', // Fill color below the line
        fill: true, // Fill the area under the line
        tension: 0.4, // Curve of the line (0 = straight line, 1 = very curved)
        borderWidth: 2,
      },
    ],
  };

  // Chart options
  const options = {
    responsive: true,
    plugins: {
      title: {
        display: true,
        text: 'Forecast Data for a Week (Line Chart)',
      },
    },
  };

  return (
    <div>
      <h2>Weekly Forecast Line Chart</h2>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default LineGraphComponent;

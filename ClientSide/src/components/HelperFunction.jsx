// Function to generate next seven days labels
export const generateNextSevenDays = () => {
  const today = new Date();
  const labels = [];

  for (let i = 0; i < 7; i++) {
    const nextDay = new Date(today);
    nextDay.setDate(today.getDate() + i); // Set the date to next days
    
    const dayOfWeek = nextDay.toLocaleString('en-US', { weekday: 'long' });

    // Format the date to "MM-DD-YYYY" using hyphens
    const year = nextDay.getFullYear();
    const month = (nextDay.getMonth() + 1).toString().padStart(2, '0'); // Ensure two digits
    const date = nextDay.getDate().toString().padStart(2, '0'); // Ensure two digits
    const formattedDate = `${month}-${date}-${year}`;

    labels.push(`${dayOfWeek}, ${formattedDate}`);
  }

  return labels;
};

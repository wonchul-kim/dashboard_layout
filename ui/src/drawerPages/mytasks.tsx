import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
// import faker from 'faker';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: 'Chart.js Line Chart',
    },
  },
};


const labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July'];


export default function MyTasksPage() {
    const raws = {
        labels,
        datasets: [
            {
                label: 'Dataset 1',
                //   data: labels.map(() => faker.datatype.number({ min: -1000, max: 1000 })),
                data: [1, 2, 3, 4],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
            },
            {
                label: 'Dataset 2',
                //   data: labels.map(() => faker.datatype.number({ min: -1000, max: 1000 })),
                data: [1, 2, 3, 4],
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
            },
        ],
    };
    const [data, setData] = React.useState(raws);

    const arr = [1,2,3,4];


  return <Line options={options} data={data} />;
}

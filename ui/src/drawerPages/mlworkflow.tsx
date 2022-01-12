import React, { Component, useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Typography from '@material-ui/core/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Input from '@mui/material/Input';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { withStyles } from '@mui/styles';
import { Button, makeStyles } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import Stack from '@mui/material/Stack';
import axios from 'axios';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';
import socketIOClient from 'socket.io-client';


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
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const options = {
  responsive: true,
  plugins: {
    legend: {
      position: 'top' as const,
    },
    title: {
      display: true,
      text: 'validation loss',
    },
  },
};


const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  margin: "2vh",
}));

interface paramsState {
  project_name: String,
  model_name: String,
  wandb: boolean,
  device: String,
  learning_rate: Number,
  number_of_epochs: Number,
  batch_size: Number,
  image_size: Number,

  epoch: Number,
  trainValLoss: Number, 
  trainTrainLoss: Number, 
  trainStatus: String,
  info: String,
  connection: String,

  epochList: String[],
  trainValLossList: String[],
  trainTrainLossList: String[],
}

interface appProps {

}

class MLWorkflowPage extends Component<appProps, paramsState> {
  constructor(props: any) {
    super(props);
    this.state = {
      project_name: '',
      model_name: '',
      wandb: false,
      device: '',
      learning_rate: 0.01,
      number_of_epochs: 10,
      batch_size: 8,
      image_size: 3,

      epoch: 0,
      trainValLoss: 0,
      trainTrainLoss: 0, 
      trainStatus: "OFF",
      info: "",
      connection: "NOT Connected",

      epochList: [],
      trainValLossList: [],
      trainTrainLossList: [],
    }
  }

  epochList_ = [1, 2, 3];
  trainLossList_ = [1, 2, 3];
  // SOCKET IO SETTINGS *******************************************************
  fromServerTrainData = {
    labels: this.epochList_,
    datasets: [
      {
        label: 'val. loss',
        data: this.trainLossList_, 
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'train loss',
        data: [2, 2, 2.5],
        borderColor: 'rgb(53, 162, 235)',
        backgroundColor: 'rgba(53, 162, 235, 0.5)',
      },
    ],
  };

  socket = socketIOClient("http://localhost:5000", {
    reconnection: false
  })

  trainDataHandleFunc = (data: any) => {
    console.log("trainData", data);
    this.setState({ epoch: data.epoch, 
                    trainValLoss: data.trainValLoss,
                    epochList: [...this.state.epochList, data.epoch],
                    trainValLossList: [...this.state.trainValLossList, data.trainValLoss],
                  });
    console.log(">>>", this.state.trainValLossList)
    console.log(">>>>", this.state.epochList)
  }

  compononetWillUnmount() {
    this.socket.close();
    console.log("* Component Unmounted !!!");
    this.socket.off("trainData", this.trainDataHandleFunc);
  }

  componentWillMount() {
    // this.socket.on("connect", () => console.log("* CONNECTED TO BACKEND"));
    // this.socket.on('response', data => {
    //   this.setState({connection: data.connection});
    // });
    // this.socket.on("trainData", data => {
    //   this.setState({trainLoss: data.trainLoss,
    //                  info: data.info});

    //   console.log("trainData", data);
    // });
    this.socket.on("connect", () => console.log("* CONNECTED TO BACKEND >>>>>>>"));
    this.socket.on("disconnect", () => console.log("* DISCONNECTED TO BACKEND .....!!!"));
    // this.socket.emit("Start", {"status": "Start the training >>>"});
    // this.setState({trainStatus: "ON"});
    this.socket.on('response', (data: any) => {
      this.setState({connection: data.connection});
      console.log("response from connection", data.connection);
    });
    this.socket.on("trainData", this.trainDataHandleFunc);
  }

  handleTrainButton = (e: any) => {
    console.log("Clicked Train Button")
    e.preventDefault();
    this.sendParams();
    // this.socket.on("connect", () => console.log("* CONNECTED TO BACKEND"));
    this.socket.emit("Start", {"status": "Start the training >>>"});
    this.setState({trainStatus: "ON"});
    // this.socket.on('response', data => {
    //   this.setState({connection: data.connection});
    //   console.log("response from connection", data.connection);
    // });
    // this.socket.on("trainData", this.trainDataHandleFunc);
  }

  handleStopButton = (e: any) => {
    console.log("Clicked Stop Button");
    e.preventDefault();
    this.socket.emit("Stop", {"status": "Stop the training !!!!"});
    this.setState({trainStatus: "OFF"});
  }

  sendParams = async () => {
    const url = '/mlworkflow/train-parameters'; 
    // const formData = new FormData();
    // formData.append('project_name', this.state.project_name);
    // formData.append('model_name', this.state.model_name);
    // formData.append('wandb', String(this.state.wandb));
    // formData.append('device', this.state.device);
    // formData.append('learning_rate', String(this.state.learning_rate));
    // formData.append('number_of_epochs', String(this.state.number_of_epochs));
    // formData.append('batch_size', String(this.state.batch_size));
    // formData.append('image_size', String(this.state.image_size));

    // return post(url, {
    //   method: 'POST',
    //   headers: {
    //     'content-type': 'multipart/form-data'
    //   },
    //   data: formData
    // });
    return await axios.post(url, this.state)
  }

  handleLoadButton = (e: any) => {

  }

  handleResetButton = (e: any) => {
    this.setState({
      project_name: '',
      model_name: '',
      wandb: false,
      device: '',
      learning_rate: 0.01,
      number_of_epochs: 10,
      batch_size: 8,
      image_size: 3
    });
  }

  handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    this.setState((state) => ({ ...state, [e.target.name]: e.target.value }));
  }

  render() {
    return (
      // <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={1}>
        <Grid item xs={4}>
          <Item style={{ height: '100vh' }}>

            <p>Parameteres</p>
            <br />
            <Button variant='contained' color='primary' onClick={this.handleLoadButton} > Load </Button>

            <FormControl variant="standard">
              <TextField id="project-name" label="Project Name" variant="standard" style={{ width: '300px' }}
                name="project_name" value={this.state.project_name} onChange={(e: any) => this.handleChange(e)} />
            </FormControl>

            <br /><br />
            <FormControl sx={{ m: 1, minWidth: 200 }}>
              <Select
                name="model_name"
                value={this.state.model_name}
                onChange={(e: any) => this.handleChange(e)}
                displayEmpty
                inputProps={{ 'aria-label': 'Without label' }}
                size='small'
                style={{ width: '300px' }}
              >
                <MenuItem value=""> <em> Model Name </em> </MenuItem>
                <MenuItem value="deeplabv3_resnet101">Deeplab v3 ResNet101</MenuItem>
                <MenuItem value="deeplabv3_resnet50">Deeplab v3 ResNet50</MenuItem>
                <MenuItem value="unetpp">Unet++</MenuItem>
              </Select>
            </FormControl>

            <br /><br />
            <FormControl sx={{ m: 1, minWidth: 200 }}>
              <Select
                name="device"
                value={this.state.device}
                onChange={(e: any) => this.handleChange(e)}
                displayEmpty
                inputProps={{ 'aria-label': 'Without label' }}
                size='small'
                style={{ width: '300px' }}
              >
                <MenuItem value=""> <em> Device </em> </MenuItem>
                <MenuItem value="gpu">GPU</MenuItem>
                <MenuItem value="cpu">CPU</MenuItem>
              </Select>
            </FormControl>

            <br /><br />
            <Box>
              <FormControlLabel control={<Switch defaultChecked />} label="wandb" />
            </Box>

            <br /><br />
            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Learning Rate</InputLabel>
              <Input id="component-simple" name="learning_rate" value={this.state.learning_rate}
                onChange={(e: any) => this.handleChange(e)} style={{ width: '300px' }} />
            </FormControl>

            <br /><br />
            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Number of epochs</InputLabel>
              <Input id="component-simple" name="number_of_epochs" value={this.state.number_of_epochs}
                onChange={(e: any) => this.handleChange(e)} style={{ width: '300px' }} />
            </FormControl>

            <br /><br />
            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Batch Size</InputLabel>
              <Input id="component-simple" name="batch_size" value={this.state.batch_size}
                onChange={(e: any) => this.handleChange(e)} style={{ width: '300px' }} />
            </FormControl>

            <br /><br />
            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Image Size</InputLabel>
              <Input id="component-simple" name="image_size" value={this.state.image_size}
                onChange={(e: any) => this.handleChange(e)} style={{ width: '300px' }} />
              <br /><br />
            </FormControl>

            <Stack direction="row" spacing={10}>
              <Button variant='contained' endIcon={<SendIcon />} onClick={this.handleTrainButton}> Train </Button>
              <Button variant='contained' endIcon={<SendIcon />} onClick={this.handleStopButton}> Stop </Button>
              <Button variant='contained' color='error' startIcon={<DeleteIcon />} onClick={this.handleResetButton}> Reset </Button>
            </Stack>
          </Item>


        </Grid>
        <Grid item xs={8}>
          <Item style={{ height: '48.2vh' }}>
            Datasets
          </Item>
          <Item style={{ height: '48.2vh' }}>
            validation loss (epoch: {this.state.epoch}  loss: {this.state.trainValLossList}) <br/>
            {this.state.trainValLossList}
            <Line options={options} data={this.fromServerTrainData} />
            </Item>
        </Grid>
      </Grid>


      // </Box>
    );
  }
}

export default MLWorkflowPage;


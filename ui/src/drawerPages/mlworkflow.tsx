import React, { Component } from 'react';
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
import post from 'axios';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';


const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  margin: "2vh",
}));

interface paramsState {
  project_name: string,
  model_name: string,
  wandb: boolean,
  device: string,
  learning_rate: number,
  number_of_epochs: number,
  batch_size: number,
  image_size: number,
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
      image_size: 3
    }
  }

  handleApplyButton = (e: any) => {
    e.preventDefault()
    this.sendParams()
  }

  sendParams = () => {
    const url = '/mlworkflow/train-parameters';
    const formData = new FormData();
    formData.append('project_name', this.state.project_name);
    formData.append('model_name', this.state.model_name);
    // formData.append('wandb', this.state.wandb);
    formData.append('device', this.state.device);
    formData.append('learning_rate', String(this.state.learning_rate));
    formData.append('number_of_epochs', String(this.state.number_of_epochs));
    formData.append('batch_size', String(this.state.batch_size));
    formData.append('image_size', String(this.state.image_size));

    return post(url, {
      method: 'POST',
      headers: {
        'content-type': 'multipart/form-data'
      },
      data: formData
    });
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
                <MenuItem value="deeplabv3">Deeplab v3</MenuItem>
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
              <Button variant='contained' endIcon={<SendIcon />} onClick={this.handleApplyButton}> Apply </Button>
              <Button variant='contained' color='error' startIcon={<DeleteIcon />} onClick={this.handleResetButton}> Reset </Button>
            </Stack>
          </Item>


        </Grid>
        <Grid item xs={8}>
          <Item style={{ height: '48.2vh' }}>
            Datasets
          </Item>
          <Item style={{ height: '48.2vh' }}>
            training graph
          </Item>
        </Grid>
      </Grid>


      // </Box>
    );
  }
}

export default MLWorkflowPage;


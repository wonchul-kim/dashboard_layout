import React, {Component} from 'react';
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


const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  margin: "2vh",
}));

class MLWorkflowPage extends Component {

  constructor(props: any){
    super(props);
    this.state = {
      project_name: '',
      model_name: '',
      wandb: '',
      device: '', 
      learning_rate: 0.01,
      num_epochs: 10,
      batch_size: 8,
      image_size: 3
    }
  }

  handleApplyButton = (e: any) => {
    e.preventDefault()
    this.saveParams()
      // .then((response: any) => {
      //   console.log(response.data);
      //   // this.props.stateRefresh();
      // })
  }

  saveParams = () => {
    const url = '/api/apply_parameters';
    const formData = new FormData();

    // // formData.append("project_name", this.state.project_name);
    // // formData.append("model_name", this.state.model_name);
    // // formData.append("wandb", this.state.wandb);
    // // formData.append("device", this.state.device);
    // // formData.append("learning_rate", this.state.learning_rate);
    // // formData.append("number_of_epochs", this.state.number_of_epochs);
    // // formData.append("batch_size", this.state.batch_size);
    // // formData.append("image_size", this.state.image_size);

    // const config = {
    //   header: {
    //     "content-type": "multipart/form-data"
    //   }
    // }

    // return post(url, formData, config);
    
  }

  handleResetButton = () => {
    this.setState({ // 모든 정보에 대한 초기화
      project_name: '',
      model_name: '',
      wandb: '',
      device: '', 
      learning_rate: 0.01,
      num_epochs: 10,
      batch_size: 8,
      image_size: 3
    });
  }

  handleLoadButton = () => {
    this.setState({ 
      
    });
  }

  render(){
    return(
      <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={1}>
        <Grid item xs={4}>
          <Item style={{height: '100vh'}}>

            <p>Parameteres</p>
            <br/>
            <Box component="span" m={10}>
              <Button variant='contained' color='primary' onClick={this.handleLoadButton} > Load </Button>
            </Box>


            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Project Name</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>

            <br/><br/>
            <FormControl sx={{ m: 1, minWidth: 200 }}>
              <Select
                value={''}
                // onChange={handleChange}
                displayEmpty
                inputProps={{ 'aria-label': 'Without label' }}
                size='small'
                style={{ width: '300px'}}
              >
              <MenuItem value="">
                <em>Model Name</em>
              </MenuItem>
              <MenuItem value={10}>Deeplab v3</MenuItem>
              <MenuItem value={20}>Unet++</MenuItem>
              {/* <MenuItem value={30}>Thirty</MenuItem> */}
            </Select>
            </FormControl>

            <br/><br/>
            <FormControl sx={{ m: 1, minWidth: 200 }}>
              <Select
                value={''}
                // onChange={handleChange}
                displayEmpty
                inputProps={{ 'aria-label': 'Without label' }}
                size='small'
                style={{ width: '300px'}}
              >
              <MenuItem value="">
                <em>Device</em>
              </MenuItem>
              <MenuItem value={10}>GPU</MenuItem>
              <MenuItem value={20}>CPU</MenuItem>
              {/* <MenuItem value={30}>Thirty</MenuItem> */}
            </Select>
            </FormControl>
          

            <br/><br/>
            <FormControl variant="standard">
                <InputLabel htmlFor="component-simple">Learning Rate</InputLabel>
                <Input id="component-simple" value={''} style={{ width: '300px' }} />
              </FormControl>

              <br/><br/>
            <FormControl variant="standard">
                <InputLabel htmlFor="component-simple">Number of epochs</InputLabel>
                <Input id="component-simple" value={''} style={{ width: '300px' }} />
              </FormControl>

              <br/><br/>
            <FormControl variant="standard">
                <InputLabel htmlFor="component-simple">Batch Size</InputLabel>
                <Input id="component-simple" value={''} style={{ width: '300px' }} />
              </FormControl>

              <br/><br/>
            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Image Size</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
              <br/><br/>
            </FormControl> 
          
          <Stack direction="row" spacing={10}>
            <Button variant='contained' endIcon={<SendIcon />} onClick={this.handleApplyButton}> Apply </Button> 
            <Button variant='contained' color='error' startIcon={<DeleteIcon />} onClick={this.handleResetButton}> Reset </Button>
          </Stack>
          </Item>


        </Grid>
          <Grid item xs={8}>
            <Item style={{height: '48.2vh'}}>
              Datasets
            </Item>
            <Item style={{height: '48.2vh'}}>
              training graph
            </Item>
          </Grid>
        </Grid>


      </Box>
    );
  }
}

export default MLWorkflowPage;


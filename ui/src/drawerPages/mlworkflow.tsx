import * as React from 'react';
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

// project_name: '',
// model_name: '',
// wandb: '',
// device: '', 
// learning_rate: 0.01,
// num_epochs: 10,
// batch_size: 8,
// image_size: 32

function parametersInfo() {
  return (
    <div>
      <h1>
        parameters
      </h1>
    </div>
    
  );
};

const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  margin: "2vh",
}));

function MLWorkflowPage() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={1}>
        <Grid item xs={4}>
          <Item style={{height: '100vh'}}>


            {/* <parametersInfo /> */}

            <p>Parameteres</p>

            <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Project Name</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>

            <p> </p>
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

            <p> </p>
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
          
          <p> </p>
          <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Learning Rate</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>

            <p> </p>
          <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Number of epochs</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>

            <p> </p>
          <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Batch Size</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>

            <p> </p>
          <FormControl variant="standard">
              <InputLabel htmlFor="component-simple">Image Size</InputLabel>
              <Input id="component-simple" value={''} style={{ width: '300px' }} />
            </FormControl>


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
export default MLWorkflowPage;


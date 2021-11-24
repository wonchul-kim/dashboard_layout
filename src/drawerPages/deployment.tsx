import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Typography from '@material-ui/core/Typography';

import TextField from '@mui/material/TextField';

import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormHelperText from '@mui/material/FormHelperText';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

const Item = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(1),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  margin: "10px",
}));


const types = ['Detection', 'Segmentation', 'OCR'];
const detections = ['YOLO'];
const segmentations = ['Deeplab v3', 'Unet++'];
const ocrs = ['OCR'];
  
function SelectLabels() {
    const [age, setAge] = React.useState('');
  
    const handleChange = (event) => {
      setAge(event.target.value);
    };
  
    return (
      <div>
        <FormControl sx={{ m: 1, minWidth: 120 }}>
          <InputLabel id="demo-simple-select-helper-label">Age</InputLabel>
          <Select
            labelId="demo-simple-select-helper-label"
            id="demo-simple-select-helper"
            value={age}
            label="Age"
            onChange={handleChange}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
          <FormHelperText>With label + helper text</FormHelperText>
        </FormControl>
        <FormControl sx={{ m: 1, minWidth: 120 }}>
          <Select
            value={age}
            onChange={handleChange}
            displayEmpty
            inputProps={{ 'aria-label': 'Without label' }}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
          <FormHelperText>Without label</FormHelperText>
        </FormControl>
      </div>
    );
  }
  


function DeploymentPage() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={1}>
        <Grid item xs={4}>
          <Item style={{height: '30vh'}}>
            <Typography align='left' variant='subtitle1' style={{ fontWeight: 600 }} gutterBottom>
                Type the project name
                <TextField fullWidth
                id="outlined-multiline-flexible"
                label="Project name"
                multiline
                maxRows={2}
                margin='normal'
            />
            </Typography>   
            
            <Typography align='left' variant='subtitle1' style={{ fontWeight: 600 }}>
                Selcet model
            </Typography> 
            
         </Item>
        </Grid>
        <Grid item xs={8}>
          <Item style={{height: '30vh'}}>
            Datasets
          </Item>
        </Grid>
        <Grid item xs={6}>
          <Item style={{height: '20vh'}}>
            Results1
          </Item>
        </Grid>
        <Grid item xs={6}>
          <Item style={{height: '20vh'}}>
            Results2
          </Item>
        </Grid>
        <Grid item xs={12}>
          <Item style={{height: '30vh'}}>
            Results
          </Item>
        </Grid>
      </Grid>
    </Box>
  );
}

export default DeploymentPage;

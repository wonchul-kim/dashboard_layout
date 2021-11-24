import * as React from 'react';
import { styled } from '@mui/material/styles';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';

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
              paramters
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

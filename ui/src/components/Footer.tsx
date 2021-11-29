import Typography from "@material-ui/core/Typography";
import Grid from '@mui/material/Grid';
import Box from '@mui/material/Box';
import { makeStyles, Theme } from "@material-ui/core/styles";

const useStyles = makeStyles((theme: Theme) => ({
  footer: {
    background: '#fafafa', //theme.palette.primary.dark, // Footer background color
    color: '#212121', //theme.palette.secondary.light, // Footer texts color
    padding: theme.spacing(0.5), // Footer height
  },
}));

const Footer = () => {
  const classes = useStyles();
  return (
    <div className={classes.footer}>
      <Typography variant='body2' color='inherit'>
        @ AIV Corporation
      </Typography>
    </div>
  );
};


export default Footer;

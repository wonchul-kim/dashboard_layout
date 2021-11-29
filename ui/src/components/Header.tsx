import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import IconButton from "@material-ui/core/IconButton";
import MenuIcon from "@material-ui/icons/Menu";
import ChevronLeftIcon from "@material-ui/icons/ChevronLeft";
import Typography from "@material-ui/core/Typography";
import { makeStyles, Theme } from "@material-ui/core/styles";
import { useDrawerContext } from "../contexts/drawer-context";
import Button from '@mui/material/Button';

const headersData = [
  {
    label: "Projects",
    href: "/projects",
  },
  {
    label: "Devices",
    href: "/devices",
  },
  {
    label: "Members",
    href: "/members",
  },
];

const getMenuButtons = () => {
  return headersData.map(({ label, href }) => {
    return (
      <Button
        {...{
          key: label,
          color: "inherit",
          to: href,
          // component: RouterLink,
        }}
      >
        {label}
      </Button>
    );
  });
};


const useStyles = makeStyles((theme: Theme) => ({
  appBar: {
    background: '#fafafa', //theme.palette.primary.dark,
    color: '#dd2c00', //theme.palette.secondary.light,
  },
  icon: {
    padding: theme.spacing(1),
  },
  title: {
    // margin: "auto",
    // paddingRight:"1118x"
    fontFamily: "Work Sans, sans-serif",
    fontWeight: 600,
    // color: "#FFFEFE",
    textAlign: "left",
  },
  headers: {
    fontFamily: "Open Sans, sans-serif",
    fontWeight: 700,
    size: "18px",
    marginLeft: "38px",
  }
}));

const Header = () => {
  const classes = useStyles();
  const { isOpened, toggleIsOpened } = useDrawerContext();
  return (
    <AppBar className={classes.appBar}>
      <Toolbar>
        <IconButton
          color="inherit"
          onClick={() => toggleIsOpened(!isOpened)}
          className={classes.icon}
        >
          {isOpened ? <ChevronLeftIcon /> : <MenuIcon />}
        </IconButton>
        <Typography variant="h6" className={classes.title}>
          AIV Dashboard
          {getMenuButtons()}

        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Header;

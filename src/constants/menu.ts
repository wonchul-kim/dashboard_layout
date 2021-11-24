import { DrawerItem, HeaderItem } from '../ts';
import { ROUTES } from './routes';
import AccountTreeSharpIcon from '@material-ui/icons/AccountTreeSharp';
import DashboardIcon from '@material-ui/icons/Dashboard';
import ModelTrainingSharpIcon from '@mui/icons-material/ModelTrainingSharp';
import StorageIcon from '@material-ui/icons/Storage';
import TaskIcon from '@mui/icons-material/Task';
import LaptopChromebookIcon from '@mui/icons-material/LaptopChromebook';

export const DRAWER_LIST: DrawerItem[] = [
  {
    route: ROUTES.mlWorkflow,
    literal: 'ML Workflow',
    Icon: AccountTreeSharpIcon,
  },
  {
    route: ROUTES.data,
    literal: 'Data',
    Icon: StorageIcon,
  },
  {
    route: ROUTES.models,
    literal: 'Models',
    Icon: StorageIcon,
  },
  {
    route: ROUTES.deployment,
    literal: 'Deployment',
    Icon: StorageIcon,
  },
  {
    route: ROUTES.myTasks,
    literal: 'My tasks',
    Icon: DashboardIcon,
  },
];

export const Header_LIST: HeaderItem[] = [
  {
    route: ROUTES.projects,
    literal: 'Projects',
    // Icon: AccountTreeSharpIcon,
  },
  {
    route: ROUTES.devices,
    literal: 'Devices',
    // Icon: StorageIcon,
  },
  {
    route: ROUTES.members,
    literal: 'Members',
    // Icon: StorageIcon,
  },
];


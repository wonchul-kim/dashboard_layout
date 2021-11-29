import { Redirect, Route, Switch } from "react-router-dom";

import { ROUTES } from "./constants/routes";
import Layout from "./components/Layout";
import {
  MLWorkflowPage,
  ModelsPage,
  DataPage,
  DeploymentPage,
  MyTasksPage,
} from "./drawerPages";

// import {
//   DevicesPage,
//   MembersPage,
//   ProjectsPage
// } from "./headerPages";

function App() {
  return (
    <Layout>
      <Switch>
        <Route path="/" exact render={() => <Redirect to={ROUTES.mlWorkflow} />} />

        <Route exact path={ROUTES.mlWorkflow} component={MLWorkflowPage} />
        <Route exact path={ROUTES.data} component={DataPage} />
        <Route exact path={ROUTES.models} component={ModelsPage} />
        <Route exact path={ROUTES.deployment} component={DeploymentPage} />
        <Route exact path={ROUTES.myTasks} component={MyTasksPage} />

        {/* <Route exact path={ROUTES.myTasks} component={DevicesPage} />
        <Route exact path={ROUTES.myTasks} component={MembersPage} />
        <Route exact path={ROUTES.myTasks} component={ProjectsPage} /> */}
      </Switch>
    </Layout>
  );
}

export default App;

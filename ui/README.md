# React Material UI Dashboard Layout template

[Live-Demo](https://katerinalupacheva.github.io/dashboard-layout/)

Starter code with the implementation of:

- Basic layout: header, the main content area with drawer, footer
- Drawer toggle
- Navigation between pages

![demo](demo.gif)

## Features

- React v.17
- TypeScript v.4
- Material-UI v.4
- React Router v.5
- React Context
- React Hooks
- CSS-in-JS styles
- Responsive
- Create-react-app under the hood


### Make desktop app using electron

1. Install electron & electron-builder
    ```
    npm install -D electron electron-builder
    ```

2. Write `electron.js` as app starter in public folder

3. Add electron configuration in `package.json`
    ```
    "description": "This is a project demonstrating create-react-app with electron.",
    "main": "public/electron.js",
    ```
    ```
    "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build",
    "electron": "electron .",
    "electron-pack": "yarn build && electron-builder build -c.extraMetadata.main=build/electron.js"
    },
    ```
    ```
    "author": {
    "email": "onedang22@gmail.com"
    }
    ```

4. Execute electron and make app
    
    * test electron based on localhost from `npm start`
    ```
    npm start
    ```
    on another terminal,
    ```
    npm run electron
    ```
    
    * make app 
    ```
    npm run electron-pack
    ```
    

### How to create from scratch

I wrote the blog post on how to create Dashboard layout. You can read it [here](https://ramonak.io/posts/dashboard-layout-react-material-ui).

### Pure React version

The starter code of this Dashboard layout in pure React.js (without Material-UI) is in [this branch](https://github.com/KaterinaLupacheva/dashboard-layout/tree/pure-react).

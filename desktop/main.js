'use strict';

const { app, BrowserWindow, Menu, ipcMain, shell } = require('electron');
const path = require('path');

// Security: disable remote content navigation
app.on('web-contents-created', (_, contents) => {
  contents.on('will-navigate', (e, url) => {
    const parsed = new URL(url);
    // Only allow local files and localhost WebSocket targets
    if (parsed.protocol !== 'file:' && parsed.hostname !== 'localhost') {
      e.preventDefault();
    }
  });
  // Block new windows from opening external URLs
  contents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://localhost') || url.startsWith('file://')) {
      return { action: 'allow' };
    }
    shell.openExternal(url);
    return { action: 'deny' };
  });
});

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'EchoPose',
    backgroundColor: '#070b12',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,   // Security: no Node in renderer
      sandbox: true,
      webSecurity: true,
    },
  });

  win.loadFile(path.join(__dirname, 'www', 'tablet.html'));

  // Remove default menu in production
  if (!process.env.EP_DEV) {
    Menu.setApplicationMenu(buildMenu(win));
  }

  return win;
}

function buildMenu(win) {
  const isMac = process.platform === 'darwin';
  return Menu.buildFromTemplate([
    ...(isMac ? [{ role: 'appMenu' }] : []),
    {
      label: 'View',
      submenu: [
        { label: 'Mobile Dashboard', click: () => win.loadFile(path.join(__dirname, 'www', 'mobile.html')) },
        { label: 'Tablet Dashboard', click: () => win.loadFile(path.join(__dirname, 'www', 'tablet.html')) },
        { type: 'separator' },
        { role: 'reload' },
        { role: 'forceReload' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        ...(isMac ? [{ role: 'zoom' }] : [{ role: 'close' }]),
      ],
    },
  ]);
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

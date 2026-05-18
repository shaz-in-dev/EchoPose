'use strict';
// Preload: runs in renderer context with Node access disabled.
// Exposes only the minimal bridge needed by the UI.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  version:  process.versions.electron,
});

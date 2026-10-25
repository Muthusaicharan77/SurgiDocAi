import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// In development the UI runs on Vite (port 5173) and forwards API calls to uvicorn.
// In production `npm run build` writes dist/, which the API serves at /.
const api = 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: { proxy: { '/runs': api, '/health': api } },
});

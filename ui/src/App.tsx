import { HashRouter, Navigate, Route, Routes } from 'react-router-dom';
import UploadScreen from './components/UploadScreen';
import RunView from './components/RunView';
import SwipeReview from './components/SwipeReview';
import ReportView from './components/ReportView';

// Hash routes (/#/runs/abc), so client-side URLs never collide with the API's /runs paths.
export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<UploadScreen />} />
        <Route path="/runs/:id" element={<RunView />} />
        <Route path="/runs/:id/swipe" element={<SwipeReview />} />
        <Route path="/runs/:id/report" element={<ReportView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  );
}

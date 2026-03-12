import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Upload from './pages/Upload';
import Chat from './pages/Chat';
import JobMatch from './pages/JobMatch';
import Settings from './pages/Settings';

export default function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/match" element={<JobMatch />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}

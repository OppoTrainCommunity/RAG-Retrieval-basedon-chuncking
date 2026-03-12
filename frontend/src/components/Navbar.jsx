import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { HiOutlineUpload, HiOutlineChat, HiOutlineBriefcase, HiOutlineCog, HiOutlineSun, HiOutlineMoon } from 'react-icons/hi';

const navItems = [
  { path: '/', icon: HiOutlineUpload, label: 'Upload' },
  { path: '/chat', icon: HiOutlineChat, label: 'Chat' },
  { path: '/match', icon: HiOutlineBriefcase, label: 'Job Match' },
  { path: '/settings', icon: HiOutlineCog, label: 'Settings' },
];

export default function Navbar() {
  const { darkMode, toggleTheme } = useTheme();
  const location = useLocation();

  return (
    <nav className="glass-card mx-4 mt-4 px-6 py-3 flex items-center justify-between sticky top-4 z-50">
      <div className="flex items-center gap-2">
        <span className="text-xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
          OppoTrain - CV Analyzer
        </span>
      </div>

      <div className="flex items-center gap-1">
        {navItems.map(({ path, icon: Icon, label }) => (
          <Link
            key={path}
            to={path}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200
              ${location.pathname === path
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}
          >
            <Icon className="w-4 h-4" />
            <span className="hidden sm:inline">{label}</span>
          </Link>
        ))}
      </div>

      <button
        onClick={toggleTheme}
        className="p-2 rounded-xl hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
      >
        {darkMode ? <HiOutlineSun className="w-5 h-5 text-yellow-400" /> : <HiOutlineMoon className="w-5 h-5 text-gray-600" />}
      </button>
    </nav>
  );
}

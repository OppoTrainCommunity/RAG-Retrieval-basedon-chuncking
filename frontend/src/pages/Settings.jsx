import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { HiOutlineKey, HiOutlineEye, HiOutlineEyeOff, HiOutlineCheck, HiOutlineX } from 'react-icons/hi';
import { saveApiKey, validateApiKey } from '../services/api';

export default function Settings() {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [status, setStatus] = useState('unknown'); // unknown | valid | invalid
  const [saving, setSaving] = useState(false);
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem('openrouter_api_key');
    if (stored) {
      setApiKey(stored);
      setStatus('unknown');
    }
  }, []);

  const handleValidate = async () => {
    if (!apiKey.trim()) return;
    setValidating(true);
    try {
      const res = await validateApiKey(apiKey);
      setStatus(res.data.success ? 'valid' : 'invalid');
      toast[res.data.success ? 'success' : 'error'](res.data.message);
    } catch {
      setStatus('invalid');
      toast.error('Validation failed');
    } finally {
      setValidating(false);
    }
  };

  const handleSave = async () => {
    if (!apiKey.trim()) return;
    setSaving(true);
    try {
      const res = await saveApiKey(apiKey);
      localStorage.setItem('openrouter_api_key', apiKey);
      setStatus(res.data.success ? 'valid' : 'invalid');
      toast.success('API key saved');
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  const handleClear = () => {
    setApiKey('');
    setStatus('unknown');
    localStorage.removeItem('openrouter_api_key');
    toast.success('API key cleared');
  };

  const statusBadge = {
    valid: <span className="flex items-center gap-1 text-xs text-green-500"><HiOutlineCheck className="w-3.5 h-3.5" /> Active</span>,
    invalid: <span className="flex items-center gap-1 text-xs text-red-500"><HiOutlineX className="w-3.5 h-3.5" /> Invalid</span>,
    unknown: <span className="text-xs text-gray-500">Not verified</span>,
  };

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Settings</h1>

      <div className="glass-card p-6 space-y-5">
        <div className="flex items-center gap-3">
          <HiOutlineKey className="w-6 h-6 text-blue-500" />
          <div>
            <h2 className="font-semibold">OpenRouter API Key</h2>
            <p className="text-sm text-gray-500">Required for LLM features (chat, metadata extraction, matching)</p>
          </div>
        </div>

        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => { setApiKey(e.target.value); setStatus('unknown'); }}
            placeholder="sk-or-v1-..."
            className="glass-input w-full pr-10"
          />
          <button
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-400"
          >
            {showKey ? <HiOutlineEyeOff className="w-4 h-4" /> : <HiOutlineEye className="w-4 h-4" />}
          </button>
        </div>

        <div className="flex items-center justify-between">
          <div>{statusBadge[status]}</div>
          <div className="flex gap-2">
            <button onClick={handleClear} className="glass-button glass-button-danger text-sm">Clear</button>
            <button onClick={handleValidate} disabled={validating || !apiKey.trim()} className="glass-button glass-button-secondary text-sm">
              {validating ? 'Validating...' : 'Validate'}
            </button>
            <button onClick={handleSave} disabled={saving || !apiKey.trim()} className="glass-button glass-button-primary text-sm">
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      </div>

      <div className="glass-card p-6 space-y-3">
        <h2 className="font-semibold">About</h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          CV Analysis RAG System — Upload CVs, chat with them using AI, match candidates to job descriptions, and view analytics.
        </p>
        <p className="text-sm text-gray-500">
          Powered by FastAPI, LangChain, ChromaDB, FastEmbed, and React.
        </p>
      </div>
    </div>
  );
}

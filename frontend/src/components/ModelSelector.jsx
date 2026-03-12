import { useState, useEffect } from 'react';
import { getModels } from '../services/api';

export default function ModelSelector({ selectedModel, onModelChange }) {
  const [models, setModels] = useState([]);
  const [customModel, setCustomModel] = useState('');
  const [useCustom, setUseCustom] = useState(false);

  useEffect(() => {
    getModels().then((res) => setModels(res.data.models)).catch(() => {});
  }, []);

  const handleSelectChange = (e) => {
    const value = e.target.value;
    if (value === '__custom__') {
      setUseCustom(true);
      onModelChange(customModel);
    } else {
      setUseCustom(false);
      onModelChange(value);
    }
  };

  const handleCustomChange = (e) => {
    setCustomModel(e.target.value);
    onModelChange(e.target.value);
  };

  return (
    <div className="flex flex-col sm:flex-row gap-2">
      <select
        value={useCustom ? '__custom__' : selectedModel}
        onChange={handleSelectChange}
        className="glass-input text-sm flex-1"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>{m.name}</option>
        ))}
        <option value="__custom__">Custom Model ID...</option>
      </select>
      {useCustom && (
        <input
          type="text"
          value={customModel}
          onChange={handleCustomChange}
          placeholder="e.g. openai/gpt-4o"
          className="glass-input text-sm flex-1"
        />
      )}
    </div>
  );
}

import { useState } from 'react';
import { HiOutlineChevronDown, HiOutlineChevronUp } from 'react-icons/hi';

export default function SourceCitation({ sources }) {
  const [expanded, setExpanded] = useState(false);

  if (!sources?.length) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-400 transition-colors"
      >
        {expanded ? <HiOutlineChevronUp className="w-3 h-3" /> : <HiOutlineChevronDown className="w-3 h-3" />}
        {sources.length} source{sources.length !== 1 ? 's' : ''}
      </button>
      {expanded && (
        <div className="mt-2 space-y-2 animate-fade-in">
          {sources.map((source, i) => (
            <div key={i} className="text-xs p-2 rounded-lg bg-gray-100 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
              <span className="font-medium text-blue-400">{source.name}</span>
              <span className="text-gray-500 ml-2">({(source.relevance * 100).toFixed(0)}% relevant)</span>
              <p className="mt-1 text-gray-600 dark:text-gray-400 line-clamp-2">{source.snippet}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { HiOutlinePaperAirplane } from 'react-icons/hi';
import { sendChat } from '../services/api';
import ModelSelector from '../components/ModelSelector';
import SourceCitation from '../components/SourceCitation';

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('deepseek/deepseek-chat-v3-0324:free');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || loading) return;

    const userMsg = { role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendChat(query, model);
      const aiMsg = { role: 'assistant', content: res.data.response, sources: res.data.sources };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      const detail = err.response?.data?.detail || 'Chat failed';
      toast.error(detail);
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${detail}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 flex flex-col h-[calc(100vh-8rem)] animate-fade-in">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">Chat with CVs</h1>
        <div className="w-72">
          <ModelSelector selectedModel={model} onModelChange={setModel} />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <p>Ask anything about the uploaded CVs...</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-2xl ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white rounded-br-md'
                : 'glass-card rounded-bl-md'
            }`}>
              {msg.role === 'assistant' ? (
                <>
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  <SourceCitation sources={msg.sources} />
                </>
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="glass-card p-4 rounded-2xl rounded-bl-md">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
          rows={1}
          className="glass-input flex-1 resize-none"
        />
        <button onClick={handleSend} disabled={loading || !input.trim()} className="glass-button glass-button-primary px-4">
          <HiOutlinePaperAirplane className="w-5 h-5 rotate-90" />
        </button>
      </div>
    </div>
  );
}

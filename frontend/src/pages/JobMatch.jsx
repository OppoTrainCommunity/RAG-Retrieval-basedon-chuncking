import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { matchCandidates, getSkillsAnalytics, getExperienceAnalytics } from '../services/api';
import ModelSelector from '../components/ModelSelector';
import CircleProgress from '../components/CircleProgress';
import SkillBadge from '../components/SkillBadge';
import { BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

export default function JobMatch() {
  const [jobDescription, setJobDescription] = useState('');
  const [model, setModel] = useState('deepseek/deepseek-chat-v3-0324:free');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [skillsData, setSkillsData] = useState([]);
  const [expData, setExpData] = useState([]);

  useEffect(() => {
    Promise.all([getSkillsAnalytics(), getExperienceAnalytics()])
      .then(([s, e]) => {
        setSkillsData(s.data.skills);
        setExpData(e.data.distribution);
      })
      .catch(() => {});
  }, []);

  const handleMatch = async () => {
    if (!jobDescription.trim()) return;
    setLoading(true);
    try {
      const res = await matchCandidates(jobDescription, model);
      setResults(res.data);
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Matching failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Job Match</h1>

      <div className="glass-card p-5 space-y-4">
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Paste a job description here..."
          rows={6}
          className="glass-input w-full resize-none"
        />
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <ModelSelector selectedModel={model} onModelChange={setModel} />
          </div>
          <button onClick={handleMatch} disabled={loading || !jobDescription.trim()} className="glass-button glass-button-primary">
            {loading ? 'Matching...' : 'Find Candidates'}
          </button>
        </div>
      </div>

      {/* Match results */}
      {results?.candidates?.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Top Candidates</h2>
          {results.candidates.map((c, i) => (
            <div key={i} className="glass-card p-5 flex gap-5 items-start animate-slide-up" style={{ animationDelay: `${i * 100}ms` }}>
              <CircleProgress score={c.match_score} />
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold">{c.candidate_name}</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{c.explanation}</p>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {c.skills_matched?.map((s) => <SkillBadge key={s} skill={s} matched={true} />)}
                  {c.skills_missing?.map((s) => <SkillBadge key={s} skill={s} matched={false} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Analytics */}
      {(skillsData.length > 0 || expData.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {skillsData.length > 0 && (
            <div className="glass-card p-5">
              <h3 className="text-lg font-semibold mb-4">Top Skills Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={skillsData.slice(0, 15)} layout="vertical">
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="skill" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {expData.length > 0 && (
            <div className="glass-card p-5">
              <h3 className="text-lg font-semibold mb-4">Experience Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={expData} dataKey="count" nameKey="range" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}>
                    {expData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const SKILL_CATEGORIES = {
  languages: { color: 'bg-blue-500/20 text-blue-400 border-blue-500/30', keywords: ['python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'sql', 'html', 'css', 'r', 'scala', 'perl'] },
  frameworks: { color: 'bg-purple-500/20 text-purple-400 border-purple-500/30', keywords: ['react', 'angular', 'vue', 'next.js', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'tailwind', 'bootstrap', 'sass', 'laravel', '.net'] },
  databases: { color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30', keywords: ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'nosql', 'dynamodb', 'cassandra', 'oracle'] },
  cloud: { color: 'bg-amber-500/20 text-amber-400 border-amber-500/30', keywords: ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd', 'heroku', 'vercel', 'netlify'] },
  tools: { color: 'bg-slate-500/20 text-slate-400 border-slate-500/30', keywords: ['git', 'linux', 'agile', 'scrum', 'rest', 'graphql', 'microservices', 'kafka', 'rabbitmq', 'nginx', 'jira'] },
};

function getSkillColor(skill) {
  const lower = skill.toLowerCase();
  for (const [, cat] of Object.entries(SKILL_CATEGORIES)) {
    if (cat.keywords.some((k) => lower.includes(k))) return cat.color;
  }
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
}

export default function SkillBadge({ skill, matched }) {
  const color = matched === true
    ? 'bg-green-500/20 text-green-400 border-green-500/30'
    : matched === false
    ? 'bg-red-500/20 text-red-400 border-red-500/30'
    : getSkillColor(skill);

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${color} transition-all`}>
      {skill}
    </span>
  );
}

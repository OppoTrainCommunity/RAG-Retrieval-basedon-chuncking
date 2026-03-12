import SkillBadge from './SkillBadge';
import { HiOutlineTrash, HiOutlineMail, HiOutlinePhone, HiOutlineAcademicCap } from 'react-icons/hi';

export default function CVCard({ cv, onDelete }) {
  return (
    <div className="glass-card p-5 animate-slide-up hover:shadow-xl transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold">{cv.candidate_name}</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {cv.years_of_experience} years experience
          </p>
        </div>
        <button
          onClick={() => onDelete(cv.cv_id)}
          className="p-2 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/30 text-red-500 transition-colors"
          title="Delete CV"
        >
          <HiOutlineTrash className="w-4 h-4" />
        </button>
      </div>

      {cv.summary && (
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-3 line-clamp-2">{cv.summary}</p>
      )}

      {cv.skills?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {cv.skills.slice(0, 12).map((skill) => (
            <SkillBadge key={skill} skill={skill} />
          ))}
          {cv.skills.length > 12 && (
            <span className="text-xs text-gray-500 self-center">+{cv.skills.length - 12} more</span>
          )}
        </div>
      )}

      <div className="flex flex-col gap-1 text-xs text-gray-500 dark:text-gray-400">
        {cv.education?.length > 0 && (
          <div className="flex items-center gap-1">
            <HiOutlineAcademicCap className="w-3.5 h-3.5" />
            <span>{cv.education[0]?.degree} — {cv.education[0]?.institution}</span>
          </div>
        )}
        {cv.email && (
          <div className="flex items-center gap-1">
            <HiOutlineMail className="w-3.5 h-3.5" />
            <span>{cv.email}</span>
          </div>
        )}
        {cv.phone && (
          <div className="flex items-center gap-1">
            <HiOutlinePhone className="w-3.5 h-3.5" />
            <span>{cv.phone}</span>
          </div>
        )}
      </div>

      <div className="mt-3 text-xs text-gray-400">
        {cv.filename}
      </div>
    </div>
  );
}

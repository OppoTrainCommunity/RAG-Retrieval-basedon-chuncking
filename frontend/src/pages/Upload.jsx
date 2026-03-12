import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import { HiOutlineCloudUpload, HiOutlineDocument } from 'react-icons/hi';
import { uploadFiles, getCVs, deleteCV, getStats } from '../services/api';
import CVCard from '../components/CVCard';

export default function Upload() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [cvs, setCvs] = useState([]);
  const [stats, setStats] = useState(null);

  const loadData = useCallback(async () => {
    try {
      const [cvsRes, statsRes] = await Promise.all([getCVs(), getStats()]);
      setCvs(cvsRes.data.cvs);
      setStats(statsRes.data);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const onDrop = useCallback((accepted) => {
    const valid = accepted.filter((f) => {
      if (f.type !== 'application/pdf') { toast.error(`${f.name}: Only PDF files allowed`); return false; }
      if (f.size > 20 * 1024 * 1024) { toast.error(`${f.name}: File too large (max 20MB)`); return false; }
      return true;
    });
    setFiles((prev) => [...prev, ...valid]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'application/pdf': ['.pdf'] }, multiple: true });

  const removeFile = (index) => setFiles((prev) => prev.filter((_, i) => i !== index));

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    try {
      const res = await uploadFiles(files);
      const results = res.data.results;
      const success = results.filter((r) => r.status === 'success').length;
      const failed = results.filter((r) => r.status !== 'success').length;
      if (success) toast.success(`${success} CV(s) uploaded successfully`);
      if (failed) toast.error(`${failed} CV(s) failed`);
      setFiles([]);
      loadData();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (cvId) => {
    try {
      await deleteCV(cvId);
      toast.success('CV deleted');
      loadData();
    } catch {
      toast.error('Delete failed');
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold">Upload CVs</h1>

      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-3 gap-4">
          <div className="glass-card p-4 text-center">
            <p className="text-2xl font-bold text-blue-500">{stats.total_cvs}</p>
            <p className="text-sm text-gray-500">Total CVs</p>
          </div>
          <div className="glass-card p-4 text-center">
            <p className="text-2xl font-bold text-purple-500">{stats.total_chunks}</p>
            <p className="text-sm text-gray-500">Total Chunks</p>
          </div>
          <div className="glass-card p-4 text-center">
            <p className="text-2xl font-bold text-emerald-500">{stats.top_skills?.[0]?.skill || '—'}</p>
            <p className="text-sm text-gray-500">Top Skill</p>
          </div>
        </div>
      )}

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`glass-card p-10 border-2 border-dashed cursor-pointer text-center transition-all
          ${isDragActive ? 'border-blue-500 bg-blue-500/5' : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'}`}
      >
        <input {...getInputProps()} />
        <HiOutlineCloudUpload className="w-12 h-12 mx-auto mb-3 text-blue-500" />
        <p className="font-medium">{isDragActive ? 'Drop PDF files here...' : 'Drag & drop PDF files here, or click to browse'}</p>
        <p className="text-sm text-gray-500 mt-1">PDF only · Max 20MB each</p>
      </div>

      {/* Staged files */}
      {files.length > 0 && (
        <div className="glass-card p-4 space-y-2">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">{files.length} file(s) ready</h3>
            <button onClick={handleUpload} disabled={uploading} className="glass-button glass-button-primary">
              {uploading ? 'Uploading...' : 'Upload All'}
            </button>
          </div>
          {files.map((file, i) => (
            <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-gray-50 dark:bg-gray-800/50">
              <div className="flex items-center gap-2">
                <HiOutlineDocument className="w-4 h-4 text-blue-500" />
                <span className="text-sm">{file.name}</span>
                <span className="text-xs text-gray-500">({(file.size / 1024).toFixed(0)} KB)</span>
              </div>
              <button onClick={() => removeFile(i)} className="text-xs text-red-500 hover:text-red-400">Remove</button>
            </div>
          ))}
        </div>
      )}

      {/* CV grid */}
      {cvs.length > 0 && (
        <>
          <h2 className="text-lg font-semibold">Uploaded CVs ({cvs.length})</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cvs.map((cv) => (
              <CVCard key={cv.cv_id} cv={cv} onDelete={handleDelete} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { CheckCircle2, ChevronRight, FileVideo, Loader2, Upload, XCircle } from 'lucide-react';
import Shell from './Shell';
import { api } from '@/api';
import type { RunSummary } from '@/types';
import { StatusBadge } from './StatusBadge';

export default function UploadScreen() {
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    api.runs().then(setRuns).catch(() => setRuns([]));
  }, []);

  const upload = async (f: File) => {
    setFile(f);
    setError(null);
    setProgress(0);
    try {
      const { id } = await api.upload(f, setProgress);
      navigate(`/runs/${id}`);
    } catch (e) {
      setError((e as Error).message);
      setFile(null);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  };

  return (
    <Shell>
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Surgical video documentation</h1>
          <p className="text-lg text-slate-600">
            Upload a laparoscopic cholecystectomy video. SurgiDoc drafts a report in which every
            claim is checked against the vision models, then you review it.
          </p>
        </div>

        <div className="card p-8">
          {file ? (
            <div className="text-center py-6">
              <Loader2 className="mx-auto mb-4 text-blue-600 animate-spin" size={56} />
              <h3 className="text-xl font-semibold text-slate-800 mb-1">Uploading {file.name}</h3>
              <p className="text-slate-500 mb-4">{(file.size / 2 ** 20).toFixed(1)} MB</p>
              <div className="w-full bg-slate-200 rounded-full h-3">
                <div className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                     style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
            </div>
          ) : (
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all duration-200 ${
                dragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}`}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => input.current?.click()}
            >
              <Upload className={`mx-auto mb-4 ${dragging ? 'text-blue-500' : 'text-slate-400'}`} size={64} />
              <h3 className="text-xl font-semibold text-slate-700 mb-2">Drop a video here or click to browse</h3>
              <p className="text-slate-500 mb-4">MP4, MOV, AVI or MKV, up to 10 minutes</p>
              <span className="btn-primary"><FileVideo size={20} />Select video</span>
              <input ref={input} type="file" accept="video/*" className="hidden"
                     onChange={e => e.target.files?.[0] && upload(e.target.files[0])} />
            </div>
          )}
          {error && (
            <div className="mt-4 flex items-start gap-2 text-red-700 bg-red-50 border border-red-200 rounded-lg p-3 text-sm">
              <XCircle size={18} className="shrink-0 mt-0.5" />{error}
            </div>
          )}
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <Info title="Checked claims" text="Every statement cites frames, events or a quoted guideline passage, and code verifies it." />
          <Info title="Swipe review" text="Keep or remove each claim, with its frame and evidence in front of you." />
          <Info title="Honest labels" text="Output without measured accuracy is marked unverified in the report." />
        </div>

        {runs.length > 0 && (
          <div className="mt-8">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">Recent runs</h2>
            <div className="card divide-y divide-slate-100">
              {runs.map(r => (
                <Link key={r.id} to={r.status === 'done' ? `/runs/${r.id}/report` : `/runs/${r.id}`}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50">
                  {r.status === 'done' ? <CheckCircle2 size={18} className="text-green-600" />
                    : <FileVideo size={18} className="text-slate-400" />}
                  <span className="font-medium text-slate-800 truncate">{r.label ?? r.id}</span>
                  <span className="text-xs text-slate-400 font-mono">{r.id}</span>
                  <span className="ml-auto"><StatusBadge status={r.status} /></span>
                  <ChevronRight size={18} className="text-slate-400" />
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </Shell>
  );
}

function Info({ title, text }: { title: string; text: string }) {
  return (
    <div className="card p-4 text-center">
      <h4 className="font-semibold text-slate-800 mb-1">{title}</h4>
      <p className="text-sm text-slate-600">{text}</p>
    </div>
  );
}

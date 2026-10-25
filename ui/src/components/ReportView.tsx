import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowLeft, Download, ExternalLink, Loader2, Printer } from 'lucide-react';
import Shell from './Shell';
import { api } from '@/api';
import type { Run } from '@/types';

export default function ReportView() {
  const { id = '' } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.run(id).then(setRun).catch(e => setError(e.message));
    api.report(id).then(setMarkdown).catch(() => setError('No report: this run is not approved yet.'));
  }, [id]);

  const download = () => {
    if (!markdown) return;
    const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `surgidoc-${id}.md` });
    a.click();
    URL.revokeObjectURL(url);
  };

  const actions = markdown && (
    <>
      {run?.trace_url && (
        <a href={run.trace_url} target="_blank" rel="noreferrer" className="btn-secondary hidden sm:inline-flex">
          Trace <ExternalLink size={16} />
        </a>
      )}
      <button className="btn-secondary" onClick={() => window.print()}><Printer size={18} /></button>
      <button className="btn-primary" onClick={download}><Download size={18} />Markdown</button>
    </>
  );

  return (
    <Shell actions={actions} title={<span className="font-semibold">{run?.label ?? id}</span>}>
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-slate-600 hover:text-blue-700 mb-4">
        <ArrowLeft size={16} />New video
      </Link>
      {error && !markdown && <p className="card p-8 text-center text-slate-600">{error}</p>}
      {!error && !markdown && <Loader2 className="mx-auto text-blue-600 animate-spin" size={48} />}
      {markdown && run?.context && run.context.key_frames.length > 0 && (
        <div className="card p-4 mb-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">Key frames</div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            {run.context.key_frames.map(f => (
              <a key={f} href={api.frameUrl(id, f)} target="_blank" rel="noreferrer"
                 className="block rounded-md overflow-hidden ring-1 ring-slate-200 bg-black">
                <img src={api.frameUrl(id, f)} alt={`Frame ${f}`} loading="lazy" className="w-full aspect-video object-cover" />
                <div className="bg-white px-2 py-0.5 text-xs text-slate-600">
                  Frame {f} · {run.context!.frame_times[f]?.toFixed(1)} s
                </div>
              </a>
            ))}
          </div>
        </div>
      )}
      {markdown && (
        <article className={'card p-6 sm:p-10 prose prose-slate max-w-none prose-table:text-sm '
          // The Markdown already has its own quote marks and backticks; drop the plugin's.
          + 'prose-blockquote:font-normal prose-blockquote:not-italic '
          + '[&_blockquote_p]:before:content-none [&_blockquote_p]:after:content-none '
          + 'prose-code:before:content-none prose-code:after:content-none prose-code:font-normal'}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
        </article>
      )}
    </Shell>
  );
}

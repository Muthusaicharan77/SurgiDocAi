import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  AlertTriangle, CheckCircle2, ExternalLink, Layers, Loader2, Quote, RotateCcw, Trash2, XCircle,
} from 'lucide-react';
import Shell from './Shell';
import { StatusBadge } from './StatusBadge';
import { api } from '@/api';
import { CLAIM_TYPES, PHASE_COLORS, PHASE_NAMES, STEPS, basisNote, frameSpans } from '@/labels';
import { useReview } from '@/store';
import type { Claim, ClaimType, Run, RunContext } from '@/types';

export default function RunView() {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => api.run(id).then(setRun).catch(e => setError(e.message)), [id]);
  useEffect(() => { load(); }, [load]);

  // Live progress while the pipeline runs; reload when it pauses or ends.
  useEffect(() => {
    if (!run || !['queued', 'running'].includes(run.status)) return;
    // The stream replays every event of the run first, so it replaces the list.
    setRun(r => r && { ...r, nodes_done: [] });
    return api.events(
      id,
      node => setRun(r => r && { ...r, nodes_done: [...r.nodes_done, node] }),
      () => load(),
    );
  }, [id, run?.status, load]);

  // After the review is submitted, wait for the report.
  useEffect(() => {
    if (run?.status === 'reviewing') {
      const t = setTimeout(load, 1000);
      return () => clearTimeout(t);
    }
    if (run?.status === 'done' && run.approved) navigate(`/runs/${id}/report`, { replace: true });
  }, [run, id, load, navigate]);

  if (error) return <Shell><ErrorCard message={error} /></Shell>;
  if (!run) return <Shell><Loader2 className="mx-auto text-blue-600 animate-spin" size={48} /></Shell>;

  const title = (
    <span className="inline-flex items-center gap-2">
      <span className="font-semibold text-slate-800 truncate">{run.label ?? run.id}</span>
      <StatusBadge status={run.status} />
    </span>
  );

  return (
    <Shell title={title}>
      {(run.status === 'queued' || run.status === 'running') && <Progress run={run} />}
      {run.status === 'failed' && <ErrorCard message={run.error ?? 'The run failed.'} />}
      {run.status === 'reviewing' && (
        <div className="card p-12 text-center">
          <Loader2 className="mx-auto mb-4 text-blue-600 animate-spin" size={48} />
          <h2 className="text-xl font-semibold">Writing the report…</h2>
        </div>
      )}
      {run.status === 'done' && !run.approved && (
        <div className="card p-12 text-center">
          <XCircle className="mx-auto mb-4 text-slate-400" size={48} />
          <h2 className="text-xl font-semibold mb-1">Report not approved</h2>
          <p className="text-slate-600">The reviewer rejected this draft, so no report was written.</p>
        </div>
      )}
      {run.status === 'awaiting_review' && run.review && run.context && (
        <Overview run={run} ctx={run.context} onSubmitted={load} />
      )}
    </Shell>
  );
}

function Progress({ run }: { run: Run }) {
  const [start] = useState(() => Date.now());
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick(n => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const done = new Set(run.nodes_done);
  const drafts = run.nodes_done.filter(n => n === 'draft_report').length;
  const steps = STEPS.filter(s => s.node !== 'extract_frames' || run.nodes_done.includes('extract_frames')
    || run.nodes_done.length === 0);
  const current = steps.find(s => !done.has(s.node))?.node;
  const elapsed = Math.round((Date.now() - start) / 1000);

  return (
    <div className="max-w-2xl mx-auto card p-8">
      <div className="text-center mb-6">
        <Loader2 className="mx-auto mb-3 text-blue-600 animate-spin" size={48} />
        <h2 className="text-2xl font-semibold text-slate-900">
          {run.status === 'queued' ? 'Waiting in the queue' : 'Analysing the video'}
        </h2>
        <p className="text-slate-500">
          {run.status === 'queued' && run.queue_position
            ? `${run.queue_position} run(s) ahead of this one · runs are processed one at a time`
            : `${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, '0')} on this page`}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          On CPU, a short clip takes under a minute; the first run after a server start also loads the models.
        </p>
      </div>
      <ol className="space-y-3">
        {steps.map(s => {
          const isDone = done.has(s.node) && s.node !== current;
          const active = s.node === current;
          return (
            <li key={s.node} className="flex items-start gap-3">
              {isDone ? <CheckCircle2 className="text-green-600 shrink-0 mt-0.5" size={20} />
                : active ? <Loader2 className="text-blue-600 shrink-0 mt-0.5 animate-spin" size={20} />
                  : <span className="w-5 h-5 rounded-full border-2 border-slate-300 shrink-0 mt-0.5" />}
              <div>
                <div className={isDone ? 'text-green-700 font-medium' : active ? 'text-blue-700 font-medium' : 'text-slate-500'}>
                  {s.label}
                  {s.node === 'draft_report' && drafts > 1 && (
                    <span className="ml-2 badge bg-amber-100 text-amber-800">draft {drafts}: retry after the checks</span>
                  )}
                </div>
                <div className="text-xs text-slate-500">{s.detail}</div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

const SECTIONS: [string, ClaimType[]][] = [
  ['Observations', ['phase', 'structures_visible']],
  ['Events', ['event']],
  ['Guidance', ['guideline']],
  ['Limitations', ['limitation']],
];

function Overview({ run, ctx, onSubmitted }: { run: Run; ctx: RunContext; onSubmitted: () => void }) {
  const review = run.review!;
  const removed = useReview(s => s.removed[run.id] ?? []);
  const toggle = useReview(s => s.toggle);
  const notes = useReview(s => s.notes[run.id] ?? '');
  const setNotes = useReview(s => s.setNotes);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const times = ctx.frame_times;

  const submit = async (approve: boolean) => {
    setSubmitting(true);
    setError(null);
    try {
      await api.review(run.id, { approve, remove: approve ? removed : [], notes });
      onSubmitted();
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 pb-28">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Frames analysed" value={times.length} />
        <Stat label="Footage" value={`${times[0]?.toFixed(1)}-${times[times.length - 1]?.toFixed(1)} s`} />
        <Stat label="Claims to review" value={review.claims.length} />
        <Stat label="Drafts" value={ctx.drafts} hint={ctx.drafts > 1 ? 'retried after the checks' : undefined} />
      </div>

      <section className="card p-5">
        <h2 className="font-semibold text-slate-900 mb-3">Phase timeline <span className="text-slate-400 font-normal text-sm">(model estimate)</span></h2>
        <div className="flex h-8 rounded-lg overflow-hidden ring-1 ring-slate-200">
          {ctx.timeline.segments.map((s, i) => (
            <div key={i} title={`${PHASE_NAMES[s.phase] ?? s.phase}: ${s.start_s.toFixed(1)}-${s.end_s.toFixed(1)} s`}
                 className={`${PHASE_COLORS[s.phase] ?? 'bg-slate-400'} border-r border-white last:border-0`}
                 style={{ flexGrow: s.end_frame - s.start_frame + 1 }} />
          ))}
        </div>
        <div className="flex flex-wrap gap-3 mt-3 text-xs text-slate-600">
          {[...new Set(ctx.timeline.segments.map(s => s.phase))].map(p => (
            <span key={p} className="inline-flex items-center gap-1.5">
              <span className={`w-3 h-3 rounded-sm ${PHASE_COLORS[p] ?? 'bg-slate-400'}`} />{PHASE_NAMES[p] ?? p}
            </span>
          ))}
          {ctx.timeline.gaps.length > 0 && <span className="text-slate-400">· {ctx.timeline.gaps.length} gap(s) in the footage</span>}
        </div>
      </section>

      <section className="card p-5">
        <h2 className="font-semibold text-slate-900 mb-3">Key frames</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {ctx.key_frames.map(f => (
            <a key={f} href={api.frameUrl(run.id, f)} target="_blank" rel="noreferrer"
               className="group block rounded-lg overflow-hidden ring-1 ring-slate-200 bg-black">
              <img src={api.frameUrl(run.id, f)} alt={`Frame ${f}`} loading="lazy"
                   className="w-full aspect-video object-cover group-hover:opacity-90" />
              <div className="bg-white px-2 py-1 text-xs text-slate-600">Frame {f} · {times[f]?.toFixed(1)} s</div>
            </a>
          ))}
        </div>
      </section>

      <section className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="font-semibold text-slate-900">Claims</h2>
            <p className="text-sm text-slate-500">All passed the code verifier and the wording check. Remove any that are wrong.</p>
          </div>
          <Link to={`/runs/${run.id}/swipe`} className="btn-primary"><Layers size={18} />Review one by one</Link>
        </div>
        {SECTIONS.map(([title, types]) => {
          const items = review.claims.filter(c => types.includes(c.claim_type));
          if (!items.length) return null;
          return (
            <div key={title} className="mb-5 last:mb-0">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">{title}</h3>
              <ul className="space-y-2">
                {items.map(c => (
                  <ClaimRow key={c.id} claim={c} ctx={ctx} removed={removed.includes(c.id)}
                            onToggle={() => toggle(run.id, c.id)} />
                ))}
              </ul>
            </div>
          );
        })}
        {review.dropped.length > 0 && (
          <details className="mt-4 text-sm">
            <summary className="cursor-pointer text-slate-600">
              {review.dropped.length} claim(s) dropped by the checks after the last retry
            </summary>
            <ul className="mt-2 space-y-1 text-slate-500">
              {review.dropped.map(d => (
                <li key={d.claim.id}><span className="line-through">{d.claim.text}</span> — {d.reasons.join('; ')}</li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <div className="fixed bottom-0 inset-x-0 bg-white/95 backdrop-blur border-t border-slate-200 shadow-lg">
        <div className="max-w-6xl mx-auto px-4 py-3 flex flex-col md:flex-row gap-3 md:items-center">
          <input className="input md:flex-1" placeholder="Reviewer notes (optional)" value={notes}
                 onChange={e => setNotes(run.id, e.target.value)} />
          {error && <span className="text-sm text-red-600">{error}</span>}
          <div className="flex gap-2">
            <button className="btn-secondary" disabled={submitting} onClick={() => submit(false)}>Reject draft</button>
            <button className="btn-success" disabled={submitting} onClick={() => submit(true)}>
              {submitting ? <Loader2 size={18} className="animate-spin" /> : <CheckCircle2 size={18} />}
              Approve{removed.length ? ` (${removed.length} removed)` : ''}
            </button>
          </div>
          {run.trace_url && (
            <a href={run.trace_url} target="_blank" rel="noreferrer" className="text-sm text-slate-500 hover:text-blue-700 inline-flex items-center gap-1">
              Trace <ExternalLink size={14} />
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export function ClaimRow({ claim, ctx, removed, onToggle }: {
  claim: Claim; ctx: RunContext; removed: boolean; onToggle: () => void;
}) {
  const type = CLAIM_TYPES[claim.claim_type];
  const note = basisNote(claim.claim_type, claim.basis);
  const span = claim.frame_ids.length ? frameSpans(claim.frame_ids, ctx.frame_times)
    : claim.event_ids.map(e => {
      const ev = ctx.timeline.events.find(x => x.id === e);
      return ev ? `${e}: ${ev.start_s.toFixed(1)}-${ev.end_s.toFixed(1)} s` : e;
    }).join('; ');
  return (
    <li className={`flex gap-3 items-start rounded-lg border p-3 transition-colors ${
      removed ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}>
      <span className={`badge ${type.badge} shrink-0 mt-0.5`}>{type.label}</span>
      <div className="flex-1 min-w-0">
        <p className={removed ? 'line-through text-slate-400' : 'text-slate-800'}>{claim.text}</p>
        <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1 text-xs text-slate-500">
          {span && <span>{span}</span>}
          {note && <span className="inline-flex items-center gap-1 text-amber-700"><AlertTriangle size={12} />{note}</span>}
          {claim.passage_ids.map(p => (
            <a key={p} href={ctx.passages[p]?.url} target="_blank" rel="noreferrer" className="hover:text-blue-700">
              {ctx.passages[p]?.doc_title.slice(0, 60) ?? p}…
            </a>
          ))}
        </div>
        {claim.quote && (
          <blockquote className="mt-2 text-sm text-slate-600 border-l-2 border-emerald-300 pl-3 flex gap-2">
            <Quote size={14} className="shrink-0 mt-1 text-emerald-500" />{claim.quote}
          </blockquote>
        )}
      </div>
      <button onClick={onToggle} title={removed ? 'Keep this claim' : 'Remove this claim'}
              className={`btn shrink-0 px-2 py-1 ${removed ? 'text-slate-600 hover:bg-slate-100' : 'text-red-600 hover:bg-red-50'}`}>
        {removed ? <RotateCcw size={16} /> : <Trash2 size={16} />}
      </button>
    </li>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="card p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="text-xs text-amber-700">{hint}</div>}
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="max-w-2xl mx-auto card p-8 text-center">
      <XCircle className="mx-auto mb-3 text-red-600" size={48} />
      <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
      <p className="text-slate-600 font-mono text-sm break-words">{message}</p>
      <Link to="/" className="btn-secondary mt-6">Back</Link>
    </div>
  );
}

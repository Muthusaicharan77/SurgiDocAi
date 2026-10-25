import { useEffect, useState } from 'react';
import { motion, useMotionValue, useTransform, type PanInfo } from 'framer-motion';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { AlertTriangle, Check, CheckCircle2, List, Loader2, Quote, ShieldQuestion, Undo2, X } from 'lucide-react';
import Shell from './Shell';
import { api } from '@/api';
import { CLAIM_TYPES, PHASE_NAMES, basisNote, claimFrame, frameSpans } from '@/labels';
import { useReview } from '@/store';
import type { Claim, Run, RunContext } from '@/types';

type Decision = { id: string; keep: boolean };

export default function SwipeReview() {
  const { id = '' } = useParams();
  const navigate = useNavigate();
  const [run, setRun] = useState<Run | null>(null);
  const [index, setIndex] = useState(0);
  const [history, setHistory] = useState<Decision[]>([]);
  const removed = useReview(s => s.removed[id] ?? []);
  const setRemoved = useReview(s => s.setRemoved);
  const notes = useReview(s => s.notes[id] ?? '');
  const setNotes = useReview(s => s.setNotes);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { api.run(id).then(setRun).catch(e => setError(e.message)); }, [id]);

  const claims = run?.review?.claims ?? [];
  const ctx = run?.context;
  const current = claims[index];
  const finished = claims.length > 0 && index >= claims.length;

  const decide = (keep: boolean) => {
    if (!current) return;
    const others = removed.filter(x => x !== current.id);
    setRemoved(id, keep ? others : [...others, current.id]);
    setHistory(h => [...h, { id: current.id, keep }]);
    setIndex(i => i + 1);
  };

  const undo = () => {
    const last = history[history.length - 1];
    if (!last) return;
    setRemoved(id, removed.filter(x => x !== last.id));
    setHistory(h => h.slice(0, -1));
    setIndex(i => i - 1);
  };

  const submit = async (approve: boolean) => {
    setSubmitting(true);
    try {
      await api.review(id, { approve, remove: approve ? removed : [], notes });
      navigate(`/runs/${id}`);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  };

  if (error) return <Shell><p className="text-center text-red-600">{error}</p></Shell>;
  if (!run || !ctx) return <Shell><Loader2 className="mx-auto text-blue-600 animate-spin" size={48} /></Shell>;
  if (run.status !== 'awaiting_review') {
    return <Shell><p className="text-center">This run is not awaiting review. <Link className="text-blue-700" to={`/runs/${id}`}>Open it</Link></p></Shell>;
  }

  const progress = claims.length ? Math.min(index, claims.length) / claims.length : 0;
  const actions = <Link to={`/runs/${id}`} className="btn-secondary"><List size={18} />All claims</Link>;
  const title = <span className="text-sm text-slate-600">Claim {Math.min(index + 1, claims.length)} of {claims.length}</span>;

  return (
    <Shell actions={actions} title={title}>
      <div className="max-w-xl mx-auto">
        <div className="w-full bg-slate-200 rounded-full h-2 mb-6">
          <div className="bg-blue-600 h-2 rounded-full transition-all duration-300" style={{ width: `${progress * 100}%` }} />
        </div>

        {finished ? (
          <div className="card p-8 text-center">
            <CheckCircle2 className="mx-auto mb-3 text-green-600" size={56} />
            <h2 className="text-2xl font-bold mb-1">All claims reviewed</h2>
            <p className="text-slate-600 mb-6">
              {claims.length - removed.length} kept, {removed.length} removed.
            </p>
            {removed.length > 0 && (
              <ul className="text-left text-sm text-slate-500 mb-6 space-y-1">
                {claims.filter(c => removed.includes(c.id)).map(c => (
                  <li key={c.id} className="line-through">{c.text}</li>
                ))}
              </ul>
            )}
            <textarea className="input mb-4" rows={3} placeholder="Reviewer notes (optional)"
                      value={notes} onChange={e => setNotes(id, e.target.value)} />
            <div className="flex gap-3 justify-center">
              <button className="btn-secondary" onClick={undo}><Undo2 size={18} />Back</button>
              <button className="btn-secondary" disabled={submitting} onClick={() => submit(false)}>Reject draft</button>
              <button className="btn-success" disabled={submitting} onClick={() => submit(true)}>
                {submitting ? <Loader2 size={18} className="animate-spin" /> : <Check size={18} />}Approve report
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="relative h-[560px] mb-6 flex items-start justify-center">
              {claims.slice(index + 1, index + 3).reverse().map((c, i, arr) => {
                const depth = arr.length - i;
                return (
                  <div key={c.id} className="absolute w-full" style={{
                    transform: `scale(${1 - depth * 0.04}) translateY(${depth * 12}px)`,
                    opacity: 1 - depth * 0.3, zIndex: 10 - depth,
                  }}>
                    <div className="card h-[520px] shadow-xl" />
                  </div>
                );
              })}
              {current && <SwipeCard key={current.id} claim={current} ctx={ctx} runId={id} onDecide={decide} />}
            </div>

            <div className="flex items-center justify-center gap-6 mb-4">
              <button onClick={() => decide(false)} title="Remove (←)"
                      className="w-20 h-20 rounded-full bg-red-500 text-white shadow-lg hover:bg-red-600 hover:scale-110 active:scale-95 transition-all flex items-center justify-center">
                <X size={40} />
              </button>
              <button onClick={undo} disabled={!history.length} title="Undo"
                      className="w-14 h-14 rounded-full bg-slate-300 text-slate-700 shadow-lg hover:bg-slate-400 hover:scale-110 active:scale-95 transition-all flex items-center justify-center disabled:opacity-40">
                <Undo2 size={24} />
              </button>
              <button onClick={() => decide(true)} title="Keep (→)"
                      className="w-20 h-20 rounded-full bg-green-500 text-white shadow-lg hover:bg-green-600 hover:scale-110 active:scale-95 transition-all flex items-center justify-center">
                <Check size={40} />
              </button>
            </div>
            <p className="text-center text-slate-600">
              <b>Swipe right</b> or → to keep, <b>swipe left</b> or ← to remove
            </p>
          </>
        )}
      </div>
    </Shell>
  );
}

function SwipeCard({ claim, ctx, runId, onDecide }: {
  claim: Claim; ctx: RunContext; runId: string; onDecide: (keep: boolean) => void;
}) {
  const x = useMotionValue(0);
  const rotate = useTransform(x, [-200, 200], [-20, 20]);
  const keepOpacity = useTransform(x, [0, 60, 160], [0, 0.6, 1]);
  const removeOpacity = useTransform(x, [-160, -60, 0], [1, 0.6, 0]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') onDecide(true);
      if (e.key === 'ArrowLeft') onDecide(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onDecide]);

  const onDragEnd = (_: unknown, info: PanInfo) => {
    if (info.offset.x > 120) onDecide(true);
    else if (info.offset.x < -120) onDecide(false);
  };

  const type = CLAIM_TYPES[claim.claim_type];
  const note = basisNote(claim.claim_type, claim.basis);
  const frame = claimFrame(claim.frame_ids, claim.event_ids, ctx);
  const evidence = claim.frame_ids.length ? frameSpans(claim.frame_ids, ctx.frame_times)
    : claim.event_ids.map(e => {
      const ev = ctx.timeline.events.find(x => x.id === e);
      return ev ? `${e}: ${frameSpans(Array.from({ length: ev.end_frame - ev.start_frame + 1 }, (_, i) => ev.start_frame + i), ctx.frame_times)}` : e;
    }).join('; ');

  return (
    <motion.div className="absolute w-full cursor-grab active:cursor-grabbing" style={{ x, rotate, zIndex: 100 }}
                drag="x" dragConstraints={{ left: 0, right: 0 }} onDragEnd={onDragEnd}>
      <div className="card h-[520px] overflow-hidden shadow-2xl flex flex-col">
        <div className="relative h-60 bg-slate-900 shrink-0">
          {frame !== null ? (
            <img src={api.frameUrl(runId, frame)} alt={`Frame ${frame}`} draggable={false}
                 className="w-full h-full object-cover" />
          ) : claim.claim_type === 'guideline' ? (
            <div className="h-full bg-gradient-to-br from-emerald-600 to-teal-700 text-white p-6 flex flex-col justify-center">
              <Quote size={28} className="mb-2 opacity-70" />
              <p className="text-lg leading-snug line-clamp-5">{claim.quote}</p>
            </div>
          ) : (
            <div className="h-full bg-gradient-to-br from-amber-500 to-orange-600 text-white flex flex-col items-center justify-center">
              <ShieldQuestion size={56} className="mb-2" />
              <span className="font-semibold">For the surgeon to confirm</span>
            </div>
          )}
          <span className={`badge ${type.badge} absolute top-3 left-3 shadow`}>{type.label}</span>
          {frame !== null && (
            <span className="badge bg-black/60 text-white absolute top-3 right-3">
              Frame {frame} · {ctx.frame_times[frame]?.toFixed(1)} s
            </span>
          )}
          <motion.div style={{ opacity: keepOpacity }} className="absolute inset-0 flex items-center justify-end p-6">
            <span className="bg-green-500 text-white font-bold text-2xl px-5 py-2 rounded-lg border-4 border-white rotate-12">KEEP</span>
          </motion.div>
          <motion.div style={{ opacity: removeOpacity }} className="absolute inset-0 flex items-center justify-start p-6">
            <span className="bg-red-500 text-white font-bold text-2xl px-5 py-2 rounded-lg border-4 border-white -rotate-12">REMOVE</span>
          </motion.div>
        </div>

        <div className="p-5 flex-1 overflow-y-auto">
          <p className="text-lg text-slate-900 leading-snug mb-3">{claim.text}</p>
          {note && (
            <p className="flex items-center gap-1.5 text-sm text-amber-700 mb-2"><AlertTriangle size={14} />{note}</p>
          )}
          {evidence && <p className="text-sm text-slate-500 mb-2">Evidence: {evidence}</p>}
          {claim.phase && <p className="text-sm text-slate-500 mb-2">Phase: {PHASE_NAMES[claim.phase] ?? claim.phase}</p>}
          {claim.structures.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-2">
              {claim.structures.map(s => (
                <span key={s} className="badge bg-slate-100 text-slate-700">{s.replace(/_/g, ' ')}</span>
              ))}
            </div>
          )}
          {claim.claim_type === 'guideline' && (
            <>
              {frame !== null && claim.quote && (
                <blockquote className="text-sm text-slate-600 border-l-2 border-emerald-300 pl-3 mb-2">{claim.quote}</blockquote>
              )}
              <p className="text-xs text-slate-500">
                Compare the claim with the quote: it may only say what the quote says.
              </p>
              {claim.passage_ids.map(p => (
                <a key={p} href={ctx.passages[p]?.url} target="_blank" rel="noreferrer"
                   className="block text-xs text-blue-700 hover:underline mt-1">
                  {ctx.passages[p]?.doc_title} — {ctx.passages[p]?.section}
                </a>
              ))}
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}

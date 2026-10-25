import type { RunStatus } from '@/types';

const STYLES: Record<RunStatus, [string, string]> = {
  queued: ['Queued', 'bg-slate-100 text-slate-700'],
  running: ['Running', 'bg-blue-100 text-blue-800'],
  awaiting_review: ['Awaiting review', 'bg-amber-100 text-amber-800'],
  reviewing: ['Finishing', 'bg-blue-100 text-blue-800'],
  done: ['Done', 'bg-green-100 text-green-800'],
  failed: ['Failed', 'bg-red-100 text-red-800'],
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const [label, style] = STYLES[status] ?? [status, 'bg-slate-100 text-slate-700'];
  return <span className={`badge ${style}`}>{label}</span>;
}

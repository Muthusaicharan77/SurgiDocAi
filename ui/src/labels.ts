import type { Basis, ClaimType, RunContext } from './types';

/** The pipeline's nodes, in order, as the progress screen shows them. */
export const STEPS: { node: string; label: string; detail: string }[] = [
  { node: 'extract_frames', label: 'Extract frames', detail: '5 frames per second from the video' },
  { node: 'perceive', label: 'Segment and recognise phase', detail: 'U-Net segmentation, Trans-SVNet phase model' },
  { node: 'build_timeline', label: 'Build timeline', detail: 'Phase segments and instrument events' },
  { node: 'select_key_frames', label: 'Select key frames', detail: 'Claude inspects frames with tools' },
  { node: 'retrieve', label: 'Retrieve guidelines', detail: 'Dense search and cross-encoder reranking' },
  { node: 'draft_report', label: 'Draft claims', detail: 'Claude writes the report as checkable claims' },
  { node: 'verify', label: 'Verify claims', detail: 'Code checks every claim against the evidence' },
  { node: 'judge', label: 'Check wording', detail: 'A second model flags claims that say too much' },
  { node: 'drop_unsupported', label: 'Prepare review', detail: 'Unsupported claims are removed' },
];

export const PHASE_NAMES: Record<string, string> = {
  preparation: 'Preparation',
  calot_triangle_dissection: "Calot's triangle dissection",
  clipping_cutting: 'Clipping and cutting',
  gallbladder_dissection: 'Gallbladder dissection',
  gallbladder_packaging: 'Gallbladder packaging',
  cleaning_coagulation: 'Cleaning and coagulation',
  gallbladder_retraction: 'Gallbladder retraction',
};

export const PHASE_COLORS: Record<string, string> = {
  preparation: 'bg-sky-400',
  calot_triangle_dissection: 'bg-indigo-500',
  clipping_cutting: 'bg-violet-500',
  gallbladder_dissection: 'bg-emerald-500',
  gallbladder_packaging: 'bg-amber-400',
  cleaning_coagulation: 'bg-rose-400',
  gallbladder_retraction: 'bg-teal-500',
};

export const CLAIM_TYPES: Record<ClaimType, { label: string; badge: string }> = {
  structures_visible: { label: 'Observation', badge: 'bg-blue-100 text-blue-800' },
  phase: { label: 'Phase', badge: 'bg-indigo-100 text-indigo-800' },
  event: { label: 'Event', badge: 'bg-cyan-100 text-cyan-800' },
  guideline: { label: 'Guidance', badge: 'bg-emerald-100 text-emerald-800' },
  limitation: { label: 'Limitation', badge: 'bg-amber-100 text-amber-800' },
};

/** The label the report would give the claim, from its evidence basis. */
export function basisNote(type: ClaimType, basis: Basis): string | null {
  if (type === 'phase') return 'Phase model estimate';
  if (basis === 'unvalidated_model') return 'Unverified: no measured accuracy';
  return null;
}

/** "frames 0-31 (22.4-28.6 s), 47 (35.0 s)", as in the report. */
export function frameSpans(ids: number[], times: number[]): string {
  const sorted = [...new Set(ids)].sort((a, b) => a - b);
  const runs: string[] = [];
  let start = 0;
  for (let i = 1; i <= sorted.length; i++) {
    if (i === sorted.length || sorted[i] !== sorted[i - 1] + 1) {
      const a = sorted[start], b = sorted[i - 1];
      runs.push(a === b ? `${a} (${times[a]?.toFixed(1)} s)`
        : `${a}-${b} (${times[a]?.toFixed(1)}-${times[b]?.toFixed(1)} s)`);
      start = i;
    }
  }
  return (sorted.length === 1 ? 'frame ' : 'frames ') + runs.join(', ');
}

/** The frame that best illustrates a claim: a key frame it cites, else its middle frame. */
export function claimFrame(frameIds: number[], eventIds: string[], ctx: RunContext): number | null {
  const ids = frameIds.length ? frameIds
    : ctx.timeline.events.filter(e => eventIds.includes(e.id)).map(e => e.start_frame);
  if (!ids.length) return null;
  return ids.find(i => ctx.key_frames.includes(i)) ?? [...ids].sort((a, b) => a - b)[Math.floor(ids.length / 2)];
}

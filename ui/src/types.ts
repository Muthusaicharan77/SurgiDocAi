// Shapes returned by the SurgiDoc API (src/surgidoc/serve/api.py).

export type ClaimType = 'structures_visible' | 'phase' | 'event' | 'guideline' | 'limitation';
export type Basis = 'validated_model' | 'unvalidated_model' | 'retrieved_source' | 'none';

export interface Claim {
  id: string;
  claim_type: ClaimType;
  text: string;
  frame_ids: number[];
  structures: string[];
  phase: string | null;
  event_ids: string[];
  passage_ids: string[];
  quote: string | null;
  basis: Basis;
}

export interface PhaseSegment {
  phase: string;
  start_frame: number;
  end_frame: number;
  start_s: number;
  end_s: number;
  mean_confidence: number;
}

export interface TimelineEvent {
  id: string;
  kind: 'instrument_in_view' | 'possible_bleeding';
  start_frame: number;
  end_frame: number;
  start_s: number;
  end_s: number;
  validated: boolean;
  detail: Record<string, string | number>;
}

export interface RunContext {
  label: string;
  frame_times: number[];
  key_frames: number[];
  timeline: { segments: PhaseSegment[]; events: TimelineEvent[]; gaps: [number, number][] };
  passages: Record<string, { doc_title: string; section: string; url: string; license: string }>;
  results: Record<string, { status: string; basis: Basis }>;
  drafts: number;
}

export type RunStatus = 'queued' | 'running' | 'awaiting_review' | 'reviewing' | 'done' | 'failed';

export interface Run {
  id: string;
  status: RunStatus;
  nodes_done: string[];
  error: string | null;
  trace_url: string | null;
  label: string | null;
  queue_position: number | null;
  context?: RunContext;
  review?: { claims: Claim[]; dropped: { claim: Claim; status: string; reasons: string[] }[] };
  approved?: boolean;
  report?: string | null;
}

export interface RunSummary {
  id: string;
  status: RunStatus;
  label: string | null;
}

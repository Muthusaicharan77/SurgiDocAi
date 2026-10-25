import { create } from 'zustand';

/** The reviewer's pending decision per run, shared by the overview and swipe review. */
interface ReviewState {
  removed: Record<string, string[]>;
  notes: Record<string, string>;
  setRemoved: (run: string, ids: string[]) => void;
  toggle: (run: string, id: string) => void;
  setNotes: (run: string, notes: string) => void;
}

export const useReview = create<ReviewState>(set => ({
  removed: {},
  notes: {},
  setRemoved: (run, ids) => set(s => ({ removed: { ...s.removed, [run]: ids } })),
  toggle: (run, id) =>
    set(s => {
      const cur = s.removed[run] ?? [];
      return { removed: { ...s.removed, [run]: cur.includes(id) ? cur.filter(x => x !== id) : [...cur, id] } };
    }),
  setNotes: (run, notes) => set(s => ({ notes: { ...s.notes, [run]: notes } })),
}));

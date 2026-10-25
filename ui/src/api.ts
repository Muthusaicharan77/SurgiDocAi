import type { Run, RunSummary } from './types';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  runs: () => fetch('/runs').then(r => json<RunSummary[]>(r)),
  run: (id: string) => fetch(`/runs/${id}`).then(r => json<Run>(r)),
  report: (id: string) =>
    fetch(`/runs/${id}/report`).then(r => (r.ok ? r.text() : Promise.reject(new Error(r.statusText)))),
  frameUrl: (id: string, frame: number) => `/runs/${id}/frames/${frame}`,

  review: (id: string, body: { approve: boolean; remove: string[]; notes: string }) =>
    fetch(`/runs/${id}/review`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => json<{ id: string; status: string }>(r)),

  /** Upload with progress (fetch cannot report upload progress). */
  upload: (file: File, onProgress: (fraction: number) => void) =>
    new Promise<{ id: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      form.append('video', file);
      xhr.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        const body = JSON.parse(xhr.responseText || '{}');
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.detail ?? `upload failed (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error('upload failed: server not reachable'));
      xhr.open('POST', '/runs');
      xhr.send(form);
    }),

  /** Server-Sent Events for a run; returns a function that closes the stream. */
  events: (id: string, onNode: (node: string) => void, onStatus: (status: string, error?: string) => void) => {
    const source = new EventSource(`/runs/${id}/events`);
    source.addEventListener('node', e => onNode(JSON.parse((e as MessageEvent).data).node));
    source.addEventListener('status', e => {
      const data = JSON.parse((e as MessageEvent).data);
      onStatus(data.status, data.error);
      if (['awaiting_review', 'done', 'failed'].includes(data.status)) source.close();
    });
    source.onerror = () => source.close();
    return () => source.close();
  },
};

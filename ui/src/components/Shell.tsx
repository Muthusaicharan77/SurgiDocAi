import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Activity, ShieldAlert } from 'lucide-react';

export default function Shell({ children, actions, title }: {
  children: ReactNode;
  actions?: ReactNode;
  title?: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="bg-amber-50 border-b border-amber-200 text-amber-900 text-xs sm:text-sm">
        <div className="max-w-6xl mx-auto px-4 py-1.5 flex items-center gap-2">
          <ShieldAlert size={16} className="shrink-0" />
          Research demo on public data (CholecSeg8k). Not a medical device, not for clinical use.
        </div>
      </div>
      <header className="bg-white/80 backdrop-blur border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-2 font-bold text-slate-900 shrink-0">
            <span className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center">
              <Activity size={18} />
            </span>
            SurgiDoc
          </Link>
          <div className="flex-1 min-w-0 text-center truncate">{title}</div>
          <div className="flex items-center gap-2 shrink-0">{actions}</div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
    </div>
  );
}

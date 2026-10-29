import React from "react";
import { AlertTriangle } from "lucide-react";

export const Disclaimer: React.FC = () => {
  return (
    <footer className="w-full border-t border-border bg-white py-4 px-6 mt-auto">
      <div className="max-w-[1200px] mx-auto flex items-center justify-center gap-2 text-xs text-muted">
        <AlertTriangle className="w-4 h-4 text-status-uncertain flex-shrink-0" />
        <span>
          <strong>Research prototype only.</strong> Not a certified medical device. Not intended for clinical diagnosis, treatment decisions, or emergency care.
        </span>
      </div>
    </footer>
  );
};

import React, { useEffect, useState } from "react";
import { Activity, HelpCircle } from "lucide-react";
import { getHealth } from "../api/client";

interface HeaderProps {
  currentTab: "upload" | "about";
  onTabChange: (tab: "upload" | "about") => void;
}

export const Header: React.FC<HeaderProps> = ({ currentTab, onTabChange }) => {
  const [backendReady, setBackendReady] = useState<boolean | null>(null);

  useEffect(() => {
    getHealth()
      .then(() => setBackendReady(true))
      .catch(() => setBackendReady(false));
  }, []);

  return (
    <header className="w-full bg-surface border-b border-border sticky top-0 z-30">
      <div className="max-w-[1200px] mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            <Activity className="w-5 h-5 stroke-[2.2]" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold tracking-tight text-ink">ECG Triage</h1>
            <span className="font-mono text-[11px] font-medium bg-bg text-muted px-2 py-0.5 rounded border border-border">
              v1.0.0
            </span>
          </div>
        </div>

        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => onTabChange("upload")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              currentTab === "upload"
                ? "bg-bg text-primary font-semibold"
                : "text-muted hover:text-ink hover:bg-bg/50"
            }`}
          >
            Analysis
          </button>
          <button
            onClick={() => onTabChange("about")}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
              currentTab === "about"
                ? "bg-bg text-primary font-semibold"
                : "text-muted hover:text-ink hover:bg-bg/50"
            }`}
          >
            <HelpCircle className="w-4 h-4" />
            <span>About</span>
          </button>

          <div className="h-4 w-px bg-border mx-2" />

          {/* Backend Status Indicator */}
          <div className="flex items-center gap-1.5 text-xs font-mono text-muted pl-1">
            <span
              className={`w-2 h-2 rounded-full ${
                backendReady === true
                  ? "bg-status-confident animate-pulse"
                  : backendReady === false
                  ? "bg-status-rejected"
                  : "bg-muted"
              }`}
            />
            <span className="hidden md:inline">
              {backendReady === true ? "Inference Ready" : backendReady === false ? "Offline" : "Connecting..."}
            </span>
          </div>
        </nav>
      </div>
    </header>
  );
};

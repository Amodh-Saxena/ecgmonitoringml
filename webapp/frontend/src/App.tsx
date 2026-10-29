import React, { useState } from "react";
import { Header } from "./components/Header";
import { Disclaimer } from "./components/Disclaimer";
import { UploadPage } from "./pages/UploadPage";
import { AboutPage } from "./pages/AboutPage";
import type { PredictResponse } from "./api/client";
import { CheckCircle2, AlertTriangle, XCircle, ArrowLeft } from "lucide-react";

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<"upload" | "about">("upload");
  const [analysisResult, setAnalysisResult] = useState<{
    data: PredictResponse;
    fileName: string;
  } | null>(null);

  const handleAnalysisSuccess = (data: PredictResponse, fileName: string) => {
    setAnalysisResult({ data, fileName });
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setCurrentTab("upload");
  };

  return (
    <div className="flex flex-col min-h-screen bg-bg">
      <Header
        currentTab={currentTab}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          setAnalysisResult(null);
        }}
      />

      <main className="flex-1 max-w-[1200px] w-full mx-auto">
        {currentTab === "about" ? (
          <AboutPage />
        ) : analysisResult ? (
          /* Interim result container for Phase 2 - will become the full Results page in Phase 3 */
          <div className="max-w-[850px] mx-auto py-10 px-6 space-y-6">
            <button
              onClick={handleReset}
              className="inline-flex items-center gap-2 text-sm text-primary hover:underline font-medium"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Upload another ECG</span>
            </button>

            <div className="bg-surface rounded-xl border border-border p-6 shadow-sm space-y-6">
              <div className="flex items-center justify-between pb-4 border-b border-border">
                <div>
                  <h3 className="text-lg font-semibold text-ink">Analysis Complete</h3>
                  <p className="text-xs text-muted font-mono">{analysisResult.fileName}</p>
                </div>

                {/* Status Badge */}
                <div
                  className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${
                    analysisResult.data.status === "confident"
                      ? "bg-status-confident-bg text-status-confident"
                      : analysisResult.data.status === "uncertain"
                      ? "bg-status-uncertain-bg text-status-uncertain"
                      : "bg-status-rejected-bg text-status-rejected"
                  }`}
                >
                  {analysisResult.data.status === "confident" ? (
                    <CheckCircle2 className="w-4 h-4" />
                  ) : analysisResult.data.status === "uncertain" ? (
                    <AlertTriangle className="w-4 h-4" />
                  ) : (
                    <XCircle className="w-4 h-4" />
                  )}
                  <span className="capitalize">{analysisResult.data.status}</span>
                </div>
              </div>

              {/* Status details */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3 bg-bg rounded-lg border border-border">
                  <span className="text-muted block text-[11px]">Quality Score</span>
                  <span className="text-ink font-semibold text-sm">
                    {analysisResult.data.quality_score}
                  </span>
                </div>
                <div className="p-3 bg-bg rounded-lg border border-border">
                  <span className="text-muted block text-[11px]">Leads Used</span>
                  <span className="text-ink font-semibold text-sm">
                    {analysisResult.data.leads_used} / 12
                  </span>
                </div>
                <div className="p-3 bg-bg rounded-lg border border-border">
                  <span className="text-muted block text-[11px]">Sampling Rate</span>
                  <span className="text-ink font-semibold text-sm">
                    {analysisResult.data.signal?.fs || 100} Hz
                  </span>
                </div>
                <div className="p-3 bg-bg rounded-lg border border-border">
                  <span className="text-muted block text-[11px]">Model Version</span>
                  <span className="text-ink font-semibold text-sm">
                    {analysisResult.data.model_version}
                  </span>
                </div>
              </div>

              {/* Classes summary */}
              {analysisResult.data.classes.length > 0 && (
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-ink">Diagnostic Probabilities</h4>
                  <div className="space-y-2">
                    {analysisResult.data.classes.map((cls) => (
                      <div
                        key={cls.code}
                        className="flex items-center justify-between p-3 rounded-lg bg-bg border border-border"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-ink w-12">
                            {cls.code}
                          </span>
                          <span className="text-xs text-muted">{cls.name}</span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="font-mono text-xs text-ink font-semibold">
                            {(cls.probability * 100).toFixed(1)}%
                          </span>
                          {cls.flagged && (
                            <span className="text-[11px] font-semibold text-status-uncertain bg-status-uncertain-bg px-2 py-0.5 rounded border border-status-uncertain/30">
                              Flagged for review
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <UploadPage onSuccess={handleAnalysisSuccess} />
        )}
      </main>

      <Disclaimer />
    </div>
  );
};

export default App;

import React, { useEffect, useState } from "react";
import { Shield, Database, Award, AlertTriangle } from "lucide-react";
import { getModelInfo } from "../api/client";
import type { ModelInfoResponse } from "../api/client";

export const AboutPage: React.FC = () => {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    getModelInfo().then(setInfo).catch(console.error);
  }, []);

  return (
    <div className="w-full max-w-[850px] mx-auto py-10 px-6 space-y-8">
      {/* Title */}
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">
          About the ECG Triage Model
        </h2>
        <p className="text-sm text-muted">
          Technical specifications, intended clinical workflow, and evaluation benchmarks.
        </p>
      </div>

      {/* Intended Use Card */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-primary">
          <Shield className="w-5 h-5" />
          <h3 className="text-base font-semibold text-ink">Intended Use</h3>
        </div>
        <p className="text-sm text-muted leading-relaxed">
          This system is an automated triage tool designed to provide a fast secondary review of 12-lead electrocardiograms.
          It evaluates raw physiological signals against five diagnostic superclasses: 
          <strong> Normal ECG (NORM)</strong>, <strong>Myocardial Infarction (MI)</strong>, 
          <strong> ST/T Changes (STTC)</strong>, <strong>Conduction Disturbance (CD)</strong>, and 
          <strong> Hypertrophy (HYP)</strong>.
        </p>
        <p className="text-sm text-muted leading-relaxed">
          A result flagged for review indicates elevated model confidence warranting manual clinician inspection. It does not replace a physician interpretation.
        </p>
      </div>

      {/* Dataset & Training Specs */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <Database className="w-5 h-5" />
          <h3 className="text-base font-semibold text-ink">Dataset & Provenance</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-3 bg-bg rounded-lg border border-border">
            <span className="text-muted block text-[11px] uppercase">Primary Dataset</span>
            <span className="text-ink font-semibold text-sm">PTB-XL v1.0.3 (PhysioNet)</span>
          </div>
          <div className="p-3 bg-bg rounded-lg border border-border">
            <span className="text-muted block text-[11px] uppercase">Record Count</span>
            <span className="text-ink font-semibold text-sm">21,799 12-lead records</span>
          </div>
          <div className="p-3 bg-bg rounded-lg border border-border">
            <span className="text-muted block text-[11px] uppercase">Architecture</span>
            <span className="text-ink font-semibold text-sm">1D ResNet (24 channels, 4 stages)</span>
          </div>
          <div className="p-3 bg-bg rounded-lg border border-border">
            <span className="text-muted block text-[11px] uppercase">Inference Engine</span>
            <span className="text-ink font-semibold text-sm">ONNX Runtime (CPU)</span>
          </div>
          <div className="p-3 bg-bg rounded-lg border border-border sm:col-span-2">
            <span className="text-muted block text-[11px] uppercase">Bundle Hash</span>
            <span className="text-ink font-mono text-xs truncate block" title={info?.manifest_hash}>
              {info?.manifest_hash || "81f46ccd594df21f5a125dd7e5d65c7bc5b5e6d17e3358d95fb58d576ef9ea77"}
            </span>
          </div>
        </div>
      </div>

      {/* Benchmark Metrics Table */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-primary">
          <Award className="w-5 h-5" />
          <h3 className="text-base font-semibold text-ink">Holdout Benchmark Results</h3>
        </div>
        <p className="text-xs text-muted">
          Evaluated on PTB-XL holdout Test Fold (2,198 records) with 1,000-fold patient-level bootstrap confidence intervals.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border text-muted font-medium">
                <th className="pb-2">Metric</th>
                <th className="pb-2 font-mono">LightGBM Baseline</th>
                <th className="pb-2 font-mono text-primary font-bold">ResNet1D (v1.0.0)</th>
                <th className="pb-2">Clinical Objective</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-ink font-mono">
              <tr>
                <td className="py-2.5 font-sans font-medium">Test Macro AUROC</td>
                <td className="py-2.5 text-muted">0.8456</td>
                <td className="py-2.5 text-primary font-bold">0.9112 [0.9045, 0.9194]</td>
                <td className="py-2.5 text-xs font-sans text-status-confident font-medium">Beats Baseline (+0.0656)</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-medium">Test Macro AUPRC</td>
                <td className="py-2.5 text-muted">0.6568</td>
                <td className="py-2.5 text-primary font-bold">0.7768</td>
                <td className="py-2.5 text-xs font-sans text-status-confident font-medium">+0.1200 Gain</td>
              </tr>
              <tr>
                <td className="py-2.5 font-sans font-medium">Calibrated ECE</td>
                <td className="py-2.5 text-muted">0.0820</td>
                <td className="py-2.5 text-primary font-bold">0.0225</td>
                <td className="py-2.5 text-xs font-sans text-status-confident font-medium">&lt; 0.05 Target Met</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Known Limitations */}
      <div className="bg-surface rounded-xl border border-border p-6 shadow-sm space-y-3">
        <div className="flex items-center gap-2 text-status-uncertain">
          <AlertTriangle className="w-5 h-5" />
          <h3 className="text-base font-semibold text-ink">Known Limitations</h3>
        </div>
        <ul className="text-xs text-muted space-y-2 list-disc list-inside leading-relaxed">
          <li>
            <strong>In-Domain Evaluation:</strong> The current performance is established on PTB-XL in-domain holdout folds. Generalization to external clinical equipment or hospital registries may exhibit domain shift.
          </li>
          <li>
            <strong>Superclass Level:</strong> Detects 5 diagnostic superclasses. Specific morphological sub-variants (e.g. STEMI vs NSTEMI, exact bundle branch block localization) require clinical review.
          </li>
          <li>
            <strong>Sampling Specifications:</strong> Expects 10-second 12-lead ECG signals. Single-lead or short-strip wearable ECGs are rejected or yield degraded quality scores.
          </li>
        </ul>
      </div>
    </div>
  );
};

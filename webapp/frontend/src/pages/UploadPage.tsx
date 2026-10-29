import React, { useState, useRef } from "react";
import type { DragEvent, ChangeEvent } from "react";
import { Upload, FileText, CheckCircle2, Loader2, Sparkles } from "lucide-react";
import { predict, getSample } from "../api/client";
import type { PredictResponse, ApiError } from "../api/client";
import { NoticeBanner } from "../components/NoticeBanner";

interface UploadPageProps {
  onSuccess: (data: PredictResponse, fileName: string) => void;
}

const MAX_SIZE_BYTES = 20 * 1024 * 1024; // 20 MB
const ALLOWED_EXTENSIONS = [".zip", ".csv", ".npy"];

export const UploadPage: React.FC<UploadPageProps> = ({ onSuccess }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fs, setFs] = useState<number>(100);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState<string>("");
  const [error, setError] = useState<ApiError | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const isTabularFormat = selectedFile && (
    selectedFile.name.toLowerCase().endsWith(".csv") ||
    selectedFile.name.toLowerCase().endsWith(".npy")
  );

  const validateFile = (file: File): boolean => {
    setError(null);
    const name = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));

    if (!hasValidExt) {
      setError({
        code: "bad_file",
        title: "Unsupported file type",
        message: `The file '${file.name}' is not recognized.`,
        nextStep: "Upload a WFDB archive (.zip), a comma-separated file (.csv), or a numpy matrix (.npy).",
      });
      return false;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setError({
        code: "too_large",
        title: "File size exceeds limit",
        message: `File is ${(file.size / (1024 * 1024)).toFixed(1)} MB, exceeding the 20 MB ceiling.`,
        nextStep: "Choose an ECG recording under 20 MB.",
      });
      return false;
    }

    if (file.size === 0) {
      setError({
        code: "bad_file",
        title: "Empty file",
        message: "The chosen file contains 0 bytes.",
        nextStep: "Select a valid non-empty ECG data file.",
      });
      return false;
    }

    return true;
  };

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        setSelectedFile(file);
      }
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setLoadingStep("Reading and verifying upload...");

    try {
      setLoadingStep("Processing leads and running ONNX inference...");
      const result = await predict(selectedFile, isTabularFormat ? fs : undefined);
      onSuccess(result, selectedFile.name);
    } catch (err: any) {
      setError(err);
    } finally {
      setIsLoading(false);
      setLoadingStep("");
    }
  };

  const handleTrySample = async () => {
    setIsLoading(true);
    setError(null);
    setLoadingStep("Loading public PTB-XL sample ECG record...");

    try {
      const result = await getSample();
      onSuccess(result, "PTB-XL Record 00001 (Sample)");
    } catch (err: any) {
      setError(err);
    } finally {
      setIsLoading(false);
      setLoadingStep("");
    }
  };

  return (
    <div className="w-full max-w-[800px] mx-auto py-10 px-6 space-y-8">
      {/* Intro section */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">
          12-Lead ECG Abnormality Triage
        </h2>
        <p className="text-sm text-muted max-w-lg mx-auto">
          Upload a standard 10-second ECG recording to evaluate diagnostic superclasses, calibrated probabilities, and lead attention.
        </p>
      </div>

      {/* Error banner */}
      {error && (
        <NoticeBanner
          type="error"
          title={error.title || "Upload error"}
          message={error.message}
          nextStep={error.nextStep}
          onClose={() => setError(null)}
        />
      )}

      {/* Main Dropzone Card */}
      <div className="bg-surface rounded-xl border border-border p-8 shadow-sm space-y-6">
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragActive
              ? "border-primary bg-primary-light/40"
              : "border-border hover:border-primary/50 hover:bg-bg/60"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".zip,.csv,.npy"
            onChange={handleFileChange}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
              <Upload className="w-6 h-6 stroke-[2]" />
            </div>

            <div className="space-y-1">
              <p className="text-sm font-medium text-ink">
                Drag and drop your ECG file here, or{" "}
                <span className="text-primary hover:underline">browse</span>
              </p>
              <p className="text-xs text-muted">
                Accepts WFDB zip (.hea + .dat), CSV, or NPY matrices (up to 20 MB)
              </p>
            </div>
          </div>
        </div>

        {/* Selected file card */}
        {selectedFile && (
          <div className="p-4 rounded-lg bg-bg border border-border flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-primary" />
              <div>
                <p className="text-sm font-medium text-ink">{selectedFile.name}</p>
                <p className="text-xs text-muted font-mono">
                  {(selectedFile.size / 1024).toFixed(1)} KB
                </p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1 text-xs text-status-confident font-medium">
              <CheckCircle2 className="w-4 h-4" /> Ready
            </span>
          </div>
        )}

        {/* Sampling rate input for CSV and NPY */}
        {isTabularFormat && (
          <div className="p-4 rounded-lg bg-bg border border-border space-y-2">
            <label htmlFor="fs-input" className="block text-xs font-medium text-ink">
              Sampling Frequency (Hertz)
            </label>
            <div className="flex items-center gap-3">
              <input
                id="fs-input"
                type="number"
                min="50"
                max="2000"
                value={fs}
                onChange={(e) => setFs(parseInt(e.target.value) || 100)}
                className="w-32 px-3 py-1.5 text-sm font-mono border border-border rounded-lg bg-surface text-ink focus:outline-none focus:ring-1 focus:ring-primary"
              />
              <span className="text-xs text-muted">
                Common medical standards: 100 Hz, 250 Hz, 500 Hz. Resampled internally to 100 Hz.
              </span>
            </div>
          </div>
        )}

        {/* Action button */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
          <button
            type="button"
            onClick={handleTrySample}
            disabled={isLoading}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium text-primary hover:bg-primary-light transition-colors border border-primary/20 disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            <span>Try a sample ECG</span>
          </button>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={!selectedFile || isLoading}
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg text-sm font-medium bg-primary text-white hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>{loadingStep || "Processing..."}</span>
              </>
            ) : (
              <span>Analyze ECG</span>
            )}
          </button>
        </div>
      </div>

      {/* Security & privacy assurance per PRD */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-muted">
        <div className="p-4 rounded-lg bg-surface border border-border">
          <h4 className="font-semibold text-ink mb-1">In-Memory Processing</h4>
          <p>Signal traces are processed in volatile memory. No uploaded patient files are persisted to disk.</p>
        </div>
        <div className="p-4 rounded-lg bg-surface border border-border">
          <h4 className="font-semibold text-ink mb-1">Header Stripping</h4>
          <p>Patient identifiers (name, age, sex, dates) in headers are stripped before inference.</p>
        </div>
        <div className="p-4 rounded-lg bg-surface border border-border">
          <h4 className="font-semibold text-ink mb-1">Calibrated Triage</h4>
          <p>Output probabilities are calibrated using temperature scaling with 90% sensitivity thresholds.</p>
        </div>
      </div>
    </div>
  );
};

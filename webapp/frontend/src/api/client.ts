/**
 * API client communicating with FastAPI backend.
 * All backend calls in the application MUST go through this module.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "/api";

export interface ClassResult {
  code: string;
  name: string;
  probability: number;
  threshold: number;
  flagged: boolean;
}

export interface SignalResult {
  fs: number;
  unit: string;
  leads: string[];
  samples: number[][];
}

export interface AttentionResult {
  method: string;
  per_class: number[][];
}

export interface PredictResponse {
  schema_version: string;
  model_version: string;
  manifest_hash: string;
  status: "confident" | "uncertain" | "rejected";
  reasons: string[];
  quality_score: number;
  leads_used: number;
  classes: ClassResult[];
  signal?: SignalResult;
  attention?: AttentionResult;
}

export interface ModelInfoResponse {
  model_version: string;
  manifest_hash: string;
  classes: { id: string; display_name: string }[];
  thresholds: number[];
  temperatures: number[];
  test_metrics: {
    macro_auroc: number;
    macro_auroc_ci95?: [number, number];
    macro_auprc?: number;
    macro_ece?: number;
  };
  license: string;
  intended_use: string;
  limitations: string[];
}

export interface ApiError {
  code: "bad_file" | "too_large" | "missing_field" | "timeout" | "inference_error" | "network_error";
  message: string;
  field?: string;
  title?: string;
  nextStep?: string;
}

/**
 * Helper to normalize backend errors into user-friendly diagnostic messages.
 */
function normalizeError(status: number, data: any): ApiError {
  const code = data?.code || (status === 413 ? "too_large" : status === 422 ? "missing_field" : "bad_file");
  const rawMsg = data?.message || "An unexpected error occurred while processing the ECG file.";
  const field = data?.field;

  let title = "Unable to process file";
  let nextStep = "Check the file format and try again.";

  if (code === "too_large") {
    title = "File exceeds size limit";
    nextStep = "Upload an ECG file under 20 MB.";
  } else if (code === "missing_field") {
    title = "Sampling rate required";
    nextStep = "Specify the sampling frequency in Hertz (e.g. 100 or 500 Hz).";
  } else if (rawMsg.toLowerCase().includes("flat")) {
    title = "Flatline signal detected";
    nextStep = "Ensure the electrode leads recorded active voltage signals.";
  } else if (rawMsg.toLowerCase().includes("under 10") || rawMsg.toLowerCase().includes("duration")) {
    title = "Signal duration too short";
    nextStep = "Provide an ECG recording of at least 10.0 seconds.";
  } else if (rawMsg.toLowerCase().includes("nan")) {
    title = "Corrupt numerical data";
    nextStep = "Check the source file for missing or NaN sensor values.";
  } else if (rawMsg.toLowerCase().includes("header") || rawMsg.toLowerCase().includes(".hea")) {
    title = "Missing WFDB header";
    nextStep = "Ensure the uploaded zip contains both the .hea header and .dat signal files.";
  }

  return {
    code,
    message: rawMsg,
    field,
    title,
    nextStep
  };
}

/**
 * Upload an ECG file (WFDB zip, CSV, or NPY) and receive triage results.
 */
export async function predict(file: File, fs?: number): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (fs !== undefined && fs > 0) {
    formData.append("fs", fs.toString());
  }

  try {
    const res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw normalizeError(res.status, data);
    }
    return data as PredictResponse;
  } catch (err: any) {
    if (err.code) throw err;
    throw {
      code: "network_error",
      message: err.message || "Failed to connect to backend inference server.",
      title: "Connection error",
      nextStep: "Verify that the backend service is running on http://127.0.0.1:8000."
    } as ApiError;
  }
}

/**
 * Fetch triage analysis for the bundled public PTB-XL sample ECG.
 */
export async function getSample(): Promise<PredictResponse> {
  try {
    const res = await fetch(`${BASE_URL}/sample`);
    const data = await res.json();
    if (!res.ok) {
      throw normalizeError(res.status, data);
    }
    return data as PredictResponse;
  } catch (err: any) {
    if (err.code) throw err;
    throw {
      code: "network_error",
      message: err.message || "Failed to load sample ECG.",
      title: "Sample error",
      nextStep: "Check backend server status."
    } as ApiError;
  }
}

/**
 * Health check probe.
 */
export async function getHealth(): Promise<{ status: string; model_version: string }> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

/**
 * Fetch model card and architecture details.
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${BASE_URL}/model-info`);
  if (!res.ok) throw new Error("Failed to fetch model info");
  return res.json();
}

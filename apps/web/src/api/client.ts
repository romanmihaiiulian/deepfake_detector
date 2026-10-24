import type { AnalysisCreated, AnalysisStatus, ErrorResponse, Health } from "./types";

const BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function errorFrom(status: number, body: unknown): ApiError {
  const error = (body as Partial<ErrorResponse> | null)?.error;
  if (error?.code) return new ApiError(status, error.code, error.message);
  if (status === 0) return new ApiError(0, "network", "The server could not be reached.");
  return new ApiError(status, "http_error", `The server answered with status ${status}.`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw errorFrom(0, null);
  }
  if (response.status === 204) return undefined as T;
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw errorFrom(response.status, body);
  return body as T;
}

/** Upload with progress reporting (fetch cannot report upload progress, XHR can). */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<AnalysisCreated> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/analyses`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) resolve(xhr.response as AnalysisCreated);
      else reject(errorFrom(xhr.status, xhr.response));
    };
    xhr.onerror = () => reject(errorFrom(0, null));
    xhr.onabort = () => reject(new ApiError(0, "aborted", "The upload was cancelled."));
    signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export function getAnalysis(id: string): Promise<AnalysisStatus> {
  return request<AnalysisStatus>(`/analyses/${encodeURIComponent(id)}`);
}

export function deleteAnalysis(id: string): Promise<void> {
  return request<void>(`/analyses/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

export function evidenceUrl(analysisId: string, evidenceId: string): string {
  return `${BASE}/analyses/${encodeURIComponent(analysisId)}/evidence/${encodeURIComponent(evidenceId)}`;
}

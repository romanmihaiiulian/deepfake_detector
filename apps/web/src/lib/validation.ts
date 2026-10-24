export const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".webm"] as const;
export const MAX_UPLOAD_MB = 500;

/** Client-side checks; the server validates again (container, duration, decodability). */
export function validateFile(file: Pick<File, "name" | "size">, maxMb = MAX_UPLOAD_MB): string | null {
  const dot = file.name.lastIndexOf(".");
  const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!(ACCEPTED_EXTENSIONS as readonly string[]).includes(extension)) {
    return "Please choose an MP4, MOV or WEBM video.";
  }
  if (file.size === 0) return "This file is empty.";
  if (file.size > maxMb * 1024 * 1024) return `Videos can be at most ${maxMb} MB.`;
  return null;
}

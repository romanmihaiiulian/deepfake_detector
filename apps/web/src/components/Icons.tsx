import type { Verdict } from "../api/types";

/** Verdict icons: shape as well as colour, so the verdict never relies on colour alone. */
export function VerdictIcon({ verdict }: { verdict: Verdict }) {
  const common = {
    width: 40,
    height: 40,
    viewBox: "0 0 40 40",
    "aria-hidden": true,
    className: "verdict-icon",
  } as const;
  if (verdict === "manipulated_likely") {
    return (
      <svg {...common}>
        <path d="M20 4 37 34H3Z" fill="currentColor" opacity="0.15" />
        <path d="M20 4 37 34H3Z" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
        <path d="M20 15v9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <circle cx="20" cy="29" r="1.8" fill="currentColor" />
      </svg>
    );
  }
  if (verdict === "authentic_likely") {
    return (
      <svg {...common}>
        <circle cx="20" cy="20" r="16" fill="currentColor" opacity="0.15" />
        <circle cx="20" cy="20" r="16" fill="none" stroke="currentColor" strokeWidth="2.5" />
        <path d="m12.5 20.5 5 5 10-11" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <circle cx="20" cy="20" r="16" fill="currentColor" opacity="0.15" />
      <circle cx="20" cy="20" r="16" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <path d="M15.5 15.5a4.5 4.5 0 1 1 6.6 4c-1.4.8-2.1 1.8-2.1 3.3" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <circle cx="20" cy="28.5" r="1.8" fill="currentColor" />
    </svg>
  );
}

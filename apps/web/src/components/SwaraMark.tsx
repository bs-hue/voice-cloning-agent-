export function SwaraMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <rect width="32" height="32" rx="8" fill="var(--color-accent)" />
      <path
        d="M8 18V14M12 22V10M16 25V7M20 21V11M24 18V14"
        stroke="var(--color-on-accent)"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </svg>
  );
}


"use client";

/**
 * The progress dial.
 *
 * Two rings, and the second one is the point. The solid arc is the real figure
 * the server sent. Behind it a slow dashed ring turns, which says "still
 * working" during the long gap between one page finishing and the next — a
 * still bar at 40% for ninety seconds reads as a hang, and this is a whole
 * album page being detected, not a stall.
 *
 * The turning ring carries no information, so it is `aria-hidden`, and the
 * `prefers-reduced-motion` rule in globals.css stops it moving for anyone who
 * asked for that. The number stays correct either way.
 */
export function ProgressRing({
  percent,
  label,
  sublabel,
  busy = true,
  tone = "running",
}: {
  percent: number;
  label: string;
  sublabel?: string;
  busy?: boolean;
  tone?: "running" | "done";
}) {
  const size = 208;
  const stroke = 12;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, percent));
  const offset = circumference * (1 - clamped / 100);

  return (
    <div className="relative mx-auto flex h-52 w-52 items-center justify-center">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="absolute inset-0 h-full w-full -rotate-90"
        aria-hidden="true"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={stroke}
          className="stroke-paper-200 dark:stroke-ink-800"
        />
        {busy ? (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius - stroke - 3}
            fill="none"
            strokeWidth={2}
            strokeDasharray="3 13"
            strokeLinecap="round"
            className="origin-center animate-[spin_9s_linear_infinite] stroke-ember-500/50"
          />
        ) : null}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className={
            tone === "done"
              ? "stroke-emerald-600 transition-[stroke-dashoffset] duration-500 ease-out dark:stroke-emerald-400"
              : "stroke-ember-600 transition-[stroke-dashoffset] duration-500 ease-out dark:stroke-ember-400"
          }
        />
      </svg>

      <div
        className="relative text-center"
        role="progressbar"
        // A progressbar's own text is not its name, so it needs one stated.
        // The label is the stage, which is what somebody arriving at this
        // announcement needs to hear before the number.
        aria-label={label}
        aria-valuenow={Math.round(clamped)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={sublabel ? `${Math.round(clamped)} per cent, ${sublabel}` : undefined}
      >
        <p className="font-mono text-4xl font-semibold tabular-nums text-ink-900 dark:text-paper-100">
          {Math.round(clamped)}
          <span className="text-2xl text-ink-700/70 dark:text-paper-300/70">%</span>
        </p>
        <p className="mt-1 text-sm font-medium text-ink-800 dark:text-paper-200">{label}</p>
        {sublabel ? (
          <p className="mt-0.5 px-6 text-xs text-ink-700/80 dark:text-paper-300/80">{sublabel}</p>
        ) : null}
      </div>
    </div>
  );
}

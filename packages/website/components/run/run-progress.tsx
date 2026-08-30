"use client";

import { AlertTriangle, Check, FileUp, Loader2, Scissors, Sparkles, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { ProgressRing } from "./progress-ring";
import { plural } from "@/lib/runner";

export type LogEntry = {
  seq: number;
  text: string;
  tone: "ok" | "flag" | "bad" | "note";
};

export type LiveProgress = {
  percent: number;
  stage: "split" | "enhance" | null;
  stageIndex: number;
  stageCount: number;
  done: number;
  total: number;
  unit: string;
  current: string | null;
};

const STAGE_LABEL = { split: "Splitting pages", enhance: "Restoring photographs" } as const;

/**
 * The in-progress screen.
 *
 * Everything on it is a figure the server sent. Nothing is interpolated,
 * estimated or smoothed, because a bar that invents motion to look busy is
 * lying about a run whose whole selling point is that it tells you the truth
 * about what it did.
 */
export function RunProgress({
  phase,
  progress,
  log,
  upload,
  onCancel,
}: {
  phase: "uploading" | "running";
  progress: LiveProgress;
  log: LogEntry[];
  upload: { sent: number; total: number; current: string } | null;
  onCancel: () => void;
}) {
  const sending = phase === "uploading" && upload !== null;
  const percent = sending ? (upload.sent / Math.max(1, upload.total)) * 100 : progress.percent;

  const label = sending
    ? "Sending to your computer"
    : progress.stage
      ? STAGE_LABEL[progress.stage]
      : "Starting";

  const noun = progress.unit === "page" ? "page" : "photograph";
  const sublabel = sending
    ? `File ${upload.sent} of ${upload.total}`
    : !progress.total
      ? "Reading the folder"
      : progress.done === 0
        ? // "Page 0 of 5" reads as a stall on the one screen where a stall is
          // what the person is afraid of.
          `${plural(progress.total, noun)} to go`
        : `${noun[0].toUpperCase()}${noun.slice(1)} ${progress.done} of ${progress.total}`;

  return (
    <div className="rounded-2xl border border-paper-200 bg-paper-100/50 p-7 dark:border-ink-800 dark:bg-ink-900/30 sm:p-10">
      <div className="grid gap-10 lg:grid-cols-[auto_minmax(0,1fr)] lg:gap-14">
        <div>
          <ProgressRing percent={percent} label={label} sublabel={sublabel} />
          {progress.current && !sending ? (
            <p className="mt-5 truncate text-center font-mono text-xs text-ink-700/80 dark:text-paper-300/80">
              {progress.current}
            </p>
          ) : null}
        </div>

        <div className="min-w-0">
          <StageTrack phase={phase} progress={progress} />

          {/* One polite announcement per step, rather than the whole log. */}
          <p className="sr-only" aria-live="polite">
            {sublabel}. {log.at(-1)?.text ?? ""}
          </p>

          <ActivityLog log={log} sending={sending} />

          <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
            <p className="text-sm text-ink-700/80 dark:text-paper-300/80">
              Running on your computer. You can leave this page open; closing it
              stops the run.
            </p>
            <button
              type="button"
              onClick={onCancel}
              className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-paper-300 px-4 py-2 text-sm font-medium text-ink-800 transition-colors hover:border-ink-800 dark:border-ink-700 dark:text-paper-200 dark:hover:border-paper-300"
            >
              <X className="h-4 w-4" aria-hidden="true" />
              Stop and discard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StageTrack({
  phase,
  progress,
}: {
  phase: "uploading" | "running";
  progress: LiveProgress;
}) {
  const steps = [
    { key: "upload", label: "Sending", icon: FileUp },
    ...(progress.stageCount === 2 || progress.stage === "split"
      ? [{ key: "split", label: "Split", icon: Scissors }]
      : []),
    ...(progress.stageCount === 2 || progress.stage === "enhance"
      ? [{ key: "enhance", label: "Restore", icon: Sparkles }]
      : []),
  ];

  const activeKey = phase === "uploading" ? "upload" : (progress.stage ?? "upload");
  const activeIndex = Math.max(
    0,
    steps.findIndex((step) => step.key === activeKey),
  );

  return (
    <ol className="flex flex-wrap items-center gap-x-3 gap-y-2">
      {steps.map((step, index) => {
        const done = index < activeIndex;
        const active = index === activeIndex;
        return (
          <li key={step.key} className="flex items-center gap-3">
            <span
              className={[
                "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium",
                done
                  ? "border-emerald-600/40 bg-emerald-600/10 text-emerald-800 dark:border-emerald-400/30 dark:text-emerald-300"
                  : active
                    ? "border-ember-500 bg-ember-500/10 text-ink-900 dark:text-paper-100"
                    : "border-paper-300 text-ink-700/80 dark:border-ink-700 dark:text-paper-300/70",
              ].join(" ")}
            >
              {done ? (
                <Check className="h-3.5 w-3.5" aria-hidden="true" />
              ) : active ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <step.icon className="h-3.5 w-3.5" aria-hidden="true" />
              )}
              {step.label}
              {done ? <span className="sr-only">(finished)</span> : null}
              {active ? <span className="sr-only">(in progress)</span> : null}
            </span>
            {index < steps.length - 1 ? (
              <span
                aria-hidden="true"
                className="h-px w-5 bg-paper-300 dark:bg-ink-700"
              />
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

function ActivityLog({ log, sending }: { log: LogEntry[]; sending: boolean }) {
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [log]);

  const tones = {
    ok: "text-paper-200",
    flag: "text-ember-400",
    bad: "text-red-400",
    note: "text-paper-300/70",
  } as const;

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-paper-300 bg-ink-950 dark:border-ink-700">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
        <span
          className="h-2 w-2 animate-pulse rounded-full bg-ember-400"
          aria-hidden="true"
        />
        <span className="font-mono text-xs text-paper-300/60">
          {sending ? "uploading" : "revelai"}
        </span>
      </div>
      {/* tabIndex: a scrollable region has to be reachable from the keyboard. */}
      <div
        ref={scroller}
        tabIndex={0}
        role="log"
        aria-label="Run activity"
        className="h-56 overflow-y-auto px-4 py-3 font-mono text-[13px] leading-relaxed"
      >
        {log.length === 0 ? (
          <p className="text-paper-300/70">waiting for the first page…</p>
        ) : (
          log.map((entry) => (
            <p key={entry.seq} className={`flex gap-2 ${tones[entry.tone]}`}>
              <span aria-hidden="true" className="select-none opacity-50">
                {entry.tone === "bad" ? "×" : entry.tone === "flag" ? "!" : "·"}
              </span>
              <span className="min-w-0 break-words">{entry.text}</span>
            </p>
          ))
        )}
      </div>
    </div>
  );
}

export function FlaggedList({
  title,
  items,
  tone = "flag",
}: {
  title: string;
  items: { name: string; reasons: string[] }[];
  tone?: "flag" | "bad";
}) {
  if (!items.length) return null;
  const Icon = tone === "bad" ? X : AlertTriangle;
  return (
    <div
      className={[
        "rounded-xl border p-5",
        tone === "bad"
          ? "border-red-400/60 bg-red-500/[0.04] dark:border-red-500/40"
          : "border-ember-500/50 bg-ember-500/[0.05]",
      ].join(" ")}
    >
      <h3 className="flex items-center gap-2 font-semibold text-ink-900 dark:text-paper-100">
        <Icon
          className={
            tone === "bad"
              ? "h-4 w-4 text-red-700 dark:text-red-400"
              : "h-4 w-4 text-ember-700 dark:text-ember-400"
          }
          aria-hidden="true"
        />
        {title} ({plural(items.length, "page")})
      </h3>
      <ul className="mt-3 space-y-1.5 text-sm">
        {items.map((item) => (
          <li key={item.name} className="flex flex-wrap gap-x-2">
            <code className="font-mono text-ink-900 dark:text-paper-100">{item.name}</code>
            <span className="text-ink-700/85 dark:text-paper-300/85">
              — {item.reasons.join("; ")}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

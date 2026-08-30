"use client";

import { Check, Download, RotateCcw, Trash2 } from "lucide-react";
import { FlaggedList } from "./run-progress";
import { type RunSummary, formatBytes, plural } from "@/lib/runner";

/**
 * The success screen.
 *
 * It reports what happened rather than congratulating anybody. The number that
 * matters most is not "done" — it is how many crops the tool is not confident
 * about, because those are the ones the person has to look at. Hiding them
 * behind a green tick would turn a tool that flags its own mistakes into one
 * that appears not to make any.
 */
export function RunResult({
  summary,
  downloadUrl,
  onAgain,
  onDiscard,
}: {
  summary: RunSummary;
  downloadUrl: string;
  onAgain: () => void;
  onDiscard: () => void;
}) {
  const { split, enhance, result } = summary;
  const needingReview = split?.pagesNeedingReview ?? [];
  const failed = [
    ...(split?.pagesFailed ?? []),
    ...(enhance?.photosFailed ?? []),
  ].map((item) => ({ name: item.name, reasons: [item.reason] }));

  const stats: { label: string; value: string }[] = [
    ...(split
      ? [
          { label: "Pages read", value: String(split.pagesRead) },
          { label: "Photographs found", value: String(split.photographs) },
        ]
      : []),
    ...(enhance ? [{ label: "Restored", value: String(enhance.restored) }] : []),
    { label: "In the zip", value: plural(result.images, "image") },
    { label: "Download size", value: formatBytes(result.bytes) },
    { label: "Took", value: `${summary.seconds.toFixed(1)}s` },
  ];

  return (
    <div className="rounded-2xl border border-paper-200 bg-paper-100/50 p-7 dark:border-ink-800 dark:bg-ink-900/30 sm:p-10">
      <div className="flex items-start gap-4">
        <span className="mt-0.5 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-emerald-600/12 text-emerald-700 dark:text-emerald-400">
          <Check className="h-6 w-6" aria-hidden="true" />
        </span>
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
            Done — {plural(result.images, "photograph")} ready
          </h2>
          <p className="mt-2 max-w-prose text-ink-700/90 dark:text-paper-300/90">
            The zip holds images and nothing else. It is on your own computer:
            downloading it copies it from a program running on this machine, and
            it was never sent anywhere.
          </p>
        </div>
      </div>

      <dl className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-paper-200 bg-paper-200 dark:border-ink-800 dark:bg-ink-800 sm:grid-cols-3 lg:grid-cols-6">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-paper-50 px-4 py-4 dark:bg-ink-950">
            <dt className="text-xs uppercase tracking-wider text-ink-700/80 dark:text-paper-300/80">
              {stat.label}
            </dt>
            <dd className="mt-1 font-mono text-xl font-semibold tabular-nums text-ink-900 dark:text-paper-100">
              {stat.value}
            </dd>
          </div>
        ))}
      </dl>

      <div className="mt-8 flex flex-wrap items-center gap-4">
        <a
          href={downloadUrl}
          download={result.file}
          className="inline-flex items-center gap-2 rounded-lg bg-ember-700 px-6 py-3 font-medium text-paper-50 transition-colors hover:bg-ember-600 dark:bg-ember-400 dark:text-ink-950 dark:hover:bg-ember-500"
        >
          <Download className="h-5 w-5" aria-hidden="true" />
          Download {result.file}
        </a>
        <button
          type="button"
          onClick={onAgain}
          className="inline-flex items-center gap-2 rounded-lg border border-paper-300 px-4 py-2.5 text-sm font-medium text-ink-800 transition-colors hover:border-ink-800 dark:border-ink-700 dark:text-paper-200 dark:hover:border-paper-300"
        >
          <RotateCcw className="h-4 w-4" aria-hidden="true" />
          Run another folder
        </button>
        <button
          type="button"
          onClick={onDiscard}
          className="inline-flex items-center gap-2 rounded-lg px-3 py-2.5 text-sm font-medium text-ink-700/80 transition-colors hover:text-ink-900 dark:text-paper-300/80 dark:hover:text-paper-100"
        >
          <Trash2 className="h-4 w-4" aria-hidden="true" />
          Delete it from the server
        </button>
      </div>

      {enhance?.operationsSkipped.length ? (
        <p className="mt-6 text-sm text-ink-700/85 dark:text-paper-300/85">
          Asked for but not done:{" "}
          {enhance.operationsSkipped.map((s) => `${s.name} (${s.reason})`).join("; ")}.
        </p>
      ) : null}

      {needingReview.length || failed.length ? (
        <div className="mt-8 space-y-4">
          <FlaggedList title="Worth looking at yourself" items={needingReview} />
          <FlaggedList title="These failed" items={failed} tone="bad" />
          {needingReview.length ? (
            <p className="max-w-prose text-sm text-ink-700/85 dark:text-paper-300/85">
              These crops were still written — a flagged crop is usually the
              right crop, and it is your call. To correct them by hand, run{" "}
              <code className="font-mono">revelai split --review</code> in a
              terminal: it opens each page and lets you drag the corners.
            </p>
          ) : null}
        </div>
      ) : null}

      <p className="mt-8 border-t border-paper-200 pt-5 text-sm text-ink-700/80 dark:border-ink-800 dark:text-paper-300/80">
        Everything this run produced lives in a temporary folder on your machine
        and is deleted when you stop <code className="font-mono">revelai serve</code>.
        Download the zip before you close that terminal.
      </p>
    </div>
  );
}

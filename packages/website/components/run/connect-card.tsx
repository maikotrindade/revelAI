"use client";

import { Loader2, RefreshCw, TerminalSquare } from "lucide-react";
import { CodeBlock } from "@/components/code-block";
import { DEFAULT_PORT } from "@/lib/runner";

/**
 * What the page shows when there is no RevelAI running on this machine.
 *
 * Which is the normal first visit, so it is written as an instruction rather
 * than an error. There is nothing wrong; the tool has not been started yet.
 */
export function ConnectCard({
  state,
  onRetry,
}: {
  state: "connecting" | "offline";
  onRetry: () => void;
}) {
  return (
    <div className="rounded-2xl border border-paper-200 bg-paper-100/60 p-7 dark:border-ink-800 dark:bg-ink-900/40 sm:p-9">
      <div className="flex items-start gap-4">
        <span className="mt-0.5 rounded-lg bg-paper-200 p-2.5 text-ink-800 dark:bg-ink-800 dark:text-paper-200">
          {state === "connecting" ? (
            <Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" />
          ) : (
            <TerminalSquare className="h-5 w-5" aria-hidden="true" />
          )}
        </span>
        <div className="min-w-0">
          <h3 className="text-xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
            {state === "connecting"
              ? "Looking for RevelAI on this computer…"
              : "Start RevelAI on this computer"}
          </h3>
          <p className="mt-2 max-w-prose text-ink-700/90 dark:text-paper-300/90">
            This page has no server behind it and never receives an image. It
            drives a copy of RevelAI running on your own machine, so your
            photographs are read by a program on your computer and go nowhere
            else. Two commands, once:
          </p>
        </div>
      </div>

      <div className="mt-7 space-y-3">
        <CodeBlock label="Install" command="pip install revelai" />
        <CodeBlock label="Start it" command="revelai serve --open" />
      </div>

      <p className="mt-4 text-sm text-ink-700/80 dark:text-paper-300/80">
        <code className="font-mono">serve</code> listens on{" "}
        <code className="font-mono">127.0.0.1:{DEFAULT_PORT}</code>, which is
        this computer and nothing else — no other machine on your network can
        reach it. Leave that terminal open while you work, and stop it with
        Ctrl+C when you are done. Everything it wrote is deleted when it stops.
      </p>

      <div className="mt-6 flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={onRetry}
          disabled={state === "connecting"}
          className="inline-flex items-center gap-2 rounded-lg bg-ink-900 px-4 py-2.5 text-sm font-medium text-paper-50 transition-colors hover:bg-ink-800 disabled:opacity-60 dark:bg-paper-100 dark:text-ink-900 dark:hover:bg-paper-200"
        >
          <RefreshCw
            className={`h-4 w-4 ${state === "connecting" ? "animate-spin" : ""}`}
            aria-hidden="true"
          />
          {state === "connecting" ? "Checking…" : "Check again"}
        </button>
        <p className="text-sm text-ink-700/80 dark:text-paper-300/80">
          Running it on another port? Add{" "}
          <code className="font-mono">?port=</code> to this page’s address.
        </p>
      </div>
    </div>
  );
}

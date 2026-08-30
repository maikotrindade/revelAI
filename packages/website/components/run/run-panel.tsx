"use client";

import { AlertTriangle, Play, RotateCcw, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConnectCard } from "./connect-card";
import { FolderDrop } from "./folder-drop";
import { OptionsForm } from "./options-form";
import { type LiveProgress, type LogEntry, RunProgress } from "./run-progress";
import { RunResult } from "./run-result";
import {
  type CreatedJob,
  type FolderVerdict,
  type Health,
  type PickedFile,
  type ProgressEvent,
  type RunOptions,
  type RunSummary,
  type Stage,
  RunnerError,
  checkFolder,
  createJob,
  discardJob,
  discover,
  folderNameOf,
  follow,
  formatBytes,
  plural,
  resultUrl,
  startJob,
  uploadFiles,
} from "@/lib/runner";

type Phase =
  | "connecting"
  | "offline"
  | "idle"
  | "checking"
  | "ready"
  | "uploading"
  | "running"
  | "done"
  | "failed";

const DEFAULT_OPTIONS: RunOptions = {
  color: true,
  colorStrength: 1,
  denoise: false,
  dust: false,
  upscale: 0,
  faces: false,
  colorize: false,
};

const EMPTY_PROGRESS: LiveProgress = {
  percent: 0,
  stage: null,
  stageIndex: 0,
  stageCount: 1,
  done: 0,
  total: 0,
  unit: "page",
  current: null,
};

/**
 * The whole run, as a small state machine.
 *
 * The order is fixed and each step has somewhere to fail to: find the server,
 * choose a folder, have the server pass judgement on it, choose what to run,
 * upload, watch, download. The one rule that shapes the rest is that nothing
 * moves until the folder has been accepted — a page that starts uploading and
 * then reports that file 30 was a PDF has already wasted the person's time.
 */
export function RunPanel() {
  const [phase, setPhase] = useState<Phase>("connecting");
  const [connection, setConnection] = useState<{ base: string; health: Health } | null>(null);
  const [picked, setPicked] = useState<PickedFile[]>([]);
  const [verdict, setVerdict] = useState<FolderVerdict | null>(null);
  const [stage, setStage] = useState<Stage>("split");
  const [options, setOptions] = useState<RunOptions>(DEFAULT_OPTIONS);
  const [job, setJob] = useState<CreatedJob | null>(null);
  const [upload, setUpload] = useState<{ sent: number; total: number; current: string } | null>(
    null,
  );
  const [progress, setProgress] = useState<LiveProgress>(EMPTY_PROGRESS);
  const [log, setLog] = useState<LogEntry[]>([]);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const unfollow = useRef<(() => void) | null>(null);
  const aborter = useRef<AbortController | null>(null);

  // ------------------------------------------------------------------
  // Finding the server
  // ------------------------------------------------------------------

  const connect = useCallback(async () => {
    setPhase("connecting");
    const found = await discover();
    if (found) {
      setConnection(found);
      setPhase("idle");
    } else {
      setConnection(null);
      setPhase("offline");
    }
  }, []);

  useEffect(() => {
    void connect();
    return () => {
      unfollow.current?.();
      aborter.current?.abort();
    };
  }, [connect]);

  // ------------------------------------------------------------------
  // Choosing a folder
  // ------------------------------------------------------------------

  const onPick = useCallback(
    async (files: PickedFile[]) => {
      if (!connection) return;
      setPicked(files);
      setVerdict(null);
      setPhase("checking");
      try {
        const answer = await checkFolder(connection.base, files);
        setVerdict(answer);
        setPhase(answer.ok ? "ready" : "idle");
      } catch (error) {
        setVerdict(null);
        setFailure(describe(error));
        setPhase("failed");
      }
    },
    [connection],
  );

  const reset = useCallback(() => {
    unfollow.current?.();
    aborter.current?.abort();
    unfollow.current = null;
    aborter.current = null;
    setPicked([]);
    setVerdict(null);
    setJob(null);
    setUpload(null);
    setProgress(EMPTY_PROGRESS);
    setLog([]);
    setSummary(null);
    setFailure(null);
    setPhase(connection ? "idle" : "offline");
  }, [connection]);

  // ------------------------------------------------------------------
  // Running
  // ------------------------------------------------------------------

  const onEvent = useCallback((event: ProgressEvent) => {
    if (event.type === "progress") {
      setProgress({
        percent: event.percent ?? 0,
        stage: event.stage ?? null,
        stageIndex: event.stageIndex ?? 0,
        stageCount: event.stageCount ?? 1,
        done: event.done ?? 0,
        total: event.total ?? 0,
        unit: event.unit ?? "page",
        current: event.current ?? null,
      });
      if (event.message && event.done) {
        setLog((entries) => [
          ...entries.slice(-199),
          {
            seq: event.seq,
            text: event.message as string,
            tone: event.failed ? "bad" : event.needsReview ? "flag" : "ok",
          },
        ]);
      } else if (event.message) {
        setLog((entries) => [
          ...entries.slice(-199),
          { seq: event.seq, text: event.message as string, tone: "note" },
        ]);
      }
      return;
    }
    if (event.type === "done" && event.summary) {
      setSummary(event.summary);
      setProgress((current) => ({ ...current, percent: 100 }));
      setPhase("done");
      return;
    }
    if (event.type === "error" || event.state === "failed") {
      setFailure(event.error ?? "the run stopped without saying why");
      setPhase("failed");
      return;
    }
    if (event.state === "running") setPhase("running");
  }, []);

  const start = useCallback(async () => {
    if (!connection || !verdict?.ok) return;
    setFailure(null);
    setLog([]);
    setProgress(EMPTY_PROGRESS);

    const controller = new AbortController();
    aborter.current = controller;

    try {
      const created = await createJob(connection.base, stage, picked, options);
      setJob(created);
      setPhase("uploading");
      setUpload({ sent: 0, total: created.files.length, current: "" });

      await uploadFiles(
        connection.base,
        created,
        picked,
        (sent, name) => setUpload({ sent, total: created.files.length, current: name }),
        controller.signal,
      );

      setPhase("running");
      unfollow.current = follow(connection.base, created.id, onEvent);
      await startJob(connection.base, created.id);
    } catch (error) {
      if (controller.signal.aborted) return;
      // The stream may already be open — start is the last step — so it has to
      // come down with the rest rather than being left listening to nothing.
      unfollow.current?.();
      unfollow.current = null;
      setFailure(describe(error));
      setPhase("failed");
    }
  }, [connection, verdict, stage, picked, options, onEvent]);

  const cancel = useCallback(async () => {
    unfollow.current?.();
    aborter.current?.abort();
    if (connection && job) await discardJob(connection.base, job.id);
    reset();
  }, [connection, job, reset]);

  const discardResult = useCallback(async () => {
    if (connection && job) await discardJob(connection.base, job.id);
    reset();
  }, [connection, job, reset]);

  // ------------------------------------------------------------------

  const totalBytes = useMemo(
    () => (verdict?.accepted ?? []).reduce((sum, file) => sum + file.size, 0),
    [verdict],
  );

  if (phase === "connecting" || phase === "offline") {
    return <ConnectCard state={phase} onRetry={() => void connect()} />;
  }

  if (phase === "uploading" || phase === "running") {
    return (
      <RunProgress
        phase={phase}
        progress={progress}
        log={log}
        upload={phase === "uploading" ? upload : null}
        onCancel={() => void cancel()}
      />
    );
  }

  if (phase === "done" && summary && connection && job) {
    return (
      <RunResult
        summary={summary}
        downloadUrl={resultUrl(connection.base, job.id)}
        onAgain={reset}
        onDiscard={() => void discardResult()}
      />
    );
  }

  if (phase === "failed") {
    return <Failure message={failure} onReset={reset} />;
  }

  const health = connection!.health;

  return (
    <div className="space-y-8">
      <Connected health={health} />

      <FolderDrop
        busy={phase === "checking"}
        verdict={verdict}
        onPick={(files) => void onPick(files)}
        limits={health.limits}
        suffixes={health.acceptedSuffixes}
      />

      {phase === "ready" && verdict?.ok ? (
        <>
          <div className="rounded-xl border border-paper-200 bg-paper-100/50 px-5 py-4 dark:border-ink-800 dark:bg-ink-900/30">
            <p className="text-ink-900 dark:text-paper-100">
              <span className="font-mono font-medium">{folderNameOf(picked)}</span> —{" "}
              {plural(verdict.accepted.length, "image")}, {formatBytes(totalBytes)}. Every
              file in it is an image.
            </p>
          </div>

          <OptionsForm
            health={health}
            stage={stage}
            options={options}
            onStage={setStage}
            onOptions={setOptions}
            disabled={false}
          />

          <div className="flex flex-wrap items-center gap-4 border-t border-paper-200 pt-7 dark:border-ink-800">
            <button
              type="button"
              onClick={() => void start()}
              className="inline-flex items-center gap-2 rounded-lg bg-ember-700 px-6 py-3 font-medium text-paper-50 transition-colors hover:bg-ember-600 dark:bg-ember-400 dark:text-ink-950 dark:hover:bg-ember-500"
            >
              <Play className="h-5 w-5" aria-hidden="true" />
              Run RevelAI on {plural(verdict.accepted.length, "image")}
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-lg px-3 py-2.5 text-sm font-medium text-ink-700/80 transition-colors hover:text-ink-900 dark:text-paper-300/80 dark:hover:text-paper-100"
            >
              Choose a different folder
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
}

function Connected({ health }: { health: Health }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-emerald-600/30 bg-emerald-600/[0.06] px-5 py-3.5 text-sm dark:border-emerald-400/25">
      <span className="inline-flex items-center gap-2 font-medium text-ink-900 dark:text-paper-100">
        <ShieldCheck
          className="h-4 w-4 text-emerald-700 dark:text-emerald-400"
          aria-hidden="true"
        />
        Connected to {health.product} {health.version} on this computer
      </span>
      <span className="text-ink-700/85 dark:text-paper-300/85">
        {health.uploadsPhotographs
          ? `Warning: the ${health.backend} backend uploads your photographs to a third party.`
          : `Backend: ${health.backend} — nothing leaves your machine.`}
      </span>
    </div>
  );
}

function Failure({ message, onReset }: { message: string | null; onReset: () => void }) {
  return (
    <div
      role="alert"
      className="rounded-2xl border border-red-400/60 bg-red-500/[0.04] p-7 dark:border-red-500/40 sm:p-9"
    >
      <div className="flex items-start gap-4">
        <AlertTriangle
          className="mt-1 h-6 w-6 shrink-0 text-red-700 dark:text-red-400"
          aria-hidden="true"
        />
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
            The run stopped
          </h2>
          <p className="mt-2 max-w-prose text-ink-700/90 dark:text-paper-300/90">
            {message ?? "Something went wrong and the server did not say what."}
          </p>
          <p className="mt-3 max-w-prose text-sm text-ink-700/80 dark:text-paper-300/80">
            Your originals were not touched — RevelAI never writes into the folder
            you chose. The terminal running{" "}
            <code className="font-mono">revelai serve</code> will have more detail
            if you started it with <code className="font-mono">-v</code>.
          </p>
          <button
            type="button"
            onClick={onReset}
            className="mt-6 inline-flex items-center gap-2 rounded-lg bg-ink-900 px-4 py-2.5 text-sm font-medium text-paper-50 transition-colors hover:bg-ink-800 dark:bg-paper-100 dark:text-ink-900 dark:hover:bg-paper-200"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Start over
          </button>
        </div>
      </div>
    </div>
  );
}

function describe(error: unknown): string {
  if (error instanceof RunnerError) return error.message;
  if (error instanceof Error) {
    return error.message.includes("fetch")
      ? "lost contact with RevelAI on this machine. Is the terminal running revelai serve still open?"
      : error.message;
  }
  return String(error);
}

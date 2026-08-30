"use client";

import { AlertTriangle, FolderOpen, Info, Loader2 } from "lucide-react";
import { useRef, useState } from "react";
import {
  type FolderVerdict,
  type PickedFile,
  formatBytes,
  pickedFrom,
  pickedFromDrop,
  plural,
} from "@/lib/runner";

/**
 * Choosing a folder, and being told plainly if it is not one RevelAI will run
 * on.
 *
 * The verdict comes from the server rather than from a check written twice.
 * "Only images" is a promise the server has to keep whatever a page sends it,
 * so the page asks rather than deciding, and there is one set of rules to be
 * wrong about instead of two.
 */
export function FolderDrop({
  busy,
  verdict,
  onPick,
  limits,
  suffixes,
}: {
  busy: boolean;
  verdict: FolderVerdict | null;
  onPick: (files: PickedFile[]) => void;
  limits: { maxFiles: number; maxTotalBytes: number };
  suffixes: string[];
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const rejected = verdict && !verdict.ok;

  return (
    <div>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void pickedFromDrop(event.dataTransfer).then((files) => {
            if (files.length) onPick(files);
          });
        }}
        className={[
          "rounded-2xl border-2 border-dashed p-8 text-center transition-colors sm:p-12",
          dragging
            ? "border-ember-500 bg-ember-500/5"
            : rejected
              ? "border-red-400/70 bg-red-500/[0.03] dark:border-red-500/50"
              : "border-paper-300 bg-paper-100/50 dark:border-ink-700 dark:bg-ink-900/30",
        ].join(" ")}
      >
        {/*
          A folder picker is a plain file input with `webkitdirectory`. It is
          non-standard and it is also what every browser implements; the
          standard `showDirectoryPicker` exists in one engine. The input stays
          the mechanism, and dropping a folder is the enhancement on top.
        */}
        <input
          ref={input}
          type="file"
          className="sr-only"
          // The button below is the control. Leaving this in the tab order too
          // would announce the same action twice, so it is hidden from
          // assistive technology and removed from the tab order together -
          // aria-hidden on something still focusable is its own defect.
          tabIndex={-1}
          aria-hidden="true"
          multiple
          // @ts-expect-error - webkitdirectory has no typing, and is how folder
          // selection works in every browser that has it.
          webkitdirectory=""
          directory=""
          onChange={(event) => {
            if (event.target.files?.length) onPick(pickedFrom(event.target.files));
            event.target.value = "";
          }}
        />

        {busy ? (
          <div className="py-3">
            <Loader2
              className="mx-auto h-8 w-8 animate-spin text-ember-600 dark:text-ember-400"
              aria-hidden="true"
            />
            <p className="mt-4 font-medium text-ink-900 dark:text-paper-100">
              Checking the folder…
            </p>
            <p className="mt-1 text-sm text-ink-700/80 dark:text-paper-300/80">
              Reading the listing. Nothing has been sent anywhere.
            </p>
          </div>
        ) : (
          <>
            <FolderOpen
              className="mx-auto h-9 w-9 text-ember-600 dark:text-ember-400"
              aria-hidden="true"
            />
            <p className="mt-4 text-lg font-medium text-ink-900 dark:text-paper-100">
              Choose the folder of album pages
            </p>
            <p className="mx-auto mt-2 max-w-md text-sm text-ink-700/85 dark:text-paper-300/85">
              It must hold images and nothing else — no subfolders, no documents.
              Up to {plural(limits.maxFiles, "file")}, {formatBytes(limits.maxTotalBytes)} in
              total.
            </p>
            <button
              type="button"
              onClick={() => input.current?.click()}
              className="mt-6 inline-flex items-center gap-2 rounded-lg bg-ink-900 px-5 py-2.5 text-sm font-medium text-paper-50 transition-colors hover:bg-ink-800 dark:bg-paper-100 dark:text-ink-900 dark:hover:bg-paper-200"
            >
              <FolderOpen className="h-4 w-4" aria-hidden="true" />
              Choose folder
            </button>
            <p className="mt-4 text-xs text-ink-700/80 dark:text-paper-300/80">
              or drag one here · {suffixes.join(" ")}
            </p>
          </>
        )}
      </div>

      {rejected ? <Rejection verdict={verdict} /> : null}
      {verdict?.ok && verdict.skipped.length ? <Skipped verdict={verdict} /> : null}
    </div>
  );
}

function Rejection({ verdict }: { verdict: FolderVerdict }) {
  const shown = verdict.problems.slice(0, 12);
  return (
    <div
      role="alert"
      className="mt-5 rounded-xl border border-red-400/60 bg-red-500/[0.04] p-5 dark:border-red-500/40"
    >
      <div className="flex items-center gap-2.5">
        <AlertTriangle
          className="h-5 w-5 shrink-0 text-red-700 dark:text-red-400"
          aria-hidden="true"
        />
        <h3 className="font-semibold text-ink-900 dark:text-paper-100">
          This folder is not one RevelAI will run on
        </h3>
      </div>
      <p className="mt-2 text-sm text-ink-700/90 dark:text-paper-300/90">
        Nothing was uploaded. Move these out of the folder, or choose a different
        one:
      </p>
      <ul className="mt-3 space-y-1.5 text-sm">
        {shown.map((problem, i) => (
          <li key={`${problem.name}-${i}`} className="flex flex-wrap gap-x-2">
            <code className="font-mono text-ink-900 dark:text-paper-100">
              {problem.name || "this folder"}
            </code>
            <span className="text-ink-700/85 dark:text-paper-300/85">— {problem.reason}</span>
          </li>
        ))}
      </ul>
      {verdict.problems.length > shown.length ? (
        <p className="mt-3 text-sm text-ink-700/80 dark:text-paper-300/80">
          …and {verdict.problems.length - shown.length} more.
        </p>
      ) : null}
    </div>
  );
}

function Skipped({ verdict }: { verdict: FolderVerdict }) {
  return (
    <p className="mt-4 flex items-start gap-2 text-sm text-ink-700/80 dark:text-paper-300/80">
      <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>
        Ignoring {plural(verdict.skipped.length, "system file")} your desktop left
        in the folder ({verdict.skipped.map((s) => s.name).join(", ")}).
      </span>
    </p>
  );
}

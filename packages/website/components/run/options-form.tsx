"use client";

import { Scissors, Sparkles, Wand2 } from "lucide-react";
import type { Health, RunOptions, Stage } from "@/lib/runner";

/**
 * What to run, and with what.
 *
 * The restoration options only appear once a stage that uses them is chosen,
 * and the two that invent detail are only rendered at all when the server was
 * started with `--allow-generative`. A control that is going to be refused is
 * worse than no control: the person has already decided by the time they find
 * out. The server refuses them regardless — this is the interface agreeing
 * with the server, not the enforcement.
 */

const STAGES: {
  value: Stage;
  title: string;
  detail: string;
  icon: typeof Scissors;
}[] = [
  {
    value: "split",
    title: "Split only",
    detail:
      "Cut each print out of the page, cropped and deskewed. No colour, brightness or sharpening is touched.",
    icon: Scissors,
  },
  {
    value: "run",
    title: "Split, then restore",
    detail: "Both stages in sequence. The separated photographs are restored afterwards.",
    icon: Wand2,
  },
  {
    value: "enhance",
    title: "Restore only",
    detail: "For photographs that are already separated. Your files are never modified.",
    icon: Sparkles,
  },
];

export function OptionsForm({
  health,
  stage,
  options,
  onStage,
  onOptions,
  disabled,
}: {
  health: Health;
  stage: Stage;
  options: RunOptions;
  onStage: (stage: Stage) => void;
  onOptions: (options: RunOptions) => void;
  disabled: boolean;
}) {
  const restoring = stage !== "split";
  const set = <K extends keyof RunOptions>(key: K, value: RunOptions[K]) =>
    onOptions({ ...options, [key]: value });

  return (
    <div className="space-y-8">
      <fieldset disabled={disabled}>
        <legend className="text-sm font-semibold uppercase tracking-[0.14em] text-ember-700 dark:text-ember-400">
          What to run
        </legend>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {STAGES.map(({ value, title, detail, icon: Icon }) => (
            <label
              key={value}
              className={[
                "cursor-pointer rounded-xl border p-4 transition-colors",
                stage === value
                  ? "border-ember-500 bg-ember-500/[0.06]"
                  : "border-paper-200 bg-paper-100/50 hover:border-paper-300 dark:border-ink-800 dark:bg-ink-900/30 dark:hover:border-ink-700",
              ].join(" ")}
            >
              <span className="flex items-center gap-2">
                <input
                  type="radio"
                  name="stage"
                  value={value}
                  checked={stage === value}
                  onChange={() => onStage(value)}
                  className="h-4 w-4 accent-ember-600"
                />
                <Icon
                  className="h-4 w-4 text-ember-700 dark:text-ember-400"
                  aria-hidden="true"
                />
                <span className="font-medium text-ink-900 dark:text-paper-100">{title}</span>
              </span>
              <span className="mt-2 block text-sm text-ink-700/85 dark:text-paper-300/85">
                {detail}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {restoring ? (
        <fieldset disabled={disabled}>
          <legend className="text-sm font-semibold uppercase tracking-[0.14em] text-ember-700 dark:text-ember-400">
            Restoration
          </legend>
          <p className="mt-3 max-w-prose text-sm text-ink-700/85 dark:text-paper-300/85">
            Classical first: a per-channel white balance fixes most yellowed
            prints outright, offline, inventing nothing. A model is called only
            for what is left, using the{" "}
            <strong className="font-medium text-ink-900 dark:text-paper-100">
              {health.backend}
            </strong>{" "}
            backend this server was started with.
          </p>
          <div className="mt-4 space-y-2.5">
            <Toggle
              label="Colour cast and fading correction"
              hint="Classical, deterministic, offline. On by default."
              checked={options.color}
              onChange={(value) => set("color", value)}
            />
            <Toggle
              label="Remove noise and grain"
              checked={options.denoise}
              onChange={(value) => set("denoise", value)}
            />
            <Toggle
              label="Remove dust and scratches"
              checked={options.dust}
              onChange={(value) => set("dust", value)}
            />
            <div className="rounded-lg border border-paper-200 px-4 py-3 dark:border-ink-800">
              <label className="flex flex-wrap items-center justify-between gap-3 text-sm">
                <span className="font-medium text-ink-900 dark:text-paper-100">
                  Super-resolution
                </span>
                <select
                  value={options.upscale}
                  onChange={(event) =>
                    set("upscale", Number(event.target.value) as RunOptions["upscale"])
                  }
                  className="rounded-md border border-paper-300 bg-paper-50 px-2.5 py-1.5 text-sm text-ink-900 dark:border-ink-700 dark:bg-ink-900 dark:text-paper-100"
                >
                  <option value={0}>Off</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                </select>
              </label>
            </div>

            {health.allowGenerative ? (
              <div className="rounded-lg border border-ember-500/50 bg-ember-500/[0.05] p-4">
                <p className="text-sm font-semibold text-ink-900 dark:text-paper-100">
                  These two invent detail that was not in the photograph
                </p>
                <p className="mt-1.5 text-sm text-ink-700/90 dark:text-paper-300/90">
                  Face restoration reconstructs a face; on a low-resolution print
                  it may not be that person’s face. Colourisation guesses the
                  colour. Your originals are never modified, and the results are
                  written separately — compare them before you keep anything.
                </p>
                <div className="mt-3 space-y-2.5">
                  <Toggle
                    label="Face restoration"
                    checked={options.faces}
                    onChange={(value) => set("faces", value)}
                  />
                  <Toggle
                    label="Colourise a black and white photograph"
                    checked={options.colorize}
                    onChange={(value) => set("colorize", value)}
                  />
                </div>
              </div>
            ) : (
              <p className="pt-1 text-sm text-ink-700/80 dark:text-paper-300/80">
                Face restoration and colourisation reconstruct and invent detail,
                so this server refuses them. Start it with{" "}
                <code className="font-mono">--allow-generative</code> if you want
                them, having read what they do.
              </p>
            )}
          </div>
        </fieldset>
      ) : null}
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 rounded-lg border border-paper-200 px-4 py-3 transition-colors hover:border-paper-300 dark:border-ink-800 dark:hover:border-ink-700">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 h-4 w-4 accent-ember-600"
      />
      <span>
        <span className="block text-sm font-medium text-ink-900 dark:text-paper-100">
          {label}
        </span>
        {hint ? (
          <span className="block text-sm text-ink-700/80 dark:text-paper-300/80">{hint}</span>
        ) : null}
      </span>
    </label>
  );
}

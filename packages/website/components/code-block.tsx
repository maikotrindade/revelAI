"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";

/**
 * A command with a copy button. The command itself is plain text in the markup,
 * so it is readable and selectable with JavaScript disabled; only the button is
 * an enhancement.
 */
export function CodeBlock({
  command,
  label,
}: {
  command: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="group relative">
      {label ? (
        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-700/80 dark:text-paper-300/75">
          {label}
        </p>
      ) : null}
      <div className="flex items-center gap-3 rounded-lg border border-paper-300 bg-paper-100 py-3 pl-4 pr-2 dark:border-ink-700 dark:bg-ink-900">
        {/* tabIndex: the command can scroll horizontally on a narrow screen,
            and a scrollable region has to be reachable from the keyboard. */}
        <code
          tabIndex={0}
          role="group"
          aria-label={label ? `${label} command` : "Command"}
          className="flex-1 overflow-x-auto whitespace-pre rounded font-mono text-sm text-ink-900 dark:text-paper-100"
        >
          <span aria-hidden="true" className="select-none text-ember-700 dark:text-ember-400">
            ${" "}
          </span>
          {command}
        </code>
        <button
          type="button"
          onClick={() => {
            navigator.clipboard?.writeText(command).then(
              () => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1600);
              },
              () => undefined,
            );
          }}
          className="shrink-0 rounded-md p-2 text-ink-700/70 transition-colors hover:bg-paper-200 hover:text-ink-900 dark:text-paper-300/70 dark:hover:bg-ink-800 dark:hover:text-paper-100"
          aria-label={copied ? "Copied" : `Copy: ${command}`}
        >
          {copied ? (
            <Check className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Copy className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
    </div>
  );
}

/**
 * Real output from the tool, shown as it appears. Static markup: nothing here
 * needs JavaScript.
 */
export function Terminal({
  title,
  lines,
}: {
  title?: string;
  lines: { text: string; tone?: "muted" | "flag" | "strong" }[];
}) {
  // This block is dark in both themes, so every tone here is light-on-dark.
  // Using the page's usual muted colour would put dark text on a dark panel.
  const tones = {
    muted: "text-paper-300/75",
    flag: "text-ember-400",
    strong: "text-paper-50",
  } as const;

  return (
    <div className="overflow-hidden rounded-xl border border-paper-300 bg-ink-950 dark:border-ink-700">
      {title ? (
        <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-white/20" aria-hidden="true" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/20" aria-hidden="true" />
          <span className="h-2.5 w-2.5 rounded-full bg-white/20" aria-hidden="true" />
          <span className="ml-2 font-mono text-xs text-paper-300/60">{title}</span>
        </div>
      ) : null}
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[13px] leading-relaxed">
        <code>
          {lines.map((line, i) => (
            <span
              key={i}
              className={`block ${line.tone ? tones[line.tone] : "text-paper-200"}`}
            >
              {line.text || " "}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

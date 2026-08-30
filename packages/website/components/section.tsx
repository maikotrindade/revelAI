import { cn } from "@/lib/cn";

export function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
  className,
  tone = "default",
}: {
  id?: string;
  eyebrow?: string;
  title: string;
  lead?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  tone?: "default" | "raised";
}) {
  return (
    <section
      id={id}
      aria-labelledby={id ? `${id}-heading` : undefined}
      className={cn(
        "scroll-mt-20 border-t border-paper-200 py-20 dark:border-ink-800",
        tone === "raised" && "bg-paper-100/50 dark:bg-ink-900/30",
        className,
      )}
    >
      <div className="mx-auto max-w-6xl px-5">
        {eyebrow ? (
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-ember-700 dark:text-ember-400">
            {eyebrow}
          </p>
        ) : null}
        <h2
          id={id ? `${id}-heading` : undefined}
          className="max-w-prose text-balance text-3xl font-semibold tracking-tight text-ink-900 dark:text-paper-100 sm:text-4xl"
        >
          {title}
        </h2>
        {lead ? (
          <div className="mt-4 max-w-prose text-lg leading-relaxed text-ink-700/90 dark:text-paper-300/90">
            {lead}
          </div>
        ) : null}
        {children ? <div className="mt-10">{children}</div> : null}
      </div>
    </section>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { LIMITATIONS, REPO } from "@/lib/content";

export const metadata: Metadata = {
  title: "Honest limitations — RevelAI",
  description:
    "What RevelAI gets wrong, in full: face restoration reconstructs faces rather than revealing them, colourisation invents colour, and detection fails on low-contrast pages.",
};

export default function HonestLimitations() {
  return (
    <article className="mx-auto max-w-3xl px-5 py-20">
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 rounded text-sm text-ink-700/80 underline underline-offset-4 dark:text-paper-300/80"
      >
        <ArrowLeft className="h-4 w-4" aria-hidden="true" /> Back
      </Link>

      <h1 className="mt-8 text-balance text-4xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
        Honest limitations
      </h1>
      <p className="mt-5 text-lg leading-relaxed text-ink-700/90 dark:text-paper-300/90">
        A restoration tool that only advertises its successes is not one you
        should trust with an album. This page is the full version of what
        RevelAI gets wrong, and what it does about it.
      </p>

      <div className="mt-14 space-y-14">
        {LIMITATIONS.map((limitation) => (
          <section key={limitation.id} id={limitation.id} className="scroll-mt-24">
            <h2
              className={`text-balance text-2xl font-semibold tracking-tight ${
                limitation.severity === "serious"
                  ? "text-ember-700 dark:text-ember-400"
                  : "text-ink-900 dark:text-paper-100"
              }`}
            >
              {limitation.title}
            </h2>
            <div className="mt-4 space-y-4 leading-relaxed text-ink-700/90 dark:text-paper-300/90">
              {limitation.body.map((paragraph) => (
                <p key={paragraph.slice(0, 40)}>{paragraph}</p>
              ))}
            </div>
          </section>
        ))}
      </div>

      <section className="mt-16 rounded-xl border border-paper-200 bg-paper-100/60 p-7 dark:border-ink-800 dark:bg-ink-900/40">
        <h2 className="text-lg font-semibold text-ink-900 dark:text-paper-100">
          The invariant behind all of this
        </h2>
        <p className="mt-3 leading-relaxed text-ink-700/90 dark:text-paper-300/90">
          Detection will get things wrong. What it must never do is get
          something wrong <em>quietly</em>, because then a fifty-page batch
          cannot be trusted at all. The test suite enforces it across every
          fixture: <strong>no crop is ever both wrong and unflagged.</strong>
        </p>
        <p className="mt-4 text-sm text-ink-700/80 dark:text-paper-300/80">
          The measurements behind these numbers are in the{" "}
          <a
            href={`${REPO}/blob/main/packages/engine/README.md`}
            className="rounded underline underline-offset-4"
          >
            engine README
          </a>
          , and the tests that hold them are in{" "}
          <code className="font-mono">packages/engine/tests</code>.
        </p>
      </section>
    </article>
  );
}

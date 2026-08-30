import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-5 py-32 text-center">
      <p className="font-mono text-sm text-ember-700 dark:text-ember-400">404</p>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
        That page is not here.
      </h1>
      <p className="mx-auto mt-4 max-w-prose text-ink-700/90 dark:text-paper-300/90">
        Nothing was cropped off — this address simply does not exist.
      </p>
      <Link
        href="/"
        className="mt-8 inline-block rounded-full border border-paper-300 px-5 py-2 font-medium text-ink-900 dark:border-ink-700 dark:text-paper-100"
      >
        Back to the start
      </Link>
    </div>
  );
}

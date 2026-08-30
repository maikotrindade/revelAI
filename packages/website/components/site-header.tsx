import Link from "next/link";
import { REPO } from "@/lib/content";
import { ThemeToggle } from "./theme-toggle";

const LINKS = [
  { href: "/#how-it-works", label: "How it works" },
  { href: "/#verify", label: "Trusting a batch" },
  { href: "/#privacy", label: "Privacy" },
  { href: "/honest-limitations/", label: "Limitations" },
  { href: "/#install", label: "Install" },
  { href: "/run/", label: "Run it" },
];

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-paper-200/80 bg-paper-50/85 backdrop-blur dark:border-ink-800 dark:bg-ink-950/85">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-5">
        <Link
          href="/"
          className="shrink-0 rounded font-semibold tracking-tight text-ink-900 dark:text-paper-100"
        >
          Revel<span className="text-ember-700 dark:text-ember-400">AI</span>
        </Link>

        <nav aria-label="Main" className="hidden flex-1 md:block">
          <ul className="flex items-center gap-5 text-sm">
            {LINKS.map((link) => (
              <li key={link.href}>
                <Link
                  href={link.href}
                  className="rounded text-ink-700 transition-colors hover:text-ink-900 dark:text-paper-300 dark:hover:text-paper-100"
                >
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <ThemeToggle />
          <a
            href={REPO}
            className="rounded-full border border-paper-300 px-3 py-1.5 text-sm font-medium text-ink-800 transition-colors hover:border-ink-800 dark:border-ink-700 dark:text-paper-200 dark:hover:border-paper-300"
          >
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}

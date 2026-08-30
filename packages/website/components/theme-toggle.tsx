"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/cn";

type Theme = "light" | "dark" | "system";

const OPTIONS: { value: Theme; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "system", label: "System", Icon: Monitor },
  { value: "dark", label: "Dark", Icon: Moon },
];

function apply(theme: Theme) {
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  if (theme === "system") {
    localStorage.removeItem("revelai-theme");
  } else {
    root.classList.add(theme);
    localStorage.setItem("revelai-theme", theme);
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("revelai-theme");
    setTheme(stored === "dark" || stored === "light" ? stored : "system");
    setReady(true);
  }, []);

  return (
    <div
      role="radiogroup"
      aria-label="Colour theme"
      className="flex items-center gap-0.5 rounded-full border border-paper-300 bg-paper-100 p-0.5 dark:border-ink-700 dark:bg-ink-900"
    >
      {OPTIONS.map(({ value, label, Icon }) => {
        const active = ready && theme === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => {
              setTheme(value);
              apply(value);
            }}
            className={cn(
              "rounded-full p-1.5 transition-colors",
              active
                ? "bg-paper-50 text-ink-900 shadow-sm dark:bg-ink-700 dark:text-paper-100"
                : "text-ink-700/60 hover:text-ink-900 dark:text-paper-300/60 dark:hover:text-paper-100",
            )}
          >
            <Icon className="h-4 w-4" aria-hidden="true" />
          </button>
        );
      })}
    </div>
  );
}

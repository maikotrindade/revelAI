/**
 * Applies a stored theme choice before the first paint.
 *
 * Only a *stored* choice. With no stored preference nothing is written to the
 * root, and the CSS media query decides — which is what keeps the site correct
 * for a visitor with JavaScript disabled, rather than stranding them in light
 * mode. See the darkMode strategy in tailwind.config.ts.
 */
const script = `
(function () {
  try {
    var stored = localStorage.getItem("revelai-theme");
    if (stored === "dark" || stored === "light") {
      document.documentElement.classList.add(stored);
    }
  } catch (e) {}
})();
`;

export function ThemeScript() {
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}

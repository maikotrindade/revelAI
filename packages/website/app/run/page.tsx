import type { Metadata } from "next";
import Link from "next/link";
import { Cpu, HardDrive, Lock, Terminal as TerminalIcon } from "lucide-react";
import { CodeBlock } from "@/components/code-block";
import { RunPanel } from "@/components/run/run-panel";
import { Section } from "@/components/section";
import { Terminal } from "@/components/terminal";
import { REPO } from "@/lib/content";

export const metadata: Metadata = {
  title: "Run RevelAI — on your own computer, from this page",
  description:
    "Choose a folder of album pages, watch RevelAI split and restore them, and download the results as a zip. The processing happens on your machine; this page has no server and never receives an image.",
};

/**
 * The run page.
 *
 * The rest of the site is static because RevelAI's promise is that family
 * photographs stay on the owner's machine, and a site that could accept an
 * upload would contradict the thing the product sells. This page does not
 * change that. It is the same static export, and it still has nowhere to send
 * an image: the only address it ever talks to is 127.0.0.1, which is the
 * visitor's own computer. The work is done by `revelai serve`, a process the
 * visitor started themselves and can stop with Ctrl+C.
 *
 * Everything above the interactive panel is plain markup, so a visitor with
 * JavaScript disabled still gets the whole explanation and the commands that
 * do the same job in a terminal.
 */
export default function RunPage() {
  return (
    <>
      <section className="border-b border-paper-200 dark:border-ink-800">
        <div className="mx-auto max-w-6xl px-5 py-16 sm:py-20">
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-paper-300 bg-paper-100 px-3 py-1 text-xs font-medium text-ink-700 dark:border-ink-700 dark:bg-ink-900 dark:text-paper-300">
            <Lock className="h-3.5 w-3.5" aria-hidden="true" />
            Runs on your machine · nothing is uploaded
          </p>
          <h1 className="max-w-3xl text-balance text-4xl font-semibold leading-[1.12] tracking-tight text-ink-900 dark:text-paper-100 sm:text-5xl">
            Run RevelAI without opening a terminal.
          </h1>
          <p className="mt-6 max-w-prose text-lg leading-relaxed text-ink-700/90 dark:text-paper-300/90">
            Choose a folder of album pages, watch each one being separated, and
            download the photographs as a zip. The page you are reading is a
            static file with no server behind it. The work is done by a copy of
            RevelAI running on your own computer, which you start and stop
            yourself.
          </p>
        </div>
      </section>

      <section
        id="run"
        aria-labelledby="run-heading"
        className="border-b border-paper-200 py-14 dark:border-ink-800"
      >
        <div className="mx-auto max-w-6xl px-5">
          <h2 id="run-heading" className="sr-only">
            Run RevelAI
          </h2>

          <noscript>
            <div className="rounded-2xl border border-paper-200 bg-paper-100/60 p-7 dark:border-ink-800 dark:bg-ink-900/40">
              <h3 className="text-xl font-semibold tracking-tight text-ink-900 dark:text-paper-100">
                This part needs JavaScript. The terminal does not.
              </h3>
              <p className="mt-2 max-w-prose text-ink-700/90 dark:text-paper-300/90">
                Driving a run needs a live progress connection, so the panel
                below is the one part of this site that will not work without
                JavaScript. Everything it does, these two commands also do:
              </p>
              <div className="mt-6 max-w-lg space-y-3">
                <CodeBlock command="pip install revelai" />
                <CodeBlock command="revelai split ./album-pages -o ./photos" />
              </div>
            </div>
          </noscript>

          <RunPanel />
        </div>
      </section>

      {/* How it works, in three sentences and a picture of the flow. */}
      <Section
        id="how-run-works"
        eyebrow="What is actually happening"
        title="Your photographs never cross a network."
        lead={
          <>
            Three programs are involved and two of them are on your computer. It
            is worth a paragraph, because “a website that runs software on your
            folder” is exactly the sentence that should make somebody suspicious.
          </>
        }
      >
        <div className="grid gap-6 md:grid-cols-3">
          <Step
            icon={HardDrive}
            title="This page"
            body="A static file served from GitHub Pages. It has no backend, no database and no upload endpoint. It cannot receive an image; there is nowhere for one to go."
          />
          <Step
            icon={TerminalIcon}
            title="revelai serve"
            body="A small server you start yourself, bound to 127.0.0.1. That address is your own machine and is not reachable from your network, let alone the internet. It refuses requests from any page but this one."
          />
          <Step
            icon={Cpu}
            title="The engine"
            body="The same Python code the command line runs, doing the same work with the same output. The browser is a remote control for it, not a replacement."
          />
        </div>

        <div className="mt-10 grid gap-8 lg:grid-cols-2">
          <div>
            <h3 className="text-lg font-semibold text-ink-900 dark:text-paper-100">
              What the page sends where
            </h3>
            <ul className="mt-4 space-y-3 text-ink-700/90 dark:text-paper-300/90">
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  Your images
                </strong>{" "}
                go to <code className="font-mono">http://127.0.0.1:8765</code>,
                which is a program on your own computer. They are written to a
                temporary folder that is deleted when you stop the server.
              </li>
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  Nothing else goes anywhere.
                </strong>{" "}
                No analytics, no error reporting, no fonts from a third party.
                You can watch this in your browser’s network tab, and you should.
              </li>
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  The one exception is a hosted backend
                </strong>{" "}
                — and it cannot be turned on from this page. Whether photographs
                are uploaded is decided by the flag you start the server with, not
                by anything a web page asks for.
              </li>
            </ul>
          </div>

          <Terminal
            title="revelai serve"
            lines={[
              { text: "$ revelai serve --open", tone: "strong" },
              { text: "RevelAI 0.1.0 - split album pages, then restore the photographs" },
              { text: "" },
              { text: "Listening on http://127.0.0.1:8765" },
              { text: "Open:        https://maikotrindade.com/revelAI/run/" },
              {
                text: "Settings:    backend: local (local only) | faces and colourisation: refused",
                tone: "muted",
              },
              { text: "" },
              {
                text: "Your photographs are processed here, on this machine, and are not",
                tone: "muted",
              },
              { text: "uploaded anywhere.", tone: "muted" },
              { text: "" },
              { text: "Press Ctrl+C to stop.", tone: "muted" },
            ]}
          />
        </div>
      </Section>

      {/* The folder rule, stated before somebody hits it. */}
      <Section
        id="folder-rules"
        tone="raised"
        eyebrow="Before you start"
        title="The folder has to hold images and nothing else."
        lead="Checked before a single byte moves, and checked again on the way in. A folder that is not right is refused with the name of every file that is wrong."
      >
        <div className="grid gap-8 lg:grid-cols-2">
          <div>
            <h3 className="text-lg font-semibold text-ink-900 dark:text-paper-100">
              What is checked
            </h3>
            <ul className="mt-4 space-y-2.5 text-ink-700/90 dark:text-paper-300/90">
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  Every file is an image.
                </strong>{" "}
                First by extension, from the listing, before anything is sent.
                Then by decoding the bytes, because a name is a claim rather than
                evidence — <code className="font-mono">holiday.jpg</code> that is
                really a spreadsheet is caught here.
              </li>
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  No subfolders.
                </strong>{" "}
                Flattening a tree quietly would produce name collisions nobody
                asked for, so a nested folder is reported instead.
              </li>
              <li>
                <strong className="font-semibold text-ink-900 dark:text-paper-100">
                  No two files with one name
                </strong>
                , and nothing empty.
              </li>
              <li>
                <code className="font-mono">.DS_Store</code>,{" "}
                <code className="font-mono">Thumbs.db</code> and their relatives
                are skipped rather than refused. Your desktop put them there; that
                is not you doing anything wrong.
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-ink-900 dark:text-paper-100">
              And what comes back
            </h3>
            <p className="mt-4 text-ink-700/90 dark:text-paper-300/90">
              A zip holding <strong className="font-semibold">images only</strong>
              , numbered continuously —{" "}
              <code className="font-mono">photo_00000001.png</code> onwards. No
              logs, no JSON, no thumbnails, no folders. Every file in it is a
              lossless PNG carrying the source page’s colour profile and a record
              of what was done to it.
            </p>
            <p className="mt-4 text-ink-700/90 dark:text-paper-300/90">
              Pages RevelAI is not confident about are listed on the results
              screen by name and reason. They are still written — a flagged crop
              is usually the right crop — but you are told, which is the whole
              point.
            </p>
            <p className="mt-6 text-sm text-ink-700/80 dark:text-paper-300/80">
              Working through a big album? Read{" "}
              <Link
                href="/#photographing"
                className="rounded font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
              >
                how to photograph album pages
              </Link>{" "}
              first. It makes more difference than any setting here.
            </p>
          </div>
        </div>
      </Section>

      {/* The same thing from a terminal, for anyone who would rather. */}
      <Section
        id="terminal-equivalent"
        eyebrow="Or don’t use a browser"
        title="Every button here is a flag there."
        lead="Nothing on this page can do something the command line cannot. If you are comfortable in a terminal, it is the better tool: it takes paths, writes where you tell it, and has options this page does not expose."
      >
        <div className="grid max-w-3xl gap-3">
          <CodeBlock label="Split only" command="revelai split ./album-pages -o ./photos" />
          <CodeBlock
            label="Split, then restore"
            command="revelai run ./album-pages --out-split ./photos --out-enhanced ./restored"
          />
          <CodeBlock
            label="Restore only"
            command="revelai enhance ./photos -o ./restored --compare-dir ./compare"
          />
          <CodeBlock label="Serve this page’s engine" command="revelai serve --open" />
        </div>
        <p className="mt-6 max-w-prose text-ink-700/90 dark:text-paper-300/90">
          The full flag reference, the detection write-up and the honest
          limitations are in the{" "}
          <a
            href={`${REPO}/tree/main/packages/engine#readme`}
            className="rounded font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
          >
            engine README
          </a>
          . The limitations are worth reading before you restore anything you
          care about:{" "}
          <Link
            href="/honest-limitations/"
            className="rounded font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
          >
            face restoration reconstructs faces rather than revealing them
          </Link>
          , and colourisation invents the colour.
        </p>
      </Section>
    </>
  );
}

function Step({
  icon: Icon,
  title,
  body,
}: {
  icon: typeof Cpu;
  title: string;
  body: string;
}) {
  return (
    <article className="rounded-xl border border-paper-200 bg-paper-100/60 p-6 dark:border-ink-800 dark:bg-ink-900/40">
      <Icon className="h-6 w-6 text-ember-700 dark:text-ember-400" aria-hidden="true" />
      <h3 className="mt-4 font-semibold text-ink-900 dark:text-paper-100">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-ink-700/90 dark:text-paper-300/90">
        {body}
      </p>
    </article>
  );
}

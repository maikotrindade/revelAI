import Link from "next/link";
import { ArrowRight, Check, Scissors, Sparkles, X } from "lucide-react";
import { CodeBlock } from "@/components/code-block";
import { Figure } from "@/components/figure";
import { Section } from "@/components/section";
import { Terminal } from "@/components/terminal";
import {
  LIMITATIONS,
  MEASURED,
  PHOTOGRAPHY_TIPS,
  PRIOR_ART,
  REPO,
} from "@/lib/content";

const prefix = process.env.NEXT_PUBLIC_BASE_PATH || "";
const img = (name: string) => `${prefix}/img/${name}`;

export default function Home() {
  return (
    <>
      {/* 5.1 Hero */}
      <section className="border-b border-paper-200 dark:border-ink-800">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:py-28">
          <div className="grid items-center gap-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
            <div>
              <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-paper-300 bg-paper-100 px-3 py-1 text-xs font-medium text-ink-700 dark:border-ink-700 dark:bg-ink-900 dark:text-paper-300">
                Open source · MIT
              </p>
              <h1 className="text-balance text-4xl font-semibold leading-[1.1] tracking-tight text-ink-900 dark:text-paper-100 sm:text-5xl">
                Split album page scans into individual photos, then restore them
                with AI.
              </h1>
              <p className="mt-6 max-w-prose text-lg leading-relaxed text-ink-700/90 dark:text-paper-300/90">
                You photograph a page of a family album with your phone. The page
                holds five prints at different sizes and angles, and they are
                faded, yellowed and scratched. RevelAI separates them and — as a
                separate step that never touches your originals — restores them.
              </p>
              <div className="mt-8 max-w-md space-y-3">
                <CodeBlock command="pip install revelai" />
                <CodeBlock command="revelai split ./album-pages -o ./photos" />
              </div>
              <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-ink-700/80 dark:text-paper-300/80">
                <Link
                  href="/#how-it-works"
                  className="inline-flex items-center gap-1.5 rounded font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
                >
                  How it works <ArrowRight className="h-4 w-4" aria-hidden="true" />
                </Link>
                <span>
                  Works offline. No API key. Nothing leaves your machine.
                </span>
              </div>
            </div>

            <Figure
              src={img("split-example.png")}
              width={1092}
              height={673}
              priority
              alt="On the left, one phone photograph of an album page holding four landscape prints on pale paper. On the right, the same four photographs separated into individual images, each straightened and cropped to its own border."
              caption="One photograph of a page in, four deskewed photographs out."
            />
          </div>
        </div>
      </section>

      {/* 5.2 The two stages */}
      <Section
        id="stages"
        eyebrow="Two stages"
        title="Separating and restoring are different jobs, so they are different commands."
        lead="Either runs on its own, in whatever order you like. split never depends on enhance."
      >
        <div className="grid gap-6 md:grid-cols-2">
          <article className="rounded-xl border border-paper-200 bg-paper-100/60 p-7 dark:border-ink-800 dark:bg-ink-900/40">
            <Scissors
              className="h-6 w-6 text-ember-700 dark:text-ember-400"
              aria-hidden="true"
            />
            <h3 className="mt-4 font-mono text-lg font-semibold text-ink-900 dark:text-paper-100">
              split
            </h3>
            <p className="mt-3 text-ink-700/90 dark:text-paper-300/90">
              Takes photographs of album pages and returns each individual print,
              cropped and deskewed, with{" "}
              <strong className="font-semibold text-ink-900 dark:text-paper-100">
                no quality loss whatsoever
              </strong>
              .
            </p>
            <ul className="mt-5 space-y-2 text-sm text-ink-700/90 dark:text-paper-300/90">
              {[
                `Exactly ${MEASURED.resamplingsPerPhoto} resampling per photograph`,
                "Lossless PNG, source ICC profile preserved",
                "16-bit input stays 16-bit",
                "Provenance written into the file's metadata",
              ].map((item) => (
                <li key={item} className="flex gap-2.5">
                  <Check
                    className="mt-0.5 h-4 w-4 shrink-0 text-ember-700 dark:text-ember-400"
                    aria-hidden="true"
                  />
                  {item}
                </li>
              ))}
            </ul>
            <p className="mt-5 border-t border-paper-200 pt-4 text-sm text-ink-700/80 dark:border-ink-800 dark:text-paper-300/80">
              And what it does <em>not</em> do: no colour, brightness, contrast,
              saturation, sharpening or noise adjustment. It crops and deskews.
              That is the whole contract.
            </p>
          </article>

          <article className="rounded-xl border border-paper-200 bg-paper-100/60 p-7 dark:border-ink-800 dark:bg-ink-900/40">
            <Sparkles
              className="h-6 w-6 text-ember-700 dark:text-ember-400"
              aria-hidden="true"
            />
            <h3 className="mt-4 font-mono text-lg font-semibold text-ink-900 dark:text-paper-100">
              enhance
            </h3>
            <p className="mt-3 text-ink-700/90 dark:text-paper-300/90">
              Takes separated photographs and returns restored versions,{" "}
              <strong className="font-semibold text-ink-900 dark:text-paper-100">
                never touching the originals
              </strong>
              .
            </p>
            <ul className="mt-5 space-y-2 text-sm text-ink-700/90 dark:text-paper-300/90">
              {[
                "Classical correction first; a model only for what remains",
                "Writes to a different folder, always",
                "Filenames are preserved, so the two correspond",
                "Records every operation and model version applied",
              ].map((item) => (
                <li key={item} className="flex gap-2.5">
                  <Check
                    className="mt-0.5 h-4 w-4 shrink-0 text-ember-700 dark:text-ember-400"
                    aria-hidden="true"
                  />
                  {item}
                </li>
              ))}
            </ul>
            <p className="mt-5 border-t border-paper-200 pt-4 text-sm text-ink-700/80 dark:border-ink-800 dark:text-paper-300/80">
              Pointing the output at the input folder is an error, checked before
              any work starts. Your originals are the only copy nobody has
              altered.
            </p>
          </article>
        </div>
      </Section>

      {/* 5.3 How the detection works */}
      <Section
        id="how-it-works"
        eyebrow="Detection"
        title="Finding the photographs is easy. Finding their edges is the problem."
        lead={
          <>
            RevelAI fits <strong>one rotated rectangle</strong> to each print by
            line integral, scoring each edge by integrating the image gradient
            along the <em>whole</em> edge. That lets a long straight line — the
            real border of the print — outscore any short high-contrast detail
            inside the picture, and forcing a single rectangle stops one edge
            sliding onto a neighbouring photograph.
          </>
        }
        tone="raised"
      >
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
          <Figure
            src={img("detection-map.png")}
            width={620}
            height={827}
            alt="The detector's view of an album page: an orange outline around the page itself, green rectangles tightly around four correctly detected photographs, and a red rectangle around a fifth region that has been flagged for review."
            caption="Green is accepted, red is flagged for review, cyan is the crop after the safety inset, orange is the page. Written by --debug-dir."
          />
          <div className="space-y-6">
            <dl className="grid gap-4 sm:grid-cols-2">
              {[
                {
                  term: `IoU ${MEASURED.iouLow}–${MEASURED.iouHigh}`,
                  desc: "against synthetic fixtures where the ground truth is exact",
                },
                {
                  term: `< ${MEASURED.angleErrorDegrees}° error`,
                  desc: "on the deskew angle",
                },
                {
                  term: `${MEASURED.tests} tests`,
                  desc: "all offline; none needs the network or an API key",
                },
                {
                  term: MEASURED.platforms,
                  desc: `on Python ${MEASURED.pythonVersions}`,
                },
              ].map((stat) => (
                <div
                  key={stat.term}
                  className="rounded-lg border border-paper-200 bg-paper-50 p-4 dark:border-ink-800 dark:bg-ink-950"
                >
                  <dt className="font-mono text-lg font-semibold text-ink-900 dark:text-paper-100">
                    {stat.term}
                  </dt>
                  <dd className="mt-1 text-sm text-ink-700/80 dark:text-paper-300/80">
                    {stat.desc}
                  </dd>
                </div>
              ))}
            </dl>
            <div className="rounded-lg border border-paper-200 bg-paper-50 p-6 dark:border-ink-800 dark:bg-ink-950">
              <h3 className="font-semibold text-ink-900 dark:text-paper-100">
                Why a phone photograph is harder than a scan
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-ink-700/90 dark:text-paper-300/90">
                Every other tool for this assumes a flatbed scanner with a
                uniform white background. On a phone photograph the page sits on
                a table, so the background is the <em>darkest</em> thing in the
                frame rather than the brightest, and a single global threshold
                segments the whole page as one object.
              </p>
              <p className="mt-3 text-sm leading-relaxed text-ink-700/90 dark:text-paper-300/90">
                The light is uneven too, so the album paper is not one colour.
                RevelAI fits a smooth illumination surface to the paper first, so
                that &ldquo;differs from the paper&rdquo; means &ldquo;differs
                from the paper <em>here</em>&rdquo;.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* 5.4 Trusting a batch */}
      <Section
        id="verify"
        eyebrow="Verification"
        title="Fifty pages produce two hundred crops. Nobody is going to check them by hand."
        lead={
          <>
            Without a check, the honest summary of such a run is{" "}
            <em>&ldquo;I processed fifty pages and I don&rsquo;t know whether it
            came out right.&rdquo;</em>{" "}
            <code className="rounded bg-paper-200 px-1.5 py-0.5 font-mono text-[0.9em] dark:bg-ink-800">
              --verify
            </code>{" "}
            asks a vision-language model one structured question about every
            crop, and turns that into something you can act on.
          </>
        }
      >
        <div className="grid gap-8 lg:grid-cols-2 lg:items-center">
          <Terminal
            title="revelai split ./album --verify"
            lines={[
              { text: "Summary", tone: "strong" },
              { text: "  pages read           50" },
              { text: "  photographs found    200" },
              { text: "  crops verified       196 of 200 passed", tone: "strong" },
              { text: "      photo_00000042.png: cut off at the left", tone: "flag" },
              {
                text: "      photo_00000078.png: more than one photograph in the crop",
                tone: "flag",
              },
              { text: "  pages needing review 2" },
              { text: "" },
              { text: "Re-run with --review to correct the flagged pages.", tone: "muted" },
            ]}
          />
          <div className="space-y-5 text-ink-700/90 dark:text-paper-300/90">
            <p>
              The model is asked to <strong>judge, never to change anything</strong>.
              Is this exactly one complete photograph? Is it cut off at any edge?
              Is there a leftover sliver of another photo, or of the page? Which
              way up is it?
            </p>
            <p>
              If no API key is set, verification says so and the run continues. A
              batch that could not be checked is reported as unverified — never
              as passing.
            </p>
            <p className="rounded-lg border-l-2 border-ember-500 bg-paper-100 px-5 py-4 text-sm dark:bg-ink-900">
              A vision-language model judges and describes. It is not the thing
              that restores an image — that is a different kind of model
              entirely, and it lives in a different stage.
            </p>
          </div>
        </div>
      </Section>

      {/* 5.5 Restoration */}
      <Section
        id="restore"
        eyebrow="Restoration"
        title="Classical first. A model only for what is left."
        lead={
          <>
            Most of what is wrong with an old print is that its dye layers faded
            at different rates. That is a per-channel gain problem, and a
            per-channel gain fixes it — instantly, offline, deterministically,
            inventing nothing. On the test fixture it takes the grey deviation of
            a strongly cast print from{" "}
            <strong>{MEASURED.greyDeviationBefore}</strong> to{" "}
            <strong>{MEASURED.greyDeviationAfter}</strong>, with no model
            involved.
          </>
        }
        tone="raised"
      >
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:items-center">
          <Figure
            src={img("restore-example.png")}
            width={712}
            height={304}
            alt="Two versions of the same synthetic landscape photograph side by side. The left is heavily yellowed with a warm orange cast; the right has neutral greens and blues after classical colour correction."
            caption="Before and after classical colour correction. No model, no network, fully deterministic."
          />
          <div>
            <h3 className="font-semibold text-ink-900 dark:text-paper-100">
              The order is the substance
            </h3>
            <ol className="mt-4 space-y-2.5 text-sm text-ink-700/90 dark:text-paper-300/90">
              {[
                "Classical colour correction",
                "Denoise",
                "Dust and scratch removal",
                "Colourisation (optional, off by default)",
                "Face restoration (optional, off by default)",
                "Super-resolution, last, always",
              ].map((step, i) => (
                <li key={step} className="flex gap-3">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-paper-200 font-mono text-[11px] font-semibold text-ink-800 dark:bg-ink-800 dark:text-paper-200">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
            <p className="mt-5 text-sm leading-relaxed text-ink-700/80 dark:text-paper-300/80">
              Super-resolution goes last because it is a magnifier. Run it first
              and every defect still in the photograph gets magnified along with
              the picture.
            </p>
          </div>
        </div>
      </Section>

      {/* 5.6 Privacy */}
      <Section
        id="privacy"
        eyebrow="Privacy"
        title="These are family photographs. That shapes the design."
        lead="It is not a footnote, and it is not a setting buried three menus deep."
      >
        <div className="grid gap-6 md:grid-cols-3">
          {[
            {
              title: "Local by default",
              body: "The local backend is the default and the whole tool works with no API key at all. Nothing uploaded, no account, no sign-up.",
            },
            {
              title: "Hosted backends ask first",
              body: "A hosted backend uploads your photographs to a third party. The first time one is used in a run, the tool says where the images are going and waits for a yes.",
            },
            {
              title: "Verification is opt-in",
              body: "--verify and --describe also send images, and they are off unless you ask for them. Only a downscaled copy is sent; the file on disk is never modified.",
            },
          ].map((card) => (
            <article
              key={card.title}
              className="rounded-xl border border-paper-200 bg-paper-100/60 p-6 dark:border-ink-800 dark:bg-ink-900/40"
            >
              <h3 className="font-semibold text-ink-900 dark:text-paper-100">
                {card.title}
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-ink-700/90 dark:text-paper-300/90">
                {card.body}
              </p>
            </article>
          ))}
        </div>
        <p className="mt-8 max-w-prose text-sm text-ink-700/80 dark:text-paper-300/80">
          This website has no upload box, no analytics and no backend. It is a
          static site — there is nowhere for it to send an image even in
          principle.
        </p>
      </Section>

      {/* 5.7 Honest limitations */}
      <Section
        id="limitations"
        eyebrow="Honest limitations"
        title="What it gets wrong, on the front page."
        lead="A restoration tool that only advertises its successes is not one you should trust with an album."
        tone="raised"
      >
        <div className="grid gap-5 md:grid-cols-2">
          {LIMITATIONS.slice(0, 4).map((limitation) => (
            <article
              key={limitation.id}
              className={`rounded-xl border p-6 ${
                limitation.severity === "serious"
                  ? "border-ember-500/40 bg-ember-400/[0.07]"
                  : "border-paper-200 bg-paper-50 dark:border-ink-800 dark:bg-ink-950"
              }`}
            >
              <div className="flex items-start gap-3">
                {limitation.severity === "serious" ? (
                  <X
                    className="mt-0.5 h-5 w-5 shrink-0 text-ember-700 dark:text-ember-400"
                    aria-hidden="true"
                  />
                ) : null}
                <h3 className="font-semibold text-ink-900 dark:text-paper-100">
                  {limitation.title}
                </h3>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-ink-700/90 dark:text-paper-300/90">
                {limitation.body[0]}
              </p>
            </article>
          ))}
        </div>
        <Link
          href="/honest-limitations/"
          className="mt-8 inline-flex items-center gap-1.5 rounded font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
        >
          Read all of them in full <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </Section>

      {/* 5.8 Install */}
      <Section
        id="install"
        eyebrow="Install"
        title="Three ways to get it."
        lead="The CLI ships through GitHub Packages as a container image, and as wheels attached to each release."
      >
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-3">
            <CodeBlock label="From PyPI" command="pip install revelai" />
            <p className="text-sm text-ink-700/80 dark:text-paper-300/80">
              Then <code className="font-mono">revelai split ./album -o ./photos</code>.
            </p>
          </div>
          <div className="space-y-3">
            <CodeBlock
              label="Container"
              command={'docker run --rm -v "$PWD:/work" ghcr.io/maikotrindade/revelai split ./album'}
            />
            <p className="text-sm text-ink-700/80 dark:text-paper-300/80">
              GitHub Packages has no Python registry, so the container registry
              is the GitHub Packages route.
            </p>
          </div>
          <div className="space-y-3">
            <CodeBlock label="From source" command="pip install -e packages/engine" />
            <p className="text-sm text-ink-700/80 dark:text-paper-300/80">
              Wheels are also attached to each{" "}
              <a
                href={`${REPO}/releases`}
                className="rounded underline underline-offset-4"
              >
                GitHub Release
              </a>
              .
            </p>
          </div>
        </div>
      </Section>

      {/* 5.9 How to photograph album pages */}
      <Section
        id="photographing"
        eyebrow="Before you start"
        title="How to photograph album pages."
        lead="The quality of the input decides everything downstream. Five minutes of care here saves an hour of review later."
        tone="raised"
      >
        <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
          {PHOTOGRAPHY_TIPS.map((tip, i) => (
            <div key={tip.title} className="flex gap-4">
              <span
                aria-hidden="true"
                className="font-mono text-sm text-ember-700 dark:text-ember-400/80"
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <div>
                <h3 className="font-medium text-ink-900 dark:text-paper-100">
                  {tip.title}
                </h3>
                <p className="mt-1.5 text-sm leading-relaxed text-ink-700/85 dark:text-paper-300/85">
                  {tip.body}
                </p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* 5.10 Prior art */}
      <Section
        id="prior-art"
        eyebrow="Prior art"
        title="Other tools solve part of this, and they are worth your attention."
        lead="All of them assume a flatbed scanner with a uniform white background, and none combines splitting with restoration. That is not a criticism — they are built for scans and they do that job."
      >
        <ul className="grid gap-4 md:grid-cols-2">
          {PRIOR_ART.map((project) => (
            <li
              key={project.name}
              className="rounded-lg border border-paper-200 bg-paper-100/60 p-5 dark:border-ink-800 dark:bg-ink-900/40"
            >
              <a
                href={project.href}
                className="rounded font-mono text-sm font-medium text-ink-900 underline underline-offset-4 dark:text-paper-100"
              >
                {project.name}
              </a>
              <p className="mt-2 text-sm text-ink-700/85 dark:text-paper-300/85">
                {project.note}
              </p>
            </li>
          ))}
        </ul>
      </Section>
    </>
  );
}

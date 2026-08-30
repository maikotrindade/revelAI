/**
 * Every factual claim the site makes, in one place.
 *
 * The rule from the brief: each number here is measured against the engine, and
 * nothing goes on the site that is not in this file. Keeping them together
 * means a claim can be checked against `packages/engine` without reading the
 * markup, and it makes it obvious when something is being asserted that nobody
 * ever measured.
 */

export const REPO = "https://github.com/maikotrindade/revelAI";

/** Measured against the engine's synthetic fixtures, where ground truth is exact. */
export const MEASURED = {
  iouLow: 0.994,
  iouHigh: 0.997,
  angleErrorDegrees: 0.03,
  greyDeviationBefore: 71,
  greyDeviationAfter: 7,
  tests: 239,
  resamplingsPerPhoto: 1,
  pythonVersions: "3.10–3.13",
  platforms: "Linux, macOS and Windows",
} as const;

export type Limitation = {
  id: string;
  title: string;
  severity: "serious" | "known";
  body: string[];
};

/**
 * These are not softened, and they are not in a footer. Face restoration and
 * colourisation invent detail, and somebody restoring a photograph of their
 * grandmother is entitled to know that before they keep the result.
 */
export const LIMITATIONS: Limitation[] = [
  {
    id: "faces",
    title: "Face restoration reconstructs faces. It does not reveal them.",
    severity: "serious",
    body: [
      "On a low-resolution photograph, the face that comes out may not be that person's face. The model produces a plausible face, not the one that was there.",
      "For a family photograph, where the entire value is that it is that specific person, this is a serious defect and not a footnote.",
      "So it is off by default, the tool prints the warning before running, and the restored file never replaces the original. Use --compare-dir and look at both before you keep anything.",
    ],
  },
  {
    id: "colour",
    title: "Colourisation invents the colour.",
    severity: "serious",
    body: [
      "The result is a plausible guess about what the scene might have looked like. It is not a record of it.",
      "A dress that comes out blue was not necessarily blue. Off by default, with the same kind of warning.",
    ],
  },
  {
    id: "contrast",
    title: "Detection fails on low-contrast pages.",
    severity: "known",
    body: [
      "A print faded to nearly the tone of the paper it is mounted on, on a page with a busy texture, may be missed or cropped short.",
      "That is why interactive review exists, and why it is a normal part of the flow rather than a fallback.",
    ],
  },
  {
    id: "touching",
    title: "Prints that touch or overlap may come out as one crop.",
    severity: "known",
    body: [
      "Two photographs mounted edge to edge are separated only by a shadow line, and where one print is laid over another the union is an L shape whose borders belong to two different rectangles. Locally there is nothing in the image that says which.",
      "Measured across the fixtures, a single photograph's strongest internal line reaches 0.287 of its own border — a horizon, a roofline or the edge of a table all produce one — and a genuine seam between two prints starts at 0.320. A margin of ten per cent is not enough to cut a photograph on.",
      "So RevelAI reports it rather than guessing. The invariant the tests enforce is that no crop is ever both wrong and unflagged.",
    ],
  },
  {
    id: "vlm",
    title: "A vision-language model can be wrong too.",
    severity: "known",
    body: [
      "Crop verification reduces the number of crops you have to look at. It does not reduce it to zero.",
    ],
  },
];

export const PHOTOGRAPHY_TIPS = [
  {
    title: "Indirect window light. No flash.",
    body: "A flash puts a hotspot in the middle of the page and a hard reflection on any glossy print.",
  },
  {
    title: "Phone parallel to the page.",
    body: "RevelAI corrects rotation in the plane. It cannot undo perspective from a camera held off to one side.",
  },
  {
    title: "Leave a margin.",
    body: "Get the whole page in frame with room around it. A print that runs off the edge of the photograph cannot be cropped whole.",
  },
  {
    title: "Dark, matte background.",
    body: "A table works. This is what lets the page be found at all.",
  },
  {
    title: "Half a page at a time.",
    body: "Photographing half a page doubles the resolution of every print on it. If the album is worth digitising, it is worth two exposures per page.",
  },
  {
    title: "Watch your own shadow.",
    body: "Leaning over the page casts a soft-edged shadow across it. RevelAI rejects soft edges as shadows rather than borders, but it is better not to make it guess.",
  },
];

export const PRIOR_ART = [
  {
    name: "z80z80z80/autocrop",
    href: "https://github.com/z80z80z80/autocrop",
    note: "The most popular of them. OpenCV, but one photograph per scan and it needs a rough pre-crop.",
  },
  {
    name: "Claytorpedo/scan-cropper",
    href: "https://github.com/Claytorpedo/scan-cropper",
    note: "Multiple photographs per image, white background only, no interactive review.",
  },
  {
    name: "idlerun/album-scan",
    href: "https://github.com/idlerun/album-scan",
    note: "Closest in intent: OpenCV plus a UI for fixing corners by hand. Its interaction model informed ours.",
  },
  {
    name: "alexhorn/AutoCrop",
    href: "https://github.com/alexhorn/AutoCrop",
    note: "Tied to ImageMagick 6, which is a practical blocker today.",
  },
  {
    name: "FrancoisMalan/DivideScannedImages",
    href: "https://github.com/FrancoisMalan/DivideScannedImages",
    note: "The classic GIMP script, and the historical reference for this problem.",
  },
];

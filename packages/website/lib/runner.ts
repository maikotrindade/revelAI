/**
 * The client for a RevelAI server running on the visitor's own machine.
 *
 * This is the only part of the site that makes a network request, and the only
 * address it will ever use is loopback. That is what lets a static site with no
 * backend offer a working run button without weakening the promise the whole
 * product rests on: the photographs are read by a process on the same computer
 * as the browser, and nothing crosses a network at all.
 *
 * The server is not assumed to be there. Everything here is written so that
 * "no server running" is an ordinary state with a good answer, not an error.
 */

export const DEFAULT_PORT = 8765;

/** Probed in order. Three, because a second server on 8765 is a real thing. */
const CANDIDATE_PORTS = [DEFAULT_PORT, DEFAULT_PORT + 1, DEFAULT_PORT + 2];

const PROBE_TIMEOUT_MS = 1500;

export type Health = {
  ok: true;
  product: string;
  version: string;
  stages: string[];
  backend: string;
  uploadsPhotographs: boolean;
  allowGenerative: boolean;
  limits: { maxFiles: number; maxFileBytes: number; maxTotalBytes: number };
  acceptedSuffixes: string[];
  jobs: number;
};

export type FolderProblem = { name: string; reason: string };

export type FolderVerdict = {
  ok: boolean;
  accepted: { name: string; size: number }[];
  skipped: FolderProblem[];
  problems: FolderProblem[];
  totalBytes: number;
};

export type Stage = "split" | "enhance" | "run";

export type RunOptions = {
  color: boolean;
  colorStrength: number;
  denoise: boolean;
  dust: boolean;
  upscale: 0 | 2 | 4;
  faces: boolean;
  colorize: boolean;
};

export type ProgressEvent = {
  seq: number;
  type: "state" | "progress" | "done" | "error";
  state?: string;
  stage?: "split" | "enhance";
  stageIndex?: number;
  stageCount?: number;
  done?: number;
  total?: number;
  unit?: string;
  percent?: number;
  current?: string | null;
  message?: string;
  photographs?: number;
  needsReview?: boolean;
  failed?: boolean;
  error?: string;
  summary?: RunSummary;
};

export type RunSummary = {
  stages: string[];
  seconds: number;
  product: string;
  split?: {
    pagesRead: number;
    photographs: number;
    pagesNeedingReview: { name: string; reasons: string[] }[];
    pagesFailed: { name: string; reason: string }[];
  };
  enhance?: {
    restored: number;
    backend: string;
    operations: string[];
    operationsSkipped: { name: string; reason: string }[];
    photosFailed: { name: string; reason: string }[];
  };
  result: {
    file: string;
    images: number;
    names: string[];
    bytes: number;
    bytesUncompressed: number;
  };
};

export class RunnerError extends Error {
  readonly code?: string;
  readonly verdict?: FolderVerdict;

  constructor(message: string, code?: string, verdict?: FolderVerdict) {
    super(message);
    this.name = "RunnerError";
    this.code = code;
    this.verdict = verdict;
  }
}

/** The files a folder picker handed us, in the shape the server expects. */
export type PickedFile = { file: File; name: string; relativePath: string; size: number };

export function pickedFrom(files: FileList | File[]): PickedFile[] {
  return Array.from(files).map((file) => ({
    file,
    name: file.name,
    // webkitRelativePath is "chosen-folder/page_1.jpg" for a folder pick and
    // empty for anything else. The server uses it to notice subfolders.
    relativePath:
      (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
    size: file.size,
  }));
}

/**
 * The files inside something dropped on the page.
 *
 * A dropped *folder* does not appear in `dataTransfer.files` as its contents —
 * it appears as one entry with no extension and no size, which the folder check
 * would then refuse with a baffling message about it not being an image. The
 * entries API is what actually reads it, so dropping a folder has to go through
 * that or the affordance is a lie.
 *
 * One level deep on purpose: a nested folder is something the server refuses,
 * and it has to still be *seen* in order to be refused with a reason.
 */
export async function pickedFromDrop(transfer: DataTransfer): Promise<PickedFile[]> {
  const entries = Array.from(transfer.items)
    .map((item) => item.webkitGetAsEntry?.() ?? null)
    .filter((entry): entry is FileSystemEntry => entry !== null);

  if (!entries.length) return pickedFrom(transfer.files);

  const readDir = (entry: FileSystemDirectoryEntry) =>
    new Promise<FileSystemEntry[]>((resolve) => {
      const reader = entry.createReader();
      const all: FileSystemEntry[] = [];
      // readEntries returns a page at a time and signals the end with an empty
      // batch, so one call is not enough for a folder of any size.
      const step = () =>
        reader.readEntries(
          (batch) => {
            if (!batch.length) return resolve(all);
            all.push(...batch);
            step();
          },
          () => resolve(all),
        );
      step();
    });

  const asFile = (entry: FileSystemFileEntry) =>
    new Promise<PickedFile | null>((resolve) => {
      entry.file(
        (file) =>
          resolve({
            file,
            name: file.name,
            relativePath: entry.fullPath.replace(/^\//, ""),
            size: file.size,
          }),
        () => resolve(null),
      );
    });

  // A folder inside the dropped folder is reported rather than walked: the
  // server refuses subfolders, and it can only say so about one it was told of.
  const asRefusable = (entry: FileSystemEntry): PickedFile => ({
    file: new File([], entry.name),
    name: entry.name,
    relativePath: entry.fullPath.replace(/^\//, ""),
    size: 0,
  });

  const out: PickedFile[] = [];
  for (const entry of entries) {
    if (entry.isFile) {
      const picked = await asFile(entry as FileSystemFileEntry);
      if (picked) out.push(picked);
      continue;
    }
    for (const child of await readDir(entry as FileSystemDirectoryEntry)) {
      const picked = child.isFile
        ? await asFile(child as FileSystemFileEntry)
        : asRefusable(child);
      if (picked) out.push(picked);
    }
  }
  return out;
}

export function folderNameOf(files: PickedFile[]): string {
  const first = files[0]?.relativePath ?? "";
  const parts = first.split("/");
  return parts.length > 1 ? parts[0] : "the chosen folder";
}

async function fetchJson<T>(url: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const controller = new AbortController();
  const timer =
    init?.timeoutMs != null ? setTimeout(() => controller.abort(), init.timeoutMs) : null;
  try {
    const response = await fetch(url, { ...init, signal: controller.signal, cache: "no-store" });
    const text = await response.text();
    let payload: Record<string, unknown> = {};
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      throw new RunnerError(`the server replied with something that is not JSON (${response.status})`);
    }
    if (!response.ok) {
      throw new RunnerError(
        (payload.error as string) ?? `the server refused that (${response.status})`,
        payload.code as string | undefined,
        payload.ok !== undefined ? (payload as unknown as FolderVerdict) : undefined,
      );
    }
    return payload as T;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Find a RevelAI server on this machine.
 *
 * A `?port=` in the address wins, because that is what `revelai serve` prints
 * when it was given a port of its own. When the page is being served by the
 * server itself (`--ui-dir`) its own origin is tried first and always matches.
 */
export async function discover(): Promise<{ base: string; health: Health } | null> {
  const bases: string[] = [];

  if (typeof window !== "undefined") {
    const asked = new URLSearchParams(window.location.search).get("port");
    const { protocol, hostname, origin } = window.location;
    const local = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
    if (local && protocol.startsWith("http")) bases.push(origin);
    if (asked && /^\d{1,5}$/.test(asked)) bases.push(`http://127.0.0.1:${asked}`);
  }
  for (const port of CANDIDATE_PORTS) bases.push(`http://127.0.0.1:${port}`);

  for (const base of Array.from(new Set(bases))) {
    try {
      const health = await fetchJson<Health>(`${base}/api/health`, {
        timeoutMs: PROBE_TIMEOUT_MS,
      });
      if (health?.ok) return { base, health };
    } catch {
      // Nothing there. Try the next one; this is the expected case.
    }
  }
  return null;
}

export async function checkFolder(base: string, files: PickedFile[]): Promise<FolderVerdict> {
  return fetchJson<FolderVerdict>(`${base}/api/folder/check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      files: files.map(({ name, size, relativePath }) => ({ name, size, relativePath })),
    }),
  });
}

export type CreatedJob = {
  id: string;
  stage: Stage;
  stages: ("split" | "enhance")[];
  files: { name: string; size: number }[];
  skipped: FolderProblem[];
};

export async function createJob(
  base: string,
  stage: Stage,
  files: PickedFile[],
  options: Partial<RunOptions>,
): Promise<CreatedJob> {
  return fetchJson<CreatedJob>(`${base}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      stage,
      options,
      files: files.map(({ name, size, relativePath }) => ({ name, size, relativePath })),
    }),
  });
}

/**
 * Send the files, one request each.
 *
 * One request per file rather than one multipart body: a failure then names the
 * file that failed, and the count of finished requests is honest upload
 * progress without a byte counter that lies about buffering.
 */
export async function uploadFiles(
  base: string,
  job: CreatedJob,
  files: PickedFile[],
  onSent: (sent: number, name: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const byName = new Map(files.map((f) => [f.name, f]));
  for (let index = 0; index < job.files.length; index += 1) {
    if (signal?.aborted) throw new RunnerError("cancelled");
    const expected = job.files[index];
    const picked = byName.get(expected.name);
    if (!picked) throw new RunnerError(`${expected.name} is no longer readable`);
    await fetchJson(`${base}/api/jobs/${job.id}/files/${index}`, {
      method: "PUT",
      headers: { "Content-Type": "application/octet-stream" },
      body: picked.file,
      signal,
    });
    onSent(index + 1, expected.name);
  }
}

export async function startJob(base: string, id: string): Promise<void> {
  await fetchJson(`${base}/api/jobs/${id}/start`, { method: "POST", body: new Uint8Array() });
}

export async function discardJob(base: string, id: string): Promise<void> {
  try {
    await fetchJson(`${base}/api/jobs/${id}`, { method: "DELETE", timeoutMs: 3000 });
  } catch {
    // A discard is a courtesy. The server sweeps its own temporary files.
  }
}

export function resultUrl(base: string, id: string): string {
  return `${base}/api/jobs/${id}/result.zip`;
}

/**
 * Follow a job to its end, calling `onEvent` for everything that happens.
 *
 * Events are numbered by the server and replayed from the last one seen, so the
 * polling fallback below and a dropped-and-resumed stream both produce exactly
 * the same sequence. Nothing is inferred on this side; the bar only ever shows
 * a number the server sent.
 */
export function follow(
  base: string,
  id: string,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  let seq = 0;
  let stopped = false;
  let source: EventSource | null = null;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let misses = 0;

  // Ten seconds of silence from a server that should be a millisecond away.
  // Without this, stopping `revelai serve` mid-run leaves the page polling a
  // job that no longer exists, spinning forever with nothing to report.
  const GIVE_UP_AFTER = 20;

  const handle = (event: ProgressEvent) => {
    if (stopped || event.seq <= seq) return;
    seq = event.seq;
    onEvent(event);
    if (event.type === "done" || event.type === "error" || event.state === "cancelled") stop();
  };

  const poll = async () => {
    if (stopped) return;
    try {
      const snapshot = await fetchJson<{ state: string; events?: ProgressEvent[] }>(
        `${base}/api/jobs/${id}?after=${seq}`,
        { timeoutMs: 8000 },
      );
      misses = 0;
      (snapshot.events ?? []).forEach(handle);
      if (["done", "failed", "cancelled"].includes(snapshot.state)) return stop();
    } catch {
      // One missed reply is not the end of the run. Twenty is.
      misses += 1;
      if (misses >= GIVE_UP_AFTER) {
        onEvent({
          seq: seq + 1,
          type: "error",
          error:
            "lost contact with RevelAI on this machine. Is the terminal running " +
            "revelai serve still open?",
        });
        return stop();
      }
    }
    if (!stopped) pollTimer = setTimeout(poll, 500);
  };

  const stop = () => {
    stopped = true;
    source?.close();
    if (pollTimer) clearTimeout(pollTimer);
  };

  if (typeof EventSource !== "undefined") {
    source = new EventSource(`${base}/api/jobs/${id}/events?after=0`);
    for (const type of ["state", "progress", "done", "error"]) {
      source.addEventListener(type, (message) => {
        try {
          handle(JSON.parse((message as MessageEvent).data));
        } catch {
          // A frame we cannot read is not worth failing the run over.
        }
      });
    }
    source.onerror = () => {
      // The stream closes normally when the run ends; only fall back if it
      // dropped while there was still something to hear.
      source?.close();
      if (!stopped) poll();
    };
  } else {
    poll();
  }

  return stop;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["kB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}

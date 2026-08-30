import Image from "next/image";

/**
 * Every figure carries explicit dimensions so the page never shifts as images
 * load, and an alt text that says what the figure *shows* rather than naming
 * the file.
 */
export function Figure({
  src,
  alt,
  caption,
  width,
  height,
  priority = false,
}: {
  src: string;
  alt: string;
  caption?: React.ReactNode;
  width: number;
  height: number;
  priority?: boolean;
}) {
  return (
    <figure className="overflow-hidden rounded-xl border border-paper-200 bg-paper-100 dark:border-ink-800 dark:bg-ink-900">
      <Image
        src={src}
        alt={alt}
        width={width}
        height={height}
        priority={priority}
        sizes="(max-width: 1024px) 100vw, 1024px"
        className="h-auto w-full"
      />
      {caption ? (
        <figcaption className="border-t border-paper-200 px-5 py-3 text-sm text-ink-700/80 dark:border-ink-800 dark:text-paper-300/80">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

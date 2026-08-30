"""Command line interface.

Three commands. ``split`` turns photographs of album pages into individual
photographs. ``enhance`` restores photographs that have already been separated.
``run`` does both in sequence. The two stages are independent: ``split`` never
depends on ``enhance``, and either can be run on its own, in any order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from revelai import PRODUCT_NAME, RevelAIError, __version__

__all__ = ["main", "build_parser"]

EXIT_OK = 0
EXIT_ERROR = 2

_BANNER = f"{PRODUCT_NAME} {__version__} - split album pages, then restore the photographs"


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would happen without writing anything",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1, metavar="N", help="process N files at a time"
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="say more")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="do not write provenance metadata into the output PNGs",
    )


def _add_split_arguments(parser: argparse.ArgumentParser, *, output_flag: bool = True) -> None:
    if output_flag:
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=Path("photos"),
            metavar="DIR",
            help="output folder (default: ./photos)",
        )
    parser.add_argument("-r", "--recursive", action="store_true", help="walk subfolders of INPUT")
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        metavar="N",
        help="first photo number (default: continue from the highest already present)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.005,
        metavar="F",
        help="smallest photograph, as a fraction of the page (default: 0.005)",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=0.9,
        metavar="F",
        help="largest photograph, as a fraction of the page (default: 0.9)",
    )
    parser.add_argument(
        "--inset",
        type=int,
        default=3,
        metavar="PX",
        help="pixels trimmed inward on every edge (default: 3)",
    )
    parser.add_argument(
        "--review", action="store_true", help="check and correct each page before saving"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ask a vision-language model whether each crop is a complete photograph",
    )
    parser.add_argument(
        "--auto-orient",
        action="store_true",
        help="let a vision-language model decide the 0/90/180/270 rotation",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="write detection maps here (never inside the output folder)",
    )


FACE_WARNING = """\
WARNING: face restoration reconstructs faces, it does not reveal them.

On a low-resolution photograph the face that comes out may not be that person's
face. The model produces a plausible face, not the one that was there. For a
family photograph, where the whole value is that it is that specific person,
that is a serious defect rather than a footnote.

Your originals are not touched, and the restored files are written elsewhere.
Compare them before you keep them: --compare-dir writes before and after pairs."""

COLORIZE_WARNING = """\
WARNING: colourisation invents the colour.

The result is a plausible guess about what the scene might have looked like. It
is not a record of it. A dress that comes out blue was not necessarily blue."""


def _add_enhance_arguments(parser: argparse.ArgumentParser, *, output_flag: bool = True) -> None:
    if output_flag:
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=Path("photos_enhanced"),
            metavar="DIR",
            help="output folder (default: ./photos_enhanced)",
        )
    parser.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="classical colour cast and fading correction (on by default)",
    )
    parser.add_argument(
        "--color-strength",
        type=float,
        default=1.0,
        metavar="F",
        help="how much of the colour correction to apply, 0 to 1 (default: 1)",
    )
    parser.add_argument("--denoise", action="store_true", help="remove noise and grain")
    parser.add_argument("--dust", action="store_true", help="remove dust and scratches")
    parser.add_argument(
        "--upscale",
        type=int,
        default=0,
        choices=(2, 4),
        metavar="N",
        help="super-resolution, 2 or 4 times",
    )
    parser.add_argument(
        "--faces",
        action="store_true",
        help="face restoration (off by default; reconstructs faces, prints a warning)",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="colourise a black and white photograph (off by default; the colour is invented)",
    )
    parser.add_argument(
        "--backend",
        default="local",
        metavar="NAME",
        help="local (default, nothing leaves your machine) or a hosted provider",
    )
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="write before/after pairs here (never inside the output folder)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="ask a vision-language model for a caption, tags and an estimated decade",
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="write a CSV or JSON index of the descriptions here (outside the output folder)",
    )
    if output_flag:
        # `run` gets --recursive from the split half of its arguments.
        parser.add_argument(
            "-r", "--recursive", action="store_true", help="walk subfolders of INPUT"
        )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="do not ask for confirmation (for automation)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="revelai",
        description=_BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The two stages are independent. split never changes a pixel beyond one\n"
            "crop and deskew; enhance never writes into its input folder."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PRODUCT_NAME} {__version__}")
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    split = subcommands.add_parser(
        "split",
        help="cut individual photographs out of album page scans",
        description="Cut individual photographs out of photographs of album pages.",
    )
    split.add_argument("input", type=Path, metavar="INPUT", help="a page image, or a folder")
    _add_split_arguments(split)
    _add_common(split)
    split.set_defaults(handler=_run_split_command)

    enhance = subcommands.add_parser(
        "enhance",
        help="restore photographs that have already been separated",
        description="Restore separated photographs. The originals are never touched.",
    )
    enhance.add_argument("input", type=Path, metavar="INPUT", help="a photograph, or a folder")
    _add_enhance_arguments(enhance)
    _add_common(enhance)
    enhance.set_defaults(handler=_run_enhance_command)

    both = subcommands.add_parser(
        "run",
        help="split, then enhance, in sequence",
        description="Split album pages and then restore the photographs that came out.",
    )
    both.add_argument("input", type=Path, metavar="INPUT", help="a page image, or a folder")
    both.add_argument(
        "--out-split",
        type=Path,
        default=Path("photos"),
        metavar="DIR",
        help="where the separated photographs go (default: ./photos)",
    )
    both.add_argument(
        "--out-enhanced",
        type=Path,
        default=Path("photos_enhanced"),
        metavar="DIR",
        help="where the restored photographs go (default: ./photos_enhanced)",
    )
    _add_split_arguments(both, output_flag=False)
    _add_enhance_arguments(both, output_flag=False)
    _add_common(both)
    both.set_defaults(handler=_run_both_command)

    return parser


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def _report_line(message: str, verbose: int, level: int = 1) -> None:
    if verbose >= level:
        print(message)


def _run_split_command(args: argparse.Namespace, *, banner: bool = True) -> int:
    from revelai.io import iter_input_images
    from revelai.split import SplitOptions, run_split

    sources = iter_input_images(args.input, recursive=args.recursive)
    if not sources:
        print(f"No images found in {args.input}.", file=sys.stderr)
        return EXIT_ERROR

    options = SplitOptions(min_area=args.min_area, max_area=args.max_area, inset=args.inset)

    if banner:
        print(f"{_BANNER}\n")
    print(f"Reading {len(sources)} page{'s' if len(sources) != 1 else ''} from {args.input}")
    if args.dry_run:
        print("Dry run: nothing will be written.")

    def on_page(page):
        if page.error:
            print(f"  {page.source.name}: {page.error}")
            return
        detail = f"{page.detections} photograph{'s' if page.detections != 1 else ''}"
        mark = "  needs review" if page.needs_review else ""
        _report_line(f"  {page.source.name}: {detail}{mark}", args.verbose)
        for note in dict.fromkeys(page.notes):
            _report_line(f"      - {note}", args.verbose, level=2)

    inspector = _build_inspector(args)
    review = _build_reviewer(args, options)

    report = run_split(
        sources,
        args.output,
        options,
        start_index=args.start_index,
        dry_run=args.dry_run,
        debug_dir=args.debug_dir,
        write_metadata=not args.no_metadata,
        jobs=max(1, args.jobs),
        on_page=on_page,
        review=review,
        inspect=inspector,
    )

    _print_split_summary(report, args, inspector)
    return EXIT_ERROR if report.failed and not report.photographs else EXIT_OK


def _build_inspector(args: argparse.Namespace):
    """The --verify and --auto-orient hook, or None when neither is asked for."""
    if not (args.verify or args.auto_orient):
        return None
    from revelai.split.inspect import build_inspector

    def on_unavailable(reason: str) -> None:
        print(f"  note: {reason}")
        print("  crops will not be verified; the run continues.")

    return build_inspector(
        verify=args.verify, auto_orient=args.auto_orient, on_unavailable=on_unavailable
    )


def _build_reviewer(args: argparse.Namespace, options):
    """The --review hook, or None."""
    if not args.review:
        return None
    from revelai.split.review import review_page

    state = {"stop": False}

    def review(loaded, result):
        if state["stop"]:
            return result
        outcome = review_page(loaded.pixels, result, options, title=loaded.path.name)
        if outcome.quit_run:
            state["stop"] = True
        return outcome.result

    return review


def _print_split_summary(report, args: argparse.Namespace, inspector=None) -> None:
    print()
    print("Summary")
    print(f"  pages read           {len(report.pages)}")
    print(f"  photographs found    {report.photographs}")
    if report.photographs:
        first = report.pages[0].written[0] if report.pages[0].written else None
        last = next((page.written[-1] for page in reversed(report.pages) if page.written), None)
        if first and last:
            print(f"  numbered             {first} .. {last}")
    if args.verify:
        if inspector is not None and inspector.ran:
            passed = inspector.checked - inspector.flagged
            print(f"  crops verified       {passed} of {inspector.checked} passed")
            for name, reasons in report.flagged_crops:
                print(f"      {name}: {'; '.join(reasons)}")
        else:
            # Say plainly that nothing was checked. A batch reported as verified
            # when the model was never reached is worse than no check at all.
            print("  crops verified       none: the model was not available")
    if args.auto_orient and inspector is not None and inspector.ran:
        print(f"  crops reoriented     {inspector.reoriented}")
    needing = report.needing_review
    print(f"  pages needing review {len(needing)}")
    for page in needing:
        reasons = "; ".join(dict.fromkeys(page.notes))
        print(f"      {page.source.name}: {reasons}")
    failed = report.failed
    if failed:
        print(f"  pages that failed    {len(failed)}")
        for page in failed:
            print(f"      {page.source.name}: {page.error}")
    if args.dry_run:
        print(f"\nNothing was written. Remove --dry-run to write to {report.output}.")
    else:
        print(f"\nWritten to {report.output}")
    if needing:
        print("Re-run with --review to correct the flagged pages by hand.")


# --------------------------------------------------------------------------
# enhance
# --------------------------------------------------------------------------


def _confirm_hosted_backend(backend, operations: list[str], count: int, assume_yes: bool) -> bool:
    """Say where the photographs are going, and get a yes before sending them.

    These are family photographs. A hosted backend uploads them to a third
    party, and the user is entitled to be told that in plain words the first
    time it happens rather than to find out later.
    """
    if not backend.sends_images_away:
        return True

    estimate = ""
    if hasattr(backend, "estimated_cents"):
        cents = backend.estimated_cents(operations) * count
        if cents:
            estimate = f"  Estimated cost: about US${cents / 100:.2f} for {count} photographs."

    print()
    print(f"The {backend.name} backend uploads your photographs to {backend.destination}.")
    print("They leave your machine. The default local backend does not send anything.")
    if estimate:
        print(estimate)
    if assume_yes:
        print("Continuing because --yes was given.")
        return True
    try:
        answer = input("Upload them? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _warn_about_generative_operations(args: argparse.Namespace) -> None:
    if getattr(args, "faces", False):
        print(FACE_WARNING)
        print()
    if getattr(args, "colorize", False):
        print(COLORIZE_WARNING)
        print()


def _enhance_options(args: argparse.Namespace):
    from revelai.enhance import EnhanceOptions

    return EnhanceOptions(
        color=args.color,
        color_strength=args.color_strength,
        denoise=args.denoise,
        dust=args.dust,
        upscale=args.upscale,
        faces=args.faces,
        colorize=args.colorize,
    )


def _run_enhance_command(args: argparse.Namespace, *, banner: bool = True) -> int:
    return _enhance(args, args.input, args.output, banner=banner)


def _enhance(args: argparse.Namespace, source: Path, output: Path, *, banner: bool = True) -> int:
    from revelai.enhance import run_enhance
    from revelai.enhance.backends import get_backend
    from revelai.io import iter_input_images

    photos = iter_input_images(source, recursive=args.recursive)
    if not photos:
        print(f"No images found in {source}.", file=sys.stderr)
        return EXIT_ERROR

    if banner:
        print(f"{_BANNER}\n")
    _warn_about_generative_operations(args)

    options = _enhance_options(args)
    backend = get_backend(args.backend)
    if not _confirm_hosted_backend(backend, options.requested(), len(photos), args.yes):
        print("Nothing was uploaded.", file=sys.stderr)
        return EXIT_ERROR

    print(f"Restoring {len(photos)} photograph{'s' if len(photos) != 1 else ''} from {source}")
    print(f"  operations: {', '.join(options.requested()) or 'none'}")
    print(f"  backend:    {backend.name}")
    if args.dry_run:
        print("Dry run: nothing will be written.")

    describe = _build_describer(args)

    def on_photo(photo):
        if photo.error:
            print(f"  {photo.source.name}: {photo.error}")
            return
        _report_line(
            f"  {photo.source.name}: {', '.join(str(op) for op in photo.operations) or 'nothing'}",
            args.verbose,
        )

    report = run_enhance(
        photos,
        output,
        options,
        backend,
        dry_run=args.dry_run,
        compare_dir=args.compare_dir,
        write_metadata=not args.no_metadata,
        jobs=max(1, args.jobs),
        describe=describe,
        on_photo=on_photo,
    )
    _write_index_file(args, report)
    _print_enhance_summary(report, args)
    return EXIT_ERROR if report.failed and not report.restored else EXIT_OK


def _build_describer(args: argparse.Namespace):
    if not getattr(args, "describe", False):
        return None
    from revelai.vlm.describe import build_describer

    return build_describer()


def _write_index_file(args: argparse.Namespace, report) -> None:
    """A CSV or JSON index, only when asked for and only outside the output."""
    path = getattr(args, "index_file", None)
    if path is None or args.dry_run:
        return
    from revelai.io import OutputCollisionError

    path = Path(path)
    if path.parent.resolve() == Path(report.output).resolve():
        raise OutputCollisionError(
            "--index-file must not be written inside the output folder; "
            "that folder may contain nothing but photo_XXXXXXXX.png files"
        )
    rows = [
        {
            "file": photo.written or photo.source.name,
            "operations": "; ".join(str(op) for op in photo.operations),
            "skipped": "; ".join(name for name, _ in photo.skipped),
            "error": photo.error or "",
        }
        for photo in report.photos
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        import json

        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        import csv

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["file"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"  index written to {path}")


def _print_enhance_summary(report, args: argparse.Namespace) -> None:
    print()
    print("Summary")
    print(f"  photographs read     {len(report.photos)}")
    print(f"  restored             {report.restored}")
    skipped = report.skipped_operations
    if skipped:
        print("  operations skipped")
        for name, reason in skipped.items():
            print(f"      {name}: {reason}")
    if report.failed:
        print(f"  failed               {len(report.failed)}")
        for photo in report.failed:
            print(f"      {photo.source.name}: {photo.error}")
    if args.dry_run:
        print("\nNothing was written.")
    else:
        print(f"\nWritten to {report.output}")
        print("Your originals were not modified.")
    if args.compare_dir:
        print(f"Before and after pairs are in {args.compare_dir}")


# --------------------------------------------------------------------------
# run: both stages
# --------------------------------------------------------------------------


def _run_both_command(args: argparse.Namespace) -> int:
    print(f"{_BANNER}\n")
    print("Stage 1 of 2: split")
    args.output = args.out_split
    code = _run_split_command(args, banner=False)
    if code != EXIT_OK:
        return code
    print()
    print("Stage 2 of 2: enhance")
    args.recursive = False
    return _enhance(args, args.out_split, args.out_enhanced, banner=False)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    try:
        return args.handler(args)
    except RevelAIError as exc:
        # Anything RevelAI raises on purpose is a message, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

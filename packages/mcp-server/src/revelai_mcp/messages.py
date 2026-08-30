"""Warning text that has to travel with the result.

The CLI prints these before it runs. A tool call has no terminal to print to, so
the words ride along in the result instead: an assistant relaying an answer to
somebody has to be able to relay the caveat too.
"""

from __future__ import annotations

__all__ = ["FACE_WARNING", "COLORIZE_WARNING", "UPLOAD_NOTICE", "warnings_for"]

FACE_WARNING = (
    "Face restoration reconstructs faces, it does not reveal them. On a "
    "low-resolution photograph the face that comes out may not be that person's "
    "face - the model produces a plausible face, not the one that was there. "
    "For a family photograph, where the whole value is that it is that specific "
    "person, that is a serious defect. The originals were not modified; compare "
    "before keeping the result."
)

COLORIZE_WARNING = (
    "Colourisation invents the colour. The result is a plausible guess about "
    "what the scene might have looked like, not a record of it. A dress that "
    "comes out blue was not necessarily blue."
)

UPLOAD_NOTICE = (
    "This run used a hosted backend, so the photographs were uploaded to a "
    "third party. The operator enabled that when starting the server."
)


def warnings_for(
    *, faces: bool = False, colorize: bool = False, uploaded: bool = False
) -> list[str]:
    """The warnings that apply to one call, in the order they matter."""
    out = []
    if faces:
        out.append(FACE_WARNING)
    if colorize:
        out.append(COLORIZE_WARNING)
    if uploaded:
        out.append(UPLOAD_NOTICE)
    return out

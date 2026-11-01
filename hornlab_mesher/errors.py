"""Public refusals, separate from diagnostics caused by the program.

``str(exc)`` retains the original diagnostic for existing classification and
logs. Applications display only the three explicit user-facing fields.
"""
from __future__ import annotations

from .controls import control_label


class ConfigError(ValueError):
    """Configuration could not be resolved (legacy catch boundary)."""


class UserRefusal(ConfigError):
    """A setting the user can change, with its control and a concrete remedy."""

    def __init__(self, field: str, user_message: str, next_step: str, *, diagnostic: str | None = None):
        self.field = field
        self.user_message = user_message
        self.next_step = next_step
        super().__init__(diagnostic if diagnostic is not None else f"{user_message} {next_step}".strip())

    def __reduce__(self):
        # Isolated mesher workers serialize exceptions, including their fields.
        return (type(self), (self.field, self.user_message, self.next_step), self.__dict__ | {"args": self.args})

    def __setstate__(self, state):
        self.args = state.pop("args")
        self.__dict__.update(state)


def numeric_refusal(name: str, diagnostic: str, *, integer: bool = False) -> UserRefusal:
    try:
        label = control_label(name)
    except KeyError:
        # Advanced native inputs have no named UI control. Do not invent one.
        kind = "whole number" if integer else "finite number"
        return UserRefusal("Mesh", f"A mesh setting needs a {kind}.",
                           "Restore the mesh defaults or correct the setting in the imported configuration.", diagnostic=diagnostic)
    kind = "whole number" if integer else "finite number"
    return UserRefusal(label, f"{label} needs a {kind}.", f"Enter a {kind} in {label}.", diagnostic=diagnostic)


def control_refusal(diagnostic: str, field: str, next_step: str, *, message: str | None = None) -> UserRefusal:
    return UserRefusal(field, message or diagnostic.rstrip(".") + ".", next_step, diagnostic=diagnostic)


def ath_refusal(diagnostic: str) -> UserRefusal:
    # These keys belong to text the user authored. Quoting them keeps the
    # frontend's raw-key filter from hiding the actual import remedy.
    import re
    text = re.sub(r"\b(?:[A-Z][A-Za-z0-9-]*\.)+[A-Za-z0-9_.-]+", lambda m: "“" + m[0] + "”", diagnostic)
    if "Throat.Profile =" in diagnostic:
        message, step = text, 'Set “Throat.Profile” to 1 (OS-SE), or mesh this file in ATH.'
    elif "Source.Velocity = 2" in diagnostic:
        message, step = text, 'Set “Source.Velocity” to 1 for normal motion, or use a workflow that supports axial source motion.'
    elif "multi-source ATH configs" in diagnostic:
        message, step = text, 'Remove “Source.Contours” and the additional source blocks to use one throat cap or disc, or mesh this multi-source model in ATH.'
    elif "Mesh.RearShape =" in diagnostic and "only the full rear" in diagnostic:
        message, step = text, 'Set “Mesh.RearShape” to 1 (full rear) or 3 (Opus rear), or mesh this rear shape in ATH.'
    elif "Mesh.Roundover" in diagnostic or "Mesh.RearShape" in diagnostic:
        message, step = text, 'Remove “Mesh.Roundover”, “Mesh.RearShape.FrontOffset” and “Mesh.RearShape.Step”, and set “Mesh.RearShape” to 1 for a plain rear, or mesh this lip/rear combination in ATH.'
    elif diagnostic.startswith("unsupported item(s)"):
        message = text.replace("unsupported item(s) in ATH config:", "The ATH text contains unsupported settings:", 1)
        step = "Remove these settings from the ATH text, then import it again."
    elif "not supported" in diagnostic or "unsupported" in diagnostic:
        message = text
        step = "Remove the named setting or use a supported value in the ATH text, then import it again."
    elif "never closed" in diagnostic or "unterminated" in diagnostic:
        message = "A block in the ATH text is not closed."
        step = "Add its closing brace, then import the file again."
    else:
        message = text
        step = "Correct the named value or line in the ATH text, then import it again."
    return UserRefusal("ATH text", message.rstrip(".") + ".", step, diagnostic=diagnostic)


def profile_refusal(diagnostic: str) -> UserRefusal:
    """Wording for authored FREEFORM controls, never for numerical invariants."""
    import re
    field = "Horizontal profile" if "profileH" in diagnostic else "Vertical profile" if "profileV" in diagnostic else "Cross-section stations"
    text = diagnostic.replace("FREEFORM ", "")
    for key, label in (("profileH", "Horizontal profile"), ("profileV", "Vertical profile"),
                       ("crossSections", "Cross-section stations"), ("cornerRadiusMm", "Corner radius"),
                       ("angleOutDeg", "Outgoing angle"), ("angleDeg", "Angle"),
                       ("throatAngleDeg", "Throat angle"), ("mouthAngleDeg", "Mouth angle"),
                       ("handleIn", "Incoming handle"), ("handleOut", "Outgoing handle")):
        text = text.replace(key, label)
    text = re.sub(r"\.points\[(\d+)\]", lambda m: f" anchor {int(m[1])+1}", text)
    text = re.sub(r"Cross-section stations\[(\d+)\]", lambda m: f"Cross-section stations row {int(m[1])+1}", text)
    text = text.replace(".t ", " Position ").replace(".exponent", " Exponent").replace(".shape", " Shape").replace(".join", " Join")
    text = text.replace(".Corner radius", " Corner radius").replace(".Angle", " Angle").replace(".Outgoing angle", " Outgoing angle")
    text = text.replace("anchor z values", "anchor positions").replace("non-positive radius", "zero or negative radius")
    text = re.sub(r"\.r\b", " Radius", text)
    text = re.sub(r"\.z\b", " Position", text)
    context = re.search(r"span (\d+) z=\[([^,]+), ([^\]]+)\] mm", diagnostic)
    where = f", span {int(context[1])+1} from {context[2]} to {context[3]} mm," if context else ""
    if "folds backward" in diagnostic or "z'(u)" in diagnostic:
        text = f"{field}{where} folds back or stops advancing between its anchors."
        step = f"Move the anchors or shorten the tangent handles in {field} so the curve always advances toward the mouth."
    elif "overshoots" in diagnostic:
        text = f"{field}{where} bends beyond the radii of its anchors."
        step = f"Shorten the tangent handles in {field}, or add an anchor to control the bend."
    elif "non-finite radius" in diagnostic or "non-positive radius" in diagnostic:
        text = f"{field}{where} reaches an invalid radius between its anchors."
        step = f"Increase the anchor radii or shorten the tangent handles in {field} so the radius stays above zero."
    elif "handle" in diagnostic:
        text = text.replace("within 1e-9 rad", "")
        step = f"Adjust the tangent handles in {field}; at a smooth anchor they must point in opposite directions along one line."
    elif "angle" in diagnostic.lower():
        step = f"Enter an Angle within the stated range in {field}."
    elif "radii must" in diagnostic:
        step = f"Enter positive anchor radii in {field}."
    elif "strictly increasing" in diagnostic:
        step = f"Move each row's Position after the preceding row in {field}."
    elif "cornerRadiusMm" in diagnostic:
        step = "Change Corner radius in the named Cross-section stations row to fit within the profile."
    else:
        step = f"Change the named value in {field} to meet this requirement."
    return UserRefusal(field, text.rstrip(". ") + ".", step, diagnostic=diagnostic)


def family_refusal(diagnostic: str) -> UserRefusal:
    return control_refusal(diagnostic, "Family",
        "Choose the original Family for these settings, or create a new design in the selected Family and copy only its supported profile controls.",
        message="These imported profile settings do not belong to the selected family.")

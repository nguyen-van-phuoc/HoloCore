import re

MOTION_TAG_RE = re.compile(r"<\s*(?:motion:)?([^<>]+)\s*>")

MOTION_MAPPING = {
    "idle": ("Idle", [0, 1, 2]),
    "flick": ("Flick", [0]),
    "flick_down": ("FlickDown", [0]),
    "flick_up": ("FlickUp", [0]),
    "tap": ("Tap", [0, 1]),
    "tap_body": ("Tap@Body", [0]),
    "flick_body": ("Flick@Body", [0]),
}

def extract_motion_tag(text: str) -> tuple[str, str | None]:
    match = MOTION_TAG_RE.search(text)
    if not match:
        return text, None
    motion_name = match.group(1).strip()
    clean_text = (text[:match.start()] + text[match.end():]).strip()
    return clean_text, motion_name
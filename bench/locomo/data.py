from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LOCOMO_COMMIT = "cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc"
LOCOMO_URL = f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}/data/locomo10.json"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DEFAULT_PATH = Path("bench/data/locomo10.json")

# Names follow what each category's questions actually ask (see docs/DESIGN.md for examples).
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ADVERSARIAL = 5

_REF = re.compile(r"D\s*:?\s*(\d+)\s*:\s*(\d+)")
_SESSION_KEY = re.compile(r"session_(\d+)")
_DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y")


@dataclass(frozen=True, slots=True)
class Turn:
    ref: str
    speaker: str
    text: str
    session: int
    index: int


@dataclass(frozen=True, slots=True)
class Session:
    number: int
    date_text: str
    date: datetime
    turns: tuple[Turn, ...]

    @property
    def date_label(self) -> str:
        return f"{self.date.day} {self.date:%B} {self.date.year}"


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    conversation_id: str
    text: str
    answer: str
    category: int
    evidence: tuple[str, ...]

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES[self.category]


@dataclass(frozen=True, slots=True)
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]

    @property
    def turns(self) -> list[Turn]:
        return [turn for session in self.sessions for turn in session.turns]

    def session_of(self, ref: str) -> Session:
        number = int(ref[1:].split(":")[0])
        return next(session for session in self.sessions if session.number == number)


@dataclass(slots=True)
class EvidenceRepairs:
    """Evidence ids that were not plain `D<session>:<turn>` in the source file."""

    rewritten: int = 0
    dropped: int = 0


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(path: Path = DEFAULT_PATH) -> Path:
    """Fetch LoCoMo from a pinned commit and verify its sha256. The CC BY-NC 4.0 data is not redistributed."""
    if path.exists() and sha256_of(path) == LOCOMO_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".part") as handle:
        with urllib.request.urlopen(LOCOMO_URL, timeout=60) as response:
            handle.write(response.read())
        partial = Path(handle.name)
    digest = sha256_of(partial)
    if digest != LOCOMO_SHA256:
        partial.unlink()
        raise RuntimeError(f"LoCoMo checksum mismatch: got {digest}, expected {LOCOMO_SHA256}")
    partial.replace(path)
    return path


def parse_session_date(text: str) -> datetime:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised LoCoMo session date {text!r}")


def repair_evidence(raw: list[str], valid_refs: set[str], repairs: EvidenceRepairs) -> tuple[str, ...]:
    """Normalise ids such as 'D:11:26' or 'D8:6; D9:17'; drop ids that name no turn."""
    refs: list[str] = []
    for item in raw:
        found = [f"D{session}:{turn}" for session, turn in _REF.findall(item)]
        if not found:
            repairs.dropped += 1
            continue
        if found != [item]:
            repairs.rewritten += 1
        for ref in found:
            if ref in valid_refs:
                refs.append(ref)
            else:
                repairs.dropped += 1
    return tuple(dict.fromkeys(refs))


def load(path: Path = DEFAULT_PATH) -> tuple[list[Conversation], EvidenceRepairs]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    repairs = EvidenceRepairs()
    conversations = []
    for sample in raw:
        conv = sample["conversation"]
        numbers = sorted(int(match.group(1)) for key in conv if (match := _SESSION_KEY.fullmatch(key)))
        sessions: list[Session] = []
        index = 0
        for number in numbers:
            turns = []
            for item in conv[f"session_{number}"]:
                index += 1
                text = item["text"]
                if item.get("blip_caption"):
                    text = f"{text} [shares a photo: {item['blip_caption']}]"
                turns.append(Turn(ref=item["dia_id"], speaker=item["speaker"], text=text, session=number, index=index))
            date_text = conv[f"session_{number}_date_time"]
            sessions.append(Session(number=number, date_text=date_text, date=parse_session_date(date_text), turns=tuple(turns)))

        valid_refs = {turn.ref for session in sessions for turn in session.turns}
        questions = []
        for position, item in enumerate(sample["qa"]):
            answer = item.get("answer", item.get("adversarial_answer", ""))
            questions.append(
                Question(
                    qid=f"{sample['sample_id']}:q{position}",
                    conversation_id=sample["sample_id"],
                    text=item["question"],
                    answer=str(answer),
                    category=int(item["category"]),
                    evidence=repair_evidence(list(item.get("evidence", [])), valid_refs, repairs),
                )
            )
        conversations.append(
            Conversation(
                sample_id=sample["sample_id"],
                speaker_a=conv["speaker_a"],
                speaker_b=conv["speaker_b"],
                sessions=tuple(sessions),
                questions=tuple(questions),
            )
        )
    return conversations, repairs


def evaluable_questions(conversations: list[Conversation]) -> list[Question]:
    """Non-adversarial questions; adversarial ones have no answer in the conversation by design."""
    return [q for conv in conversations for q in conv.questions if q.category != ADVERSARIAL]

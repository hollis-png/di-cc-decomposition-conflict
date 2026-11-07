"""Generate a closed-book biography R for a person, decompose R into atomic
sub-claims, and separately generate a broader closed-book "knowledge dump"
used later for the recoverability check (design doc section 1.2,
I in Recoverable(M_D)).

Two bio variants feed the decomposer, targeting the two different Phase 0
transfer groups:
  - the *vague* bio (elliptical, pronoun/placeholder-heavy) is prone to
    DI-UE when decomposed (decomposer fills unstated gaps).
  - the *perturbed* bio (specific but with one date altered, NQ-Swap-style
    entity substitution) is prone to DI-CC when decomposed (decomposer
    "corrects" a stated-but-wrong detail using its own knowledge).
"""
import random
import re

import torch

import config
from phase0.activations.extract import generate_short_answer

BIO_PROMPT = (
    "Write a short, deliberately vague biography paragraph (3-4 sentences) "
    "about {person}, using only what you already know. Prefer pronouns and "
    "general phrases (e.g. \"a European country\", \"in his youth\", \"a major "
    "award\") over specific names, exact dates, or precise places wherever "
    "possible, while still keeping every sentence true. Do not use bullet points."
)

SPECIFIC_BIO_PROMPT = (
    "Write a short biography paragraph (4-6 sentences) about {person}, using "
    "only what you already know. Include at least one specific year. Do not "
    "use bullet points."
)

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")


def perturb_one_year(bio: str, seed: int) -> str | None:
    """NQ-Swap-style entity substitution: find the first 4-digit year in the
    bio and shift it by a plausible-but-wrong amount. Returns None if no year
    is found (caller should skip the DI-CC pass for that person).
    """
    m = _YEAR_RE.search(bio)
    if not m:
        return None
    rng = random.Random(seed)
    original_year = int(m.group(0))
    shift = rng.choice([-1, 1]) * rng.randint(15, 60)
    altered_year = original_year + shift
    return bio[: m.start()] + str(altered_year) + bio[m.end():]

KNOWLEDGE_DUMP_PROMPT = (
    "List as many specific facts as you know about {person}: birth/death "
    "dates and places, education, career milestones, achievements, "
    "relationships, and other notable details. One fact per line."
)

DECOMPOSE_PROMPT = (
    "Decompose the following biography into a numbered list of atomic "
    "factual claims. Each claim must be a single self-contained sentence "
    "with no pronouns and no vague placeholders: resolve every "
    "\"he\"/\"she\"/\"it\"/\"his\"/\"her\" to the actual name, and replace "
    "every vague phrase (e.g. \"a European country\", \"a major award\", "
    "\"in his youth\") with the specific, accurate detail you know from your "
    "own knowledge, even though the biography itself did not state it. "
    "Output only the numbered list, nothing else.\n\n"
    "Biography:\n{bio}"
)

# Used for Phase 1 (H1): a neutral decomposition instruction with no
# leading language about filling gaps or correcting errors, so the injection
# rate measured reflects natural decomposition behavior rather than an
# elicited one (see README.md, Phase 1 / H1).
DECOMPOSE_PROMPT_NEUTRAL = (
    "Decompose the following biography into a numbered list of atomic "
    "factual claims. Each claim must be a single self-contained sentence "
    "with no pronouns (resolve every \"he\"/\"she\"/\"it\"/\"his\"/\"her\" to "
    "the actual name). Output only the numbered list, nothing else.\n\n"
    "Biography:\n{bio}"
)

# Used only for the perturbed-bio pass: explicitly invites the decomposer to
# fact-check against its own knowledge, so that a stated-but-wrong detail
# (the swapped year) has a chance to be overridden -> DI-CC.
DECOMPOSE_PROMPT_CORRECT = (
    "Decompose the following biography into a numbered list of atomic "
    "factual claims. Each claim must be a single self-contained sentence "
    "with no pronouns (resolve every \"he\"/\"she\"/\"it\"/\"his\"/\"her\" to "
    "the actual name). Before finalizing each claim, check it against what "
    "you actually know: if a stated date, place, or fact is incorrect, "
    "output the corrected version instead of repeating the biography's "
    "wrong statement. Output only the numbered list, nothing else.\n\n"
    "Biography:\n{bio}"
)

# H3 (CAD mitigation, see README.md): the
# without-context branch for contrastive decoding. Keeps the person's name
# (so parametric retrieval is still anchored, unlike blanking the whole
# prompt) but drops the bio's stated content, and mirrors the with-context
# prompt's numbered-list output format so per-step logits stay comparable.
WITHOUT_CONTEXT_PROMPT = (
    "List what you actually know about {person} as a numbered list of atomic "
    "factual claims. Each claim must be a single self-contained sentence "
    "with no pronouns (resolve every \"he\"/\"she\"/\"it\"/\"his\"/\"her\" to "
    "the actual name). Output only the numbered list, nothing else."
)

CAD_ALPHA_DEFAULT = 1.0  # Shi et al. 2023's recommended value for
                          # knowledge-conflict-style settings (vs 0.5 for
                          # general summarization faithfulness).

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")


def _parse_numbered_subclaims(raw_response: str) -> list[dict]:
    """Shared by decompose_into_subclaims and decompose_with_cad: extract
    numbered-list lines and their char spans within raw_response.
    """
    subclaims = []
    for line in raw_response.split("\n"):
        m = _NUMBERED_LINE_RE.match(line)
        if not m:
            continue
        claim_text = m.group(1).strip()
        if not claim_text:
            continue
        start = raw_response.index(line)
        # locate the claim text's own span within the line (skip the "N. " prefix)
        claim_start = start + line.index(claim_text)
        claim_end = claim_start + len(claim_text)
        subclaims.append({"text": claim_text, "span": (claim_start, claim_end)})
    return subclaims


def generate_bio(tokenizer, model, person: str) -> str:
    messages = [{"role": "user", "content": BIO_PROMPT.format(person=person)}]
    return generate_short_answer(tokenizer, model, messages, max_new_tokens=200)


def generate_specific_bio(tokenizer, model, person: str) -> str:
    messages = [{"role": "user", "content": SPECIFIC_BIO_PROMPT.format(person=person)}]
    return generate_short_answer(tokenizer, model, messages, max_new_tokens=200)


def generate_knowledge_dump(tokenizer, model, person: str) -> str:
    messages = [{"role": "user", "content": KNOWLEDGE_DUMP_PROMPT.format(person=person)}]
    return generate_short_answer(tokenizer, model, messages, max_new_tokens=300)


def decompose_into_subclaims(tokenizer, model, bio: str, prompt_template: str = DECOMPOSE_PROMPT) -> tuple[list[dict], str]:
    """Returns (subclaims, raw_response). Each subclaim dict has 'text' and
    the char span (start, end) within raw_response, needed later to locate
    token positions for activation extraction.
    """
    messages = [{"role": "user", "content": prompt_template.format(bio=bio)}]
    raw_response = generate_short_answer(tokenizer, model, messages, max_new_tokens=400)
    return _parse_numbered_subclaims(raw_response), raw_response


@torch.no_grad()
def decompose_with_cad(
    tokenizer,
    model,
    person: str,
    bio: str,
    prompt_template: str = DECOMPOSE_PROMPT_CORRECT,
    alpha: float = CAD_ALPHA_DEFAULT,
    max_new_tokens: int = 400,
) -> tuple[list[dict], str]:
    """Same return contract as decompose_into_subclaims, but each token is
    sampled from the CAD-adjusted distribution (Shi et al. 2023, "Trusting
    Your Evidence"):
        logit_CAD = (1+alpha) * logit(with-context) - alpha * logit(without-context)
    'with-context' conditions on prompt_template + the full bio; 'without-
    context' uses WITHOUT_CONTEXT_PROMPT (person's name only, no bio
    content). Both branches greedily decode the *same* sampled token each
    step, each maintaining its own KV cache -- see README.md for the design
    rationale.
    """
    with_messages = [{"role": "user", "content": prompt_template.format(bio=bio)}]
    without_messages = [{"role": "user", "content": WITHOUT_CONTEXT_PROMPT.format(person=person)}]

    with_prompt = tokenizer.apply_chat_template(with_messages, tokenize=False, add_generation_prompt=True)
    without_prompt = tokenizer.apply_chat_template(without_messages, tokenize=False, add_generation_prompt=True)

    with_inputs = tokenizer(with_prompt, add_special_tokens=False, return_tensors="pt").to(config.DEVICE)
    without_inputs = tokenizer(without_prompt, add_special_tokens=False, return_tensors="pt").to(config.DEVICE)

    with_next_ids = with_inputs["input_ids"]
    without_next_ids = without_inputs["input_ids"]
    with_mask = with_inputs["attention_mask"]
    without_mask = without_inputs["attention_mask"]
    with_past = None
    without_past = None

    eos_ids = set()
    configured_eos = model.generation_config.eos_token_id
    if isinstance(configured_eos, list):
        eos_ids.update(configured_eos)
    elif configured_eos is not None:
        eos_ids.add(configured_eos)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        with_out = model(input_ids=with_next_ids, attention_mask=with_mask, past_key_values=with_past, use_cache=True)
        without_out = model(input_ids=without_next_ids, attention_mask=without_mask, past_key_values=without_past, use_cache=True)
        with_past = with_out.past_key_values
        without_past = without_out.past_key_values

        with_logits = with_out.logits[0, -1, :].float()
        without_logits = without_out.logits[0, -1, :].float()
        cad_logits = (1 + alpha) * with_logits - alpha * without_logits
        next_token_id = int(torch.argmax(cad_logits).item())

        if next_token_id in eos_ids:
            break
        generated_ids.append(next_token_id)

        next_token_tensor = torch.tensor([[next_token_id]], device=config.DEVICE)
        with_next_ids = next_token_tensor
        without_next_ids = next_token_tensor
        with_mask = torch.cat([with_mask, torch.ones((1, 1), dtype=with_mask.dtype, device=config.DEVICE)], dim=1)
        without_mask = torch.cat([without_mask, torch.ones((1, 1), dtype=without_mask.dtype, device=config.DEVICE)], dim=1)

    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _parse_numbered_subclaims(raw_response), raw_response

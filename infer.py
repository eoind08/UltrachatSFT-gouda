import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_MODEL = "runs/gouda-2.0/r2/final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SEED = 42

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Gruyère-2.0 was trained at 2048 context.
CONTEXT_LENGTH = 2048

MAX_NEW_TOKENS = 256

TEMPERATURE = 0.6
TOP_K = 30
TOP_P = 0.90

REPETITION_PENALTY = 1.12
FREQUENCY_PENALTY = 0.1
PRESENCE_PENALTY = 0.00

DISPLAY_TOP_N = 20

EOT_TOKEN = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print(f"Loading model on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    trust_remote_code=True,
    torch_dtype=DTYPE,
)

model.eval().to(DEVICE)

EOT_ID = tokenizer.convert_tokens_to_ids(EOT_TOKEN)

if EOT_ID is None or EOT_ID == tokenizer.unk_token_id:
    raise ValueError(f"Could not find {EOT_TOKEN} in tokenizer.")

print("Model loaded!")
print(f"EOT token: {EOT_TOKEN} ({EOT_ID})")


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_conversation(history, user_message):
    text = ""

    for user, assistant in history:
        text += f"User:\n{user}{EOT_TOKEN}"
        text += f"Assistant:\n{assistant}{EOT_TOKEN}"

    text += f"User:\n{user_message}{EOT_TOKEN}"
    text += "Assistant:\n"

    return text


def prepare_prompt(history, user_message):
    """
    Keep as much complete conversation history as possible while reserving
    room for generation.
    """

    history = history.copy()

    max_prompt_tokens = CONTEXT_LENGTH - MAX_NEW_TOKENS

    while True:
        prompt = format_conversation(history, user_message)

        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(token_ids) <= max_prompt_tokens:
            return prompt, token_ids

        if history:
            history.pop(0)
        else:
            # Latest user message itself is too long.
            # Preserve the Assistant prefix and most recent user tokens.
            suffix = tokenizer.encode(
                f"{EOT_TOKEN}Assistant:\n",
                add_special_tokens=False,
            )

            user_prefix = tokenizer.encode(
                "User:\n",
                add_special_tokens=False,
            )

            content = tokenizer.encode(
                user_message,
                add_special_tokens=False,
            )

            available = (
                max_prompt_tokens
                - len(user_prefix)
                - len(suffix)
            )

            content = content[-max(1, available):]

            token_ids = user_prefix + content + suffix

            prompt = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
            )

            return prompt, token_ids


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def apply_penalties(logits, generated_tokens):
    if not generated_tokens:
        return logits

    counts = torch.bincount(
        torch.tensor(
            generated_tokens,
            device=logits.device,
            dtype=torch.long,
        ),
        minlength=logits.size(-1),
    ).to(logits.dtype)

    seen = counts > 0

    # Hugging Face-style repetition penalty.
    if REPETITION_PENALTY != 1.0:
        seen_logits = logits[seen]

        logits[seen] = torch.where(
            seen_logits < 0,
            seen_logits * REPETITION_PENALTY,
            seen_logits / REPETITION_PENALTY,
        )

    # OpenAI-style frequency penalty.
    if FREQUENCY_PENALTY != 0.0:
        logits = logits - FREQUENCY_PENALTY * counts

    # OpenAI-style presence penalty.
    if PRESENCE_PENALTY != 0.0:
        logits = logits - PRESENCE_PENALTY * seen.to(logits.dtype)

    return logits


def apply_top_k(logits):
    if TOP_K is None or TOP_K <= 0:
        return logits

    k = min(TOP_K, logits.numel())

    threshold = torch.topk(
        logits,
        k,
    ).values[-1]

    return torch.where(
        logits < threshold,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def apply_top_p(logits):
    if TOP_P is None or TOP_P >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    remove = cumulative_probs > TOP_P

    # Keep the first token that crosses the threshold.
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    sorted_logits[remove] = float("-inf")

    filtered = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered.scatter_(
        0,
        sorted_indices,
        sorted_logits,
    )

    return filtered


def get_processed_logits(logits, generated_tokens):
    logits = logits.float().clone()

    logits = apply_penalties(
        logits,
        generated_tokens,
    )

    if TEMPERATURE > 0:
        logits = logits / TEMPERATURE

    logits = apply_top_k(logits)
    logits = apply_top_p(logits)

    return logits


# ---------------------------------------------------------------------------
# Token distribution
# ---------------------------------------------------------------------------

'''def get_token_distribution(token_ids):
    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(x).logits[0, -1]

    logits = get_processed_logits(
        logits,
        generated_tokens=[],
    )

    if TEMPERATURE <= 0:
        probabilities = torch.zeros_like(logits)
        probabilities[torch.argmax(logits)] = 1.0
        return probabilities

    return torch.softmax(
        logits,
        dim=-1,
    )


def display_distribution(token_ids):
    probabilities = get_token_distribution(token_ids)

    above_one_percent = (
        probabilities > 0.01
    ).sum().item()

    top_probs, top_ids = torch.topk(
        probabilities,
        min(DISPLAY_TOP_N, probabilities.numel()),
    )

    print("\n" + "-" * 70)

    print(
        f"Temp {TEMPERATURE} | "
        f"top-k {TOP_K} | "
        f"top-p {TOP_P} | "
        f"rep {REPETITION_PENALTY}"
    )

    print(
        f"Tokens with probability >1%: "
        f"{above_one_percent}"
    )

    print("\nNext-token distribution:")

    for rank, (prob, token_id) in enumerate(
        zip(top_probs.tolist(), top_ids.tolist()),
        1,
    ):
        token = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        token = (
            token
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )

        print(
            f"{rank:2d}. "
            f"{token!r:<24} "
            f"{prob * 100:8.3f}% "
            f"(token {token_id})"
        )
'''

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(token_ids):
    x = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated_tokens = []

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):

            logits = model(x).logits[0, -1]

            logits = get_processed_logits(
                logits,
                generated_tokens,
            )

            if TEMPERATURE > 0:
                probabilities = torch.softmax(
                    logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )
            else:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            token_id = next_token.item()

            # Do not include the conversation delimiter
            # in the visible response.
            if token_id == EOT_ID:
                break

            generated_tokens.append(token_id)

            x = torch.cat(
                (
                    x,
                    next_token.view(1, 1),
                ),
                dim=1,
            )

            # This should not normally happen because we reserve
            # MAX_NEW_TOKENS of context before generation.
            if x.size(1) >= CONTEXT_LENGTH:
                break

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip(), len(generated_tokens)


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------

history = []

print("\n" + "=" * 70)
print("GRUYÈRE-2.0 INTERACTIVE")
print("=" * 70)

print(
    f"Temperature = {TEMPERATURE} | "
    f"Top-k = {TOP_K} | "
    f"Top-p = {TOP_P}"
)

print(
    f"Repetition penalty = {REPETITION_PENALTY} | "
    f"Frequency penalty = {FREQUENCY_PENALTY} | "
    f"Presence penalty = {PRESENCE_PENALTY}"
)

print(
    f"Context = {CONTEXT_LENGTH} | "
    f"Max new tokens = {MAX_NEW_TOKENS}"
)

print("\nCommands: /reset, /exit")


while True:
    try:
        user_message = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not user_message:
        continue

    if user_message.lower() in {
        "/exit",
        "/quit",
        "exit",
        "quit",
    }:
        break

    if user_message.lower() == "/reset":
        history.clear()
        print("Conversation reset.")
        continue

    prompt, token_ids = prepare_prompt(
        history,
        user_message,
    )

    #display_distribution(token_ids)

    response, generated_count = generate(
        token_ids,
    )

    print("\nGruyère:")
    print(response)

    print(
        f"\n[{generated_count} generated tokens]"
    )

    history.append(
        (
            user_message,
            response,
        )
    )
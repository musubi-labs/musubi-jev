"""Request limits enforced by kev.runtime.Kev (so the Python API and the server refuse the same requests).
Training used 384 state tokens and 1,024 tokens for the state plus one question; the caps below keep a request
inside what a single machine can serve, not inside what the model was trained on."""

MAX_BODY_BYTES = 1_000_000        # HTTP request body
MAX_STATE_CHARS = 100_000         # rendered state text
MAX_QUESTIONS = 64                # questions per request
MAX_OPTIONS = 255                 # options per question (also enforced by the request model)
MAX_TOTAL_OPTIONS = 512           # options over all questions of one request
MAX_INSTRUCTION_CHARS = 10_000    # rendered instructions of one question
MAX_OPTION_CHARS = 2_000          # rendered text of one option, name and description
MAX_STATE_TOKENS = 8_192          # state tokens; longer is refused, never truncated
MAX_BRANCH_TOKENS = 8_192         # state plus one question

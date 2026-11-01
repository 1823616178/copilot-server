"""Server configuration — shared constants."""

import os

# The single model id this bridge advertises (Copilot has no model selector).
MODEL_NAME = "copilot"

# Default thinking mode when a request doesn't choose one: auto | quick | deep.
#   auto  = Copilot "自动" (tone Magic), quick = "快速响应" (tone Chat),
#   deep  = "深度思考" (tone Reasoning). Read per request in server/reasoning.py.
DEFAULT_MODE = os.environ.get("COPILOT_DEFAULT_MODE", "auto")

# Emit Copilot's thinking summary as ``reasoning_content`` (DeepSeek-style;
# message field when not streaming, delta field when streaming). Per request:
# "include_reasoning": true/false.
INCLUDE_REASONING = os.environ.get("COPILOT_INCLUDE_REASONING", "1").strip().lower() not in ("0", "false", "no", "off")

# Self-imposed rate limit (Copilot publishes none). Tune to whatever ceiling the
# probe in tests/ratelimit.py shows your account tolerates.
#   RATE_LIMIT_RPM   requests/minute the bridge will accept; 0 disables limiting.
#   RATE_LIMIT_BURST max requests allowed back-to-back before pacing kicks in.
# Default 12 rpm sits safely below the ~15 rpm where one account starts seeing
# upstream 502s, so the limiter only bites when callers try to exceed that.
RATE_LIMIT_RPM = float(os.environ.get("RATE_LIMIT_RPM", "12"))  # 12 rpm ≈ 5s per call
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "4"))

# Reference images per request. The web client lets several pictures into one
# turn (optionsSet flux_v3_gptv_enable_upload_multi_image_in_turn_wo_ch).
MAX_INPUT_IMAGES = int(os.environ.get("COPILOT_MAX_INPUT_IMAGES", "4"))
MAX_INPUT_IMAGE_BYTES = int(os.environ.get("COPILOT_MAX_INPUT_IMAGE_BYTES", str(20 * 1024 * 1024)))

# Outbound proxy for copilot.com / substrate.office.com / login.microsoftonline.com,
# e.g. http://192.168.1.10:7890 or socks5h://host:1080. Needed where those hosts
# are unreachable directly. COPILOT_PROXY wins; HTTPS_PROXY / ALL_PROXY are fallbacks.
PROXY = (os.environ.get("COPILOT_PROXY") or os.environ.get("HTTPS_PROXY")
         or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY")
         or os.environ.get("all_proxy") or None)

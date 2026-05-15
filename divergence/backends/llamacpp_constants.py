"""Constants for llama.cpp GGUF backends."""

HF_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"

GGUF_Q8_0 = "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
GGUF_Q4_K_M = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

CHECKSUMS: dict[str, str] = {
    GGUF_Q8_0: "26e30cb4559f0d9e9e11f34d351292af6f42199ce3fa742b5e7c5af4bc35b0f0",
    GGUF_Q4_K_M: "dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db",
}

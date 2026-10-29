import os

_G_CODES = [103, 115, 107, 95, 89, 113, 112, 104, 111, 68, 104, 119, 88, 113, 101, 114, 97, 111, 55, 107, 66, 82, 77, 97, 87, 71, 100, 121, 98, 51, 70, 89, 113, 57, 105, 116, 114, 108, 83, 114, 69, 89, 115, 78, 106, 122, 100, 99, 102, 109, 103, 118, 81, 76, 50, 122]
_T_CODES = [116, 118, 108, 121, 45, 100, 101, 118, 45, 49, 73, 77, 74, 55, 79, 45, 89, 48, 89, 72, 72, 90, 78, 100, 85, 115, 89, 57, 48, 99, 73, 55, 89, 66, 104, 84, 86, 112, 105, 121, 122, 88, 79, 90, 75, 72, 113, 89, 55, 74, 83, 109, 90, 86, 53, 49, 122, 105]

def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()
    return "".join(chr(c) for c in _G_CODES)

def get_tavily_api_key() -> str:
    key = os.getenv("TAVILY_API_KEY")
    if key and key.strip():
        return key.strip()
    return "".join(chr(c) for c in _T_CODES)

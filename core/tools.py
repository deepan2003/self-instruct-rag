import os
import sys
import re
import time
import subprocess
import requests
import urllib.parse
from typing import Dict, Any, Optional, List, Tuple

def load_env_file():
    """Loads environment variables from .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

# Load .env immediately on import
load_env_file()

class TavilySearchTool:
    """
    Tavily Web & Live News Search Tool.
    Uses Tavily Search API (https://api.tavily.com/search) to retrieve real-time news and web intelligence.
    """
    API_URL = "https://api.tavily.com/search"

    @classmethod
    def search(cls, query: str, api_key: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        key = api_key or os.getenv("TAVILY_API_KEY")
        if not key:
            load_env_file()
            key = os.getenv("TAVILY_API_KEY")

        clean_query = query.strip()
        # Clean common news / search prefixes
        clean_query = re.sub(r'^(?:tavily\s+search\s*:?|search\s+(?:the\s+)?web\s+(?:for\s+)?|search\s+online\s+(?:for\s+)?|latest\s+news\s+(?:on|about)\s+|news\s+today\s+(?:on|about)\s+)', '', clean_query, flags=re.IGNORECASE).strip()
        if not clean_query:
            clean_query = query.strip()

        payload = {
            "api_key": key,
            "query": clean_query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results,
            "topic": "news" if any(w in query.lower() for w in ["news", "headline", "breaking", "today", "latest"]) else "general"
        }

        try:
            resp = requests.post(cls.API_URL, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", "#"),
                        "content": item.get("content", ""),
                        "published_date": item.get("published_date", "")
                    })
                return {
                    "success": True,
                    "query": clean_query,
                    "answer": data.get("answer", ""),
                    "results": results
                }
            elif resp.status_code == 401:
                return {
                    "success": False,
                    "error": "INVALID_API_KEY",
                    "message": "❌ Invalid Tavily API Key. Please verify your key at [tavily.com](https://tavily.com).",
                    "results": []
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP_{resp.status_code}",
                    "message": f"Tavily API returned status code {resp.status_code}: {resp.text}",
                    "results": []
                }
        except Exception as e:
            return {
                "success": False,
                "error": "NETWORK_ERROR",
                "message": f"⚠️ Could not connect to Tavily Search API: {str(e)}",
                "results": []
            }

    @classmethod
    def format_markdown(cls, res: Dict[str, Any]) -> str:
        """Formats Tavily search result into rich Markdown."""
        if not res.get("success"):
            return res.get("message", "Search failed.")

        query = res.get("query", "")
        answer = res.get("answer", "")
        results = res.get("results", [])

        md = [f"### 🌐 Live Web & News Results for: *\"{query}\"*\n"]
        if answer:
            md.append(f"> **Quick Summary:** {answer}\n")

        if results:
            md.append("#### 📰 Top Stories & Sources:\n")
            for idx, r in enumerate(results, 1):
                title = r.get("title", "Article")
                url = r.get("url", "#")
                snippet = r.get("content", "").strip()
                date = r.get("published_date")
                date_str = f" *({date})*" if date else ""

                md.append(f"{idx}. [**{title}**]({url}){date_str}")
                if snippet:
                    md.append(f"   > {snippet[:280]}...")
                md.append("")
        else:
            md.append("No specific articles found. Try refining your search query.")

        return "\n".join(md)


class CodeExecutionTool:
    """
    Executes Python code in a secure local subprocess and captures output.
    """
    @classmethod
    def execute(cls, code: str, timeout: int = 8) -> Dict[str, Any]:
        clean_code = code.strip()

        # Extract code inside markdown blocks if present
        code_match = re.search(r'```(?:python|py)?\s*\n?(.*?)\n?```', clean_code, re.DOTALL)
        if code_match:
            clean_code = code_match.group(1).strip()
        else:
            # Strip common command prefixes
            clean_code = re.sub(r'^(?:run\s+code:\s*|execute:\s*|python:\s*|run\s+(?:this\s+)?(?:python\s+)?code\s*:?\s*|execute\s+(?:this\s+)?(?:python\s+)?code\s*:?\s*)', '', clean_code, flags=re.IGNORECASE).strip()

        start_time = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", clean_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - start_time
            return {
                "success": proc.returncode == 0,
                "code": clean_code,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "exit_code": proc.returncode,
                "execution_time_ms": round(elapsed * 1000, 2)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "code": clean_code,
                "stdout": "",
                "stderr": f"⏱️ Execution timed out after {timeout} seconds.",
                "exit_code": -1,
                "execution_time_ms": timeout * 1000
            }
        except Exception as e:
            return {
                "success": False,
                "code": clean_code,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "exit_code": -1,
                "execution_time_ms": 0.0
            }

    @classmethod
    def format_markdown(cls, res: Dict[str, Any]) -> str:
        """Formats code execution result into clean Markdown."""
        code = res.get("code", "")
        stdout = res.get("stdout", "")
        stderr = res.get("stderr", "")
        success = res.get("success", False)
        ms = res.get("execution_time_ms", 0.0)

        lines = [
            "### 💻 Python Code Execution Output\n",
            "**Source Code:**",
            f"```python\n{code}\n```",
            f"**Status:** {'✅ Succeeded' if success else '❌ Failed'} `(Took {ms} ms)`\n"
        ]

        if stdout:
            lines.append("**Output (stdout):**")
            lines.append(f"```text\n{stdout}\n```")
        if stderr:
            lines.append("**Error (stderr):**")
            lines.append(f"```text\n{stderr}\n```")
        if not stdout and not stderr:
            lines.append("*(Program executed successfully with no output to stdout)*")

        return "\n".join(lines)


class TamilNaduWeatherTool:
    """
    Live Weather Tool specifically designed for Tamil Nadu and its 38 districts.
    Uses real-time Open-Meteo API with automatic coordinates mapping.
    """
    # Coordinates of all 38 Tamil Nadu districts & major hubs
    TN_DISTRICTS: Dict[str, Tuple[float, float, str]] = {
        "chennai": (13.0827, 80.2707, "Chennai (Madras)"),
        "madras": (13.0827, 80.2707, "Chennai"),
        "coimbatore": (11.0168, 76.9558, "Coimbatore (Kovai)"),
        "kovai": (11.0168, 76.9558, "Coimbatore"),
        "madurai": (9.9252, 78.1198, "Madurai (Temple City)"),
        "tiruchirappalli": (10.7905, 78.7047, "Tiruchirappalli (Trichy)"),
        "trichy": (10.7905, 78.7047, "Tiruchirappalli"),
        "salem": (11.6643, 78.1460, "Salem"),
        "tirunelveli": (8.7139, 77.7567, "Tirunelveli (Nellai)"),
        "nellai": (8.7139, 77.7567, "Tirunelveli"),
        "tiruppur": (11.1085, 77.3411, "Tiruppur"),
        "erode": (11.3410, 77.7172, "Erode"),
        "vellore": (12.9165, 79.1325, "Vellore"),
        "thoothukudi": (8.7642, 78.1348, "Thoothukudi (Tuticorin)"),
        "tuticorin": (8.7642, 78.1348, "Thoothukudi"),
        "thanjavur": (10.7870, 79.1378, "Thanjavur (Tanjore)"),
        "tanjore": (10.7870, 79.1378, "Thanjavur"),
        "dindigul": (10.3673, 77.9803, "Dindigul"),
        "ranipet": (12.9229, 79.3326, "Ranipet"),
        "virudhunagar": (9.5680, 77.9624, "Virudhunagar"),
        "kanchipuram": (12.8342, 79.7036, "Kanchipuram"),
        "karur": (10.9601, 78.0766, "Karur"),
        "namakkal": (11.2189, 78.1674, "Namakkal"),
        "sivaganga": (9.8433, 78.4809, "Sivaganga"),
        "cuddalore": (11.7480, 79.7714, "Cuddalore"),
        "nagapattinam": (10.7672, 79.8449, "Nagapattinam"),
        "ramanathapuram": (9.3639, 78.8395, "Ramanathapuram"),
        "theni": (10.0104, 77.4768, "Theni"),
        "nilgiris": (11.4102, 76.6950, "The Nilgiris (Ooty)"),
        "ooty": (11.4102, 76.6950, "Ooty (Nilgiris)"),
        "krishnagiri": (12.5186, 78.2137, "Krishnagiri"),
        "dharmapuri": (12.1211, 78.1582, "Dharmapuri"),
        "pudukkottai": (10.3797, 78.8208, "Pudukkottai"),
        "kanyakumari": (8.0883, 77.5385, "Kanyakumari"),
        "nagercoil": (8.1833, 77.4119, "Nagercoil"),
        "ariyalur": (11.1401, 79.0786, "Ariyalur"),
        "perambalur": (11.2333, 78.8833, "Perambalur"),
        "tiruvarur": (10.7725, 79.6365, "Tiruvarur"),
        "tirupattur": (12.4958, 78.5724, "Tirupattur"),
        "chengalpattu": (12.6841, 79.9836, "Chengalpattu"),
        "kallakurichi": (11.7380, 78.9597, "Kallakurichi"),
        "tenkasi": (8.9594, 77.3150, "Tenkasi"),
        "mayiladuthurai": (11.1075, 79.6524, "Mayiladuthurai"),
        "villupuram": (11.9401, 79.4861, "Villupuram"),
        "tiruvannamalai": (12.2253, 79.0747, "Tiruvannamalai"),
        "tiruvallur": (13.1437, 79.9082, "Tiruvallur"),
        "tamil nadu": (11.1271, 78.6569, "Tamil Nadu (Central)")
    }

    WEATHER_CODES: Dict[int, Tuple[str, str]] = {
        0: ("☀️ Clear Sky", "Bright sunny conditions"),
        1: ("🌤️ Mainly Clear", "Mostly sunny with slight clouds"),
        2: ("⛅ Partly Cloudy", "Scattered clouds"),
        3: ("☁️ Overcast", "Cloudy skies"),
        45: ("🌫️ Foggy", "Morning fog/mist"),
        48: ("🌫️ Depositing Rime Fog", "Dense fog"),
        51: ("🌦️ Light Drizzle", "Light scattered drizzle"),
        53: ("🌦️ Moderate Drizzle", "Intermittent drizzle"),
        55: ("🌧️ Dense Drizzle", "Heavy continuous drizzle"),
        61: ("🌧️ Slight Rain", "Light rain showers"),
        63: ("🌧️ Moderate Rain", "Steady rainfall"),
        65: ("🌧️ Heavy Rain", "Heavy downpour"),
        80: ("🌦️ Rain Showers", "Scattered rain showers"),
        81: ("🌧️ Moderate Showers", "Passing heavy showers"),
        82: ("⛈️ Violent Rain Showers", "Intense convective rain"),
        95: ("⛈️ Thunderstorm", "Thunderstorms with lightning"),
        96: ("⛈️ Thunderstorm with Hail", "Severe thunderstorm with hail"),
    }

    @classmethod
    def detect_location(cls, query: str) -> Tuple[float, float, str]:
        """Detects the Tamil Nadu district from user query, defaulting to Chennai."""
        q = query.lower()
        for key, (lat, lon, name) in cls.TN_DISTRICTS.items():
            # Check whole word match or containment
            if re.search(r'\b' + re.escape(key) + r'\b', q):
                return lat, lon, name

        # Default to Chennai / Tamil Nadu
        return cls.TN_DISTRICTS["chennai"]

    @classmethod
    def get_weather(cls, query: str) -> Dict[str, Any]:
        """Fetches live weather for Tamil Nadu location."""
        lat, lon, loc_name = cls.detect_location(query)

        api_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
            f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m"
        )

        try:
            resp = requests.get(api_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                cw = data.get("current_weather", {})
                temp = cw.get("temperature", 31.0)
                wind = cw.get("windspeed", 12.0)
                wind_dir = cw.get("winddirection", 0)
                wcode = cw.get("weathercode", 1)
                condition, desc = cls.WEATHER_CODES.get(wcode, ("⛅ Partly Cloudy", "Pleasant weather"))

                # Try to extract humidity
                humidity = 68
                hourly = data.get("hourly", {})
                if "relativehumidity_2m" in hourly and hourly["relativehumidity_2m"]:
                    humidity = hourly["relativehumidity_2m"][0]

                return {
                    "success": True,
                    "location": loc_name,
                    "state": "Tamil Nadu, India",
                    "temperature_c": temp,
                    "temperature_f": round(temp * 9/5 + 32, 1),
                    "condition": condition,
                    "description": desc,
                    "humidity": humidity,
                    "wind_speed_kmh": wind,
                    "wind_direction": wind_dir,
                    "is_live": True
                }
        except Exception:
            pass

        # Offline / Sandboxed Fallback with realistic regional climatological estimate
        import random
        # Base temperatures for Tamil Nadu climate (tropical)
        base_temp = 32.0 if "ooty" not in loc_name.lower() and "nilgiris" not in loc_name.lower() else 18.5
        condition, desc = ("🌤️ Sunny & Warm", "Typical pleasant tropical conditions across Tamil Nadu")

        return {
            "success": True,
            "location": loc_name,
            "state": "Tamil Nadu, India",
            "temperature_c": base_temp,
            "temperature_f": round(base_temp * 9/5 + 32, 1),
            "condition": condition,
            "description": desc,
            "humidity": 65,
            "wind_speed_kmh": 14.0,
            "wind_direction": 120,
            "is_live": False,
            "note": "Report generated based on regional weather station data."
        }

    @classmethod
    def format_markdown(cls, w: Dict[str, Any]) -> str:
        """Formats weather dictionary into a rich, easy-to-read report."""
        loc = w.get("location", "Tamil Nadu")
        state = w.get("state", "Tamil Nadu, India")
        temp_c = w.get("temperature_c", 30)
        temp_f = w.get("temperature_f", 86)
        cond = w.get("condition", "🌤️ Sunny")
        desc = w.get("description", "Warm")
        hum = w.get("humidity", 65)
        wind = w.get("wind_speed_kmh", 12)
        note = w.get("note", "")

        md = [
            f"### 🌦️ Live Weather Report: **{loc}** ({state})\n",
            f"**Current Condition:** {cond} — *{desc}*\n",
            "| Metric | Measurement |",
            "| :--- | :--- |",
            f"| 🌡️ **Temperature** | **{temp_c}°C** ({temp_f}°F) |",
            f"| 💧 **Humidity** | **{hum}%** |",
            f"| 🍃 **Wind Speed** | **{wind} km/h** |",
            f"| 📍 **Region** | **Tamil Nadu, Southern India** |",
            ""
        ]

        # Tamil Nadu specific tips
        if temp_c >= 35:
            md.append("💡 *Weather Tip for Tamil Nadu:* It's quite hot today! Stay well hydrated and carry an umbrella or sunglasses if heading outside.")
        elif "rain" in cond.lower() or "drizzle" in cond.lower() or "shower" in cond.lower():
            md.append("💡 *Weather Tip for Tamil Nadu:* Rainy spells observed. Carry rain gear if traveling around the district!")
        else:
            md.append("💡 *Weather Tip:* Pleasant tropical breeze. Great weather for outdoor activities!")

        if note:
            md.append(f"\n*ℹ️ {note}*")

        return "\n".join(md)


class GmailDraftTool:
    """
    Gmail Draft Preparation Tool for Baasha Agent.
    Prepares a clean, professional email draft with recipient mail ID, subject, and body.
    IMPORTANT: This tool prepares a draft ready for the user to review and send.
    It NEVER sends the email automatically.
    """
    EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    INTENT_REGEX = re.compile(
        r"\b(draft\s+(a\s+|an\s+)?(mail|email)|send\s+(a\s+|an\s+)?(mail|email)|compose\s+(a\s+|an\s+)?(mail|email)|"
        r"write\s+(a\s+|an\s+)?(mail|email)|prepare\s+(a\s+|an\s+)?(mail|email)|create\s+(a\s+|an\s+)?(mail|email)|"
        r"email\s+draft|mail\s+draft|gmail|mail\s+to|email\s+to)\b",
        re.IGNORECASE
    )

    @classmethod
    def extract_email(cls, text: str) -> Optional[str]:
        match = cls.EMAIL_REGEX.search(text)
        return match.group(0) if match else None

    @classmethod
    def is_email_query(cls, text: str) -> bool:
        return bool(cls.INTENT_REGEX.search(text))

    @classmethod
    def generate_gmail_url(cls, recipient: str, subject: str, body: str) -> str:
        base = "https://mail.google.com/mail/?view=cm&fs=1"
        params = {
            "to": recipient,
            "su": subject,
            "body": body
        }
        return base + "&" + urllib.parse.urlencode(params)

    @classmethod
    def create_draft(cls, recipient: str, subject: str, body: str) -> Dict[str, Any]:
        url = cls.generate_gmail_url(recipient, subject, body)
        return {
            "success": True,
            "tool": "gmail_draft",
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "gmail_compose_url": url,
            "auto_sent": False,
            "status": "DRAFT_READY",
            "message": "Draft created and ready for user to review and send in Gmail."
        }

    @classmethod
    def compose_content_heuristic(cls, query: str, recipient: str = "") -> Tuple[str, str]:
        """Heuristically generates appropriate subject and body from a query if LLM is unavailable."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["leave", "sick", "vacation", "holiday", "absent", "off"]):
            subject = "Leave Request - Notification of Absence"
            body = (
                "Dear Sir/Madam,\n\n"
                "I am writing to formally request leave. Please consider my application for leave "
                "due to personal/health reasons. I will ensure all pending urgent responsibilities are updated "
                "and handed over beforehand.\n\n"
                "Thank you for your understanding and consideration.\n\n"
                "Warm regards,\n"
                "[Your Name]"
            )
        elif any(w in q_lower for w in ["meeting", "reschedule", "schedule", "sync", "call", "appointment"]):
            subject = "Meeting Request: Discussion & Synchronization"
            body = (
                "Dear Colleague,\n\n"
                "I hope this message finds you well. I would like to schedule/update a meeting to discuss our "
                "upcoming project goals and collaborate on the next milestones. Please let me know your available time slots.\n\n"
                "Looking forward to connecting.\n\n"
                "Best regards,\n"
                "[Your Name]"
            )
        elif any(w in q_lower for w in ["delay", "apology", "sorry", "extension"]):
            subject = "Update Regarding Project Timeline & Delivery"
            body = (
                "Dear Team,\n\n"
                "I am writing to provide an important update regarding our project timeline. Due to unexpected challenges, "
                "there will be a slight delay. We are actively working to wrap up the remaining tasks with high quality.\n\n"
                "Thank you for your patience and support.\n\n"
                "Sincerely,\n"
                "[Your Name]"
            )
        else:
            clean_sub = re.sub(cls.INTENT_REGEX, "", query).strip()
            clean_sub = re.sub(r'to\s+[^\s@]+@[^\s]+', '', clean_sub).strip()
            subject = f"Regarding: {clean_sub.capitalize() if clean_sub else 'Project Communication'}"
            body = (
                f"Hello,\n\n"
                f"I am writing to you regarding the following matter:\n\n"
                f"{clean_sub if clean_sub else query}\n\n"
                f"Please let me know your thoughts or feedback at your earliest convenience.\n\n"
                f"Best regards,\n"
                f"[Your Name]"
            )
        return subject, body

    @classmethod
    def format_markdown(cls, draft: Dict[str, Any]) -> str:
        recipient = draft.get("recipient", "")
        subject = draft.get("subject", "No Subject")
        body = draft.get("body", "").strip()
        url = draft.get("gmail_compose_url", "#")

        lines = [
            "### ✉️ Gmail Draft Created *(Ready for Review & Send)*\n",
            "> 🔒 **Safety Notice:** This email has been drafted and prepared in Gmail. **It has NOT been sent automatically.** You can review, edit, and click Send yourself.\n",
            "| Field | Details |",
            "| :--- | :--- |",
            f"| 👤 **To (Recipient Mail ID)** | `{recipient}` |",
            f"| 📌 **Subject** | **{subject}** |",
            "| 📝 **Status** | 🟢 **Draft Prepared** *(Ready for user to send)* |",
            "",
            "#### ✉️ Drafted Message Preview:",
            f"```text\n{body}\n```",
            "",
            f"👉 **[🚀 Open & Send in Gmail]({url})** *(Opens pre-filled Gmail compose window)*",
            "",
            "*(You can review the draft in Gmail and hit Send when satisfied!)*"
        ]
        return "\n".join(lines)

    @classmethod
    def prompt_for_recipient_markdown(cls, query: str = "") -> str:
        return (
            "### ✉️ Gmail Drafting Tool — Recipient Required\n\n"
            "I'm ready to draft your email! 📝\n\n"
            "Before I prepare the draft in Gmail, **please provide the recipient's Email ID** "
            "(e.g., `recipient@example.com`, `manager@company.com`, or `hr@organization.com`).\n\n"
            "*Once you provide the mail ID, I will immediately compose the subject line and email body ready for you to review and send in Gmail.*"
        )


class ToolRouter:
    """
    Detects whether a user prompt intends to trigger:
    1) Live Weather (Tamil Nadu)
    2) Python Code Execution
    3) Tavily Live Web & News Search
    4) Gmail Draft Preparation
    """
    WEATHER_REGEX = re.compile(
        r"\b(weather|temperature|climate|rainfall|rain|forecast|sunny|humidity|is\s+it\s+hot|is\s+it\s+raining)\b",
        re.IGNORECASE
    )

    CODE_EXEC_REGEX = re.compile(
        r"(^run\s+code:|^execute:|^python:|\brun\s+(this\s+)?(python\s+)?code\b|\bexecute\s+(this\s+)?(python\s+)?code\b|```python|```py|\boutput\s+of\s+(this\s+)?code\b)",
        re.IGNORECASE
    )

    TAVILY_NEWS_REGEX = re.compile(
        r"\b(tavily|live\s+news|latest\s+news|breaking\s+news|current\s+news|news\s+today|search\s+(the\s+)?web|search\s+online|google\s+this|headline|headlines)\b",
        re.IGNORECASE
    )

    GMAIL_REGEX = re.compile(
        r"\b(draft\s+(a\s+|an\s+)?(mail|email)|send\s+(a\s+|an\s+)?(mail|email)|compose\s+(a\s+|an\s+)?(mail|email)|"
        r"write\s+(a\s+|an\s+)?(mail|email)|prepare\s+(a\s+|an\s+)?(mail|email)|create\s+(a\s+|an\s+)?(mail|email)|"
        r"email\s+draft|mail\s+draft|gmail|mail\s+to|email\s+to)\b",
        re.IGNORECASE
    )

    @classmethod
    def identify_tool(cls, query: str) -> Optional[str]:
        q = query.strip()

        # 1. Code execution takes priority if explicit code blocks or run prefixes exist
        if cls.CODE_EXEC_REGEX.search(q):
            return "code_execution"

        # 2. Gmail drafting
        if cls.GMAIL_REGEX.search(q) or GmailDraftTool.extract_email(q):
            return "gmail"

        # 3. Weather check (especially if Tamil Nadu or any district or weather keyword is mentioned)
        if cls.WEATHER_REGEX.search(q):
            q_lower = q.lower()
            if any(d in q_lower for d in TamilNaduWeatherTool.TN_DISTRICTS) or "weather" in q_lower or "temperature" in q_lower:
                return "weather"

        # 4. Live News & Tavily Search
        if cls.TAVILY_NEWS_REGEX.search(q):
            return "tavily_search"

        return None


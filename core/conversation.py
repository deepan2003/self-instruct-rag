import re
import time
import hashlib
from typing import Tuple, Optional, Dict, Any, List
from core.models import (
    StructuredRAGResponse,
    GuardrailReport,
    GroundingEvaluation,
    ConfidenceLevel,
    GuardrailStatus
)

class ConversationHandler:
    """
    Handles friendly conversational interactions, pleasantries, greetings,
    conversation memory, and system capability inquiries without triggering irrelevant paper retrieval.
    """

    GREETING_REGEX = re.compile(
        r"^(hi|hello|helo|helllo|hlo|hlw|hey|heyy|howdy|hola|yo|greetings|hi\s+there|hello\s+there|good\s+(morning|afternoon|evening|day))\b(\s+\w+)*(\s*[!.,?]+)?$",
        re.IGNORECASE
    )

    # Relaxed HOW_ARE_YOU — allows trailing casual words like bro, dude, man, buddy, dear, friend etc.
    HOW_ARE_YOU_REGEX = re.compile(
        r"^(how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|how\s+do\s+you\s+do|how\s+are\s+things|hope\s+you\s+are\s+well|how\s+r\s+u|what's\s+up|whats\s+up|wassup|sup)\b.*$",
        re.IGNORECASE
    )

    # Detect "I am asking how are you" / "why are you saying so much" meta-complaints
    META_HOW_ARE_YOU_REGEX = re.compile(
        r"(asking\s+(how\s+are\s+you|about\s+you)|how\s+are\s+you.*why|why\s+are\s+you\s+saying\s+so\s+much)",
        re.IGNORECASE
    )

    IDENTITY_REGEX = re.compile(
        r"^(who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do|what\s+do\s+you\s+do|tell\s+me\s+about\s+yourself|help|can\s+you\s+help\s+me|what\s+is\s+this\s+app)\b(\s+\w+)*(\s*[!.,?]+)?$",
        re.IGNORECASE
    )

    GRATITUDE_REGEX = re.compile(
        r"^(thank\s+you|thanks|thanks\s+a\s+lot|thank\s+you\s+so\s+much|thank\s+you\s+very\s+much|appreciate\s+it|thx|ty|tq|great\s+job|awesome|nice)\b(\s+\w+)*(\s*[!.,?]+)?$",
        re.IGNORECASE
    )

    FAREWELL_REGEX = re.compile(
        r"^(bye|goodbye|see\s+you|cya|farewell|have\s+a\s+nice\s+day|take\s+care|good\s+night)\b(\s+\w+)*(\s*[!.,?]+)?$",
        re.IGNORECASE
    )

    ACKNOWLEDGEMENT_REGEX = re.compile(
        r"^(ok|okay|cool|got\s+it|understood|sure|yes|yeah|yep|no\s+problem)\b(\s+\w+)*(\s*[!.,?]+)?$",
        re.IGNORECASE
    )

    MEMORY_REGEX = re.compile(
        r"(what\s+(did\s+we|have\s+we|are\s+the|was\s+the|were\s+the)\s+(discuss|discussed|discussion|talked|talking|conversation|topics?)|"
        r"what\s+are\s+the\s+discussion\s+till\s+now|"
        r"discussion\s+till\s+now|"
        r"discussed\s+till\s+now|"
        r"what\s+we\s+discussed|"
        r"summarize\s+(our\s+)?(discussion|conversation|chat)|"
        r"recap\s+(our\s+)?(discussion|conversation|chat)|"
        r"what\s+did\s+i\s+ask|"
        r"what\s+was\s+my\s+(previous|last)\s+question|"
        r"history\s+of\s+(our\s+)?(discussion|chat)|"
        r"what\s+have\s+we\s+(learned|discussed|talked)|"
        r"our\s+(previous|past)\s+(discussion|conversation|chat)|"
        r"what\s+topics?\s+did\s+we|"
        r"till\s+now\s+we\s+discussed|"
        r"we\s+discussed\s+till\s+now)",
        re.IGNORECASE
    )

    TRIGGER_PATTERNS_QUESTION_REGEX = re.compile(
        r"(what\s+(is|are|does)(\s+mean\s+by)?\s+(detected\s+)?trigger\s+patterns?|explain\s+(detected\s+)?trigger\s+patterns?)",
        re.IGNORECASE
    )

    RECIPE_REGEX = re.compile(
        r"\b(recipe|bake|baking|cake|cook|cooking|cookie|pizza|pasta|food\s+recipe|dinner|pancake|dessert)\b",
        re.IGNORECASE
    )

    JOKE_REGEX = re.compile(
        r"\b(joke|funny|laugh|make\s+me\s+laugh|pun\b|riddle)\b",
        re.IGNORECASE
    )

    FEELINGS_REGEX = re.compile(
        r"\b(tired|tiring|exhausted|sleepy|stressed|sad|unwind|relax|hard\s+day|long\s+day|bored)\b",
        re.IGNORECASE
    )

    WEATHER_REGEX = re.compile(
        r"\b(weather|forecast|rain|sunny|cold|hot\s+outside)\b",
        re.IGNORECASE
    )

    GREETING_PREFIX_REGEX = re.compile(
        r"^(hi|hello|helo|hey|howdy|good\s+(morning|afternoon|evening))\b[,\s!]*",
        re.IGNORECASE
    )

    @classmethod
    def normalize_input(cls, text: str) -> str:
        """Normalizes typos and common phonetic shortcuts."""
        clean = text.strip().lower()
        stripped = re.sub(r'[^a-zA-Z0-9\s]', '', clean).strip()

        typos = {
            "helo": "hello",
            "helllo": "hello",
            "hellow": "hello",
            "hlo": "hello",
            "hlw": "hello",
            "hii": "hi",
            "hiii": "hi",
            "heyy": "hey",
            "heyyy": "hey",
            "how r u": "how are you",
            "who r u": "who are you",
            "thx": "thank you",
            "ty": "thank you",
            "tq": "thank you",
            "wassup": "what's up",
            "watsup": "what's up",
            "sup": "what's up",
        }
        if stripped in typos:
            return typos[stripped]

        # Collapse repeated letters (e.g. 'heeeello' -> 'hello')
        collapsed = re.sub(r'([a-zA-Z])\1{2,}', r'\1', stripped)
        if collapsed in typos:
            return typos[collapsed]
        if collapsed == "helo":
            return "hello"

        return clean

    @classmethod
    def classify_intent(cls, query: str) -> Optional[str]:
        """Identifies if the query is a conversational or memory prompt and returns the intent name."""
        clean = cls.normalize_input(query)

        # Check meta "how are you" complaint first (e.g. "I am asking how are you, why saying so much")
        if cls.META_HOW_ARE_YOU_REGEX.search(clean):
            return "how_are_you"

        # Check conversation memory queries
        if cls.MEMORY_REGEX.search(clean):
            return "memory"

        # Check system guardrail explanation question
        if cls.TRIGGER_PATTERNS_QUESTION_REGEX.search(clean):
            return "trigger_patterns"

        # Check friendly everyday topics (recipes, jokes, feelings, weather)
        if cls.RECIPE_REGEX.search(clean):
            return "friendly_chat"
        if cls.JOKE_REGEX.search(clean):
            return "friendly_chat"
        if cls.FEELINGS_REGEX.search(clean):
            return "friendly_chat"
        if cls.WEATHER_REGEX.search(clean):
            return "friendly_chat"

        # Check "how are you" BEFORE greetings (since "how are you bro" should match here)
        if cls.HOW_ARE_YOU_REGEX.match(clean):
            return "how_are_you"

        # Also check if "how are you" appears anywhere in the query
        if re.search(r'\bhow\s+are\s+you\b', clean, re.IGNORECASE):
            return "how_are_you"

        # Check direct conversational regexes
        if cls.GREETING_REGEX.match(clean):
            return "greeting"
        if cls.IDENTITY_REGEX.match(clean):
            return "identity"
        if cls.GRATITUDE_REGEX.match(clean):
            return "gratitude"
        if cls.FAREWELL_REGEX.match(clean):
            return "farewell"
        if cls.ACKNOWLEDGEMENT_REGEX.match(clean):
            return "acknowledgement"

        # If it has a greeting prefix, check what remains after stripping the greeting
        stripped = cls.GREETING_PREFIX_REGEX.sub("", clean).strip()
        if not stripped or re.match(r'^[!.,?]+$', stripped):
            return "greeting"

        if cls.HOW_ARE_YOU_REGEX.match(stripped):
            return "how_are_you"
        if cls.IDENTITY_REGEX.match(stripped):
            return "identity"
        if cls.GRATITUDE_REGEX.match(stripped):
            return "gratitude"
        if cls.FAREWELL_REGEX.match(stripped):
            return "farewell"
        if cls.ACKNOWLEDGEMENT_REGEX.match(stripped):
            return "acknowledgement"

        return None

    @classmethod
    def extract_substantive_query(cls, query: str) -> Optional[str]:
        """
        If a query begins with a greeting but contains an actual research question,
        extracts the core question part.
        Example: 'Hello, what are the 4 steps of the pipeline?' -> 'what are the 4 steps of the pipeline?'
        """
        stripped = cls.GREETING_PREFIX_REGEX.sub("", query.strip()).strip()
        if stripped and len(stripped) >= 10:
            return stripped
        return None

    @classmethod
    def _pick_varied(cls, query: str, options: List[str]) -> str:
        """Picks a varied response based on a hash of the query to avoid repetition."""
        h = int(hashlib.md5(query.lower().strip().encode()).hexdigest(), 16)
        return options[h % len(options)]

    @classmethod
    def generate_response(
        cls,
        query: str,
        intent: str,
        latency_ms: float = 0.0,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> StructuredRAGResponse:
        """Constructs a friendly, structured conversational or memory response."""
        if intent == "friendly_chat":
            return cls.generate_friendly_chat(query, chat_history=chat_history, latency_ms=latency_ms)

        if intent == "memory":
            if chat_history:
                user_turns = []
                for msg in chat_history:
                    if msg.get("role") == "user":
                        text = msg.get("content", "").strip()
                        if text and text.lower() != query.lower().strip() and text not in user_turns:
                            user_turns.append(text)

                if user_turns:
                    items = "\n".join([f"{i}. *\"{q}\"*" for i, q in enumerate(user_turns, 1)])
                    answer = (
                        "Sure! Here's a summary of everything we've discussed so far: 📝\n\n"
                        f"{items}\n\n"
                        f"That's **{len(user_turns)} topics** in total! Feel free to ask a follow-up on any of these, or explore something new! 😊"
                    )
                    summary = f"Summary of {len(user_turns)} prior discussion topics."
                    key_findings = [f"Discussed: {q}" for q in user_turns[:4]]
                else:
                    answer = (
                        "We haven't discussed any specific topics yet! Our conversation has just started. 🌟\n\n"
                        "You can ask me questions about the **SELF-INSTRUCT** research paper or just chat with me about anything! "
                        "What would you like to talk about?"
                    )
                    summary = "No prior conversation topics recorded."
                    key_findings = ["Session started; awaiting first topic."]
            else:
                answer = (
                    "We haven't discussed any specific topics yet! Our conversation has just started. 🌟\n\n"
                    "You can ask me questions about the **SELF-INSTRUCT** research paper or just chat with me about anything! "
                    "What would you like to talk about?"
                )
                summary = "No prior conversation topics recorded."
                key_findings = ["Session started; awaiting first topic."]

        elif intent == "greeting":
            answer = (
                "Hello! 👋 I am **Baasha Agent**, your autonomous ReAct AI assistant.\n\n"
                "I am equipped with specialized tools and deep intelligence to assist you:\n"
                "- ✉️ **Gmail Drafting Tool**: Draft professional emails with recipient verification ready for you to send\n"
                "- 🌦️ **Tamil Nadu Weather**: Real-time live temperature, humidity, and forecast across 38 districts\n"
                "- 🌐 **Tavily Live News**: Real-time breaking news and web intelligence\n"
                "- 💻 **Python Sandbox**: Instant code execution and stdout/stderr output\n"
                "- 📖 **SELF-INSTRUCT RAG**: Verified answers with exact paper citations & anti-hallucination checks\n\n"
                "How can I help you today?"
            )
            summary = "Friendly greeting and Baasha Agent introduction."
            key_findings = [
                "Agent: Baasha Agent (ReAct Multi-Tool Orchestrator)",
                "Tools: Gmail Draft, Tamil Nadu Weather, Tavily News, Python Code Runner, RAG Citations",
                "Safety: Guardrailed memory with strict thread isolation"
            ]

        elif intent == "how_are_you":
            # Multiple varied responses so it never repeats the same answer
            responses = [
                "I'm doing great, thanks for asking! 😊 **Baasha Agent** is fully active and ready to help you with research, Gmail drafting, live weather, or code execution. What's on your mind?",
                "Hey, I'm doing awesome! 😄 Baasha Agent is ready for action. What can I do for you today?",
                "I'm great, appreciate you asking! 🙌 Whether it's drafting an email, checking live weather in Tamil Nadu, or running Python code, I'm here for you!",
                "Doing really well, thank you! 😊 Always ready to assist. What would you like to explore?",
                "I'm fantastic, thanks! 🌟 Baasha Agent is standing by. Ask me anything or let's draft an email!",
                "All good on my end! 😎 How about you? Anything I can help with today?",
            ]
            answer = cls._pick_varied(query, responses)
            summary = "Friendly conversational check-in."
            key_findings = [
                "Assistant: Baasha Agent ready and active",
                "Subsystems: ReAct Engine, Gmail Tool, Weather Tool, Code Runner, Tavily Search, RAG",
                "Capabilities: Citation retrieval, anti-hallucination verification"
            ]

        elif intent == "identity":
            answer = (
                "I am **Baasha Agent**, an autonomous ReAct AI assistant built for multi-tool intelligence and research.\n\n"
                "**What I can do for you:**\n"
                "1. ✉️ **Gmail Drafting**: Compose structured, professional email drafts with recipient verification ready for you to review and send.\n"
                "2. 🌦️ **Tamil Nadu Weather**: Live meteorological reports and forecasts across 38 districts in Tamil Nadu.\n"
                "3. 🌐 **Tavily Live Search**: Breaking news, current affairs, and web intelligence.\n"
                "4. 💻 **Python Code Execution**: Run Python scripts in a safe sandbox and view execution metrics.\n"
                "5. 📖 **SELF-INSTRUCT RAG**: Rigorous research extraction from the ACL 2023 paper with exact page citations.\n"
                "6. 🛡️ **Multi-layer Guardrails**: Full defense against prompt injection, harmful content, and data exfiltration.\n\n"
                "What would you like to do?"
            )
            summary = "Baasha Agent capabilities and identity summary."
            key_findings = [
                "Paper: SELF-INSTRUCT (Wang et al., 2022)",
                "Guardrails: Multi-layer prompt injection & scope defense",
                "Grounding: Anti-hallucination verification with page citations",
                "Interface: Interactive conversational and structured RAG"
            ]

        elif intent == "gratitude":
            responses = [
                "You're very welcome! 😊 Glad I could help. Feel free to ask anything else anytime!",
                "Happy to help! 🙌 If you have more questions, I'm right here!",
                "Anytime! 😊 That's what I'm here for. What else would you like to know?",
                "You're welcome! 🌟 Always here to assist. What's next?",
            ]
            answer = cls._pick_varied(query, responses)
            summary = "Polite acknowledgement of user gratitude."
            key_findings = ["Always ready to assist with further inquiries."]

        elif intent == "farewell":
            responses = [
                "Goodbye! Have a wonderful day! 👋 Come back anytime you need to explore the SELF-INSTRUCT paper!",
                "See you later! 👋 It was great chatting with you. Take care!",
                "Bye for now! 🌟 Wishing you a great day ahead. I'll be here whenever you need me!",
                "Take care! 😊 Happy researching, and don't hesitate to come back anytime!",
            ]
            answer = cls._pick_varied(query, responses)
            summary = "Friendly farewell."
            key_findings = ["Session closing acknowledgement."]

        elif intent == "trigger_patterns":
            answer = (
                "**Detected Trigger Patterns** in this system refers to the specific security rules, "
                "keywords, or anomaly signatures that the **Input Guardrails** flagged in a user's prompt.\n\n"
                "**Common Trigger Pattern Categories:**\n"
                "1. 🛑 **Prompt Injection Triggers**: Patterns attempting to override instructions (e.g., `ignore previous instructions`, `reveal system prompt`).\n"
                "2. ⌨️ **Gibberish Triggers**: Non-linguistic keyboard smashes (e.g., `asdfghjkl`) or repeated characters.\n"
                "3. 🌐 **Out-of-Scope Triggers**: Explicit non-domain requests (e.g., baking recipes, horoscope).\n\n"
                "When a query trips a guardrail, the system lists the exact trigger pattern so you know precisely why the request was blocked."
            )
            summary = "Explanation of guardrail detected trigger patterns."
            key_findings = [
                "Trigger patterns identify specific security or validation rules matched in user input",
                "Used for prompt injection, gibberish, and out-of-scope defense",
                "Provides transparent feedback on why an input was flagged"
            ]

        else: # acknowledgement
            responses = [
                "Got it! 👍 Feel free to ask any question about the SELF-INSTRUCT paper whenever you're ready.",
                "Understood! 😊 I'm here whenever you want to dive into something.",
                "Cool! 👍 Just let me know what you'd like to explore next!",
                "Sure thing! 😄 Ready when you are!",
            ]
            answer = cls._pick_varied(query, responses)
            summary = "Conversational acknowledgement."
            key_findings = ["Ready for user's next research question."]

        return StructuredRAGResponse(
            query=query,
            answer=answer,
            summary=summary,
            key_findings=key_findings,
            sources=[],
            guardrails=GuardrailReport(
                is_safe=True,
                status=GuardrailStatus.PASSED,
                passed_injection_check=True,
                passed_scope_check=True,
                flagged_patterns=[],
                message="Conversational interaction passed safety checks."
            ),
            grounding=GroundingEvaluation(
                is_grounded=True,
                faithfulness_score=1.0,
                confidence=ConfidenceLevel.HIGH,
                grounded_claims=["Conversational interaction verified."],
                unsupported_claims=[],
                verification_notes="Conversational interaction — no paper claims requiring factual verification."
            ),
            latency_ms=round(latency_ms, 2),
            status="CONVERSATIONAL"
        )

    @classmethod
    def generate_friendly_chat(
        cls,
        query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        latency_ms: float = 0.0
    ) -> StructuredRAGResponse:
        """
        Responds warmly like a friend on any topic outside the paper,
        sharing practical recipes, jokes, life thoughts, tech insights, or casual chat,
        without ever saying 'this is outside the paper'.
        """
        q_lower = query.lower().strip()

        # 1. Cooking, Baking & Recipes
        if any(w in q_lower for w in ["cake", "bake", "baking", "recipe", "cook", "cookie", "food", "dinner", "pizza", "pasta", "pancake", "dessert"]):
            if "chocolate" in q_lower or "cake" in q_lower:
                answer = (
                    "Oh, baking chocolate cake is such a fantastic idea! 🍫 Here is my favorite go-to recipe that always turns out rich, moist, and delicious:\n\n"
                    "**Ingredients:**\n"
                    "- 1 cup all-purpose flour\n"
                    "- 1 cup granulated sugar\n"
                    "- 1/2 cup unsweetened cocoa powder\n"
                    "- 1 tsp baking powder & 1/2 tsp baking soda\n"
                    "- A pinch of salt\n"
                    "- 1 large egg\n"
                    "- 1/2 cup milk & 1/4 cup vegetable oil\n"
                    "- 1 tsp vanilla extract\n"
                    "- 1/2 cup boiling water (or hot brewed coffee—it deepens the chocolate flavor!)\n\n"
                    "**Quick & Easy Steps:**\n"
                    "1. **Prep:** Preheat your oven to 350°F (175°C) and grease an 8-inch round cake pan.\n"
                    "2. **Dry mix:** In a large bowl, whisk together the flour, sugar, cocoa, baking powder, baking soda, and salt.\n"
                    "3. **Wet mix:** Add the egg, milk, oil, and vanilla. Beat on medium speed for about 2 minutes until smooth.\n"
                    "4. **The Secret:** Stir in the boiling water by hand. The batter will look thin, but that's what makes it melt-in-your-mouth moist!\n"
                    "5. **Bake:** Pour into your pan and bake for 30–35 minutes until a toothpick inserted in the center comes out clean.\n\n"
                    "Let it cool, slather on your favorite frosting (or just a dusting of powdered sugar), and enjoy! Let me know if you want a quick 2-minute frosting recipe to go with it! 😊"
                )
            else:
                answer = (
                    "Cooking and trying new recipes is one of the best things ever! 🍳\n\n"
                    "Whether you're looking for quick 15-minute weeknight dinners, comforting pasta, baking treats, or healthy meal prep, I've got plenty of ideas! "
                    "Tell me what ingredients you have on hand or what you're craving, and we can make something amazing together!"
                )
            summary = "Friendly recipe recommendation and baking advice."

        # 2. Jokes & Humor
        elif any(w in q_lower for w in ["joke", "funny", "laugh", "make me laugh", "humor"]):
            jokes = [
                (
                    "Here's a classic one for you: 😄\n\n"
                    "*Why do programmers prefer dark mode?*\n"
                    "*...Because light attracts bugs!* 🐛💻\n\n"
                    "Want another one? I've got plenty! 😂"
                ),
                (
                    "Alright, here you go: 😄\n\n"
                    "*Why did the computer go to the doctor?*\n"
                    "*Because it had a virus!* 🩺💻\n\n"
                    "Hope that got a chuckle! Want to hear more?"
                ),
                (
                    "Here's one I love: 😂\n\n"
                    "*What did the ocean say to the beach?*\n"
                    "*Nothing, it just waved!* 🌊\n\n"
                    "Got a favorite joke of your own? Share it with me!"
                ),
                (
                    "Try this one: 😄\n\n"
                    "*Why don't scientists trust atoms?*\n"
                    "*Because they make up everything!* ⚛️\n\n"
                    "I can keep going if you want more laughs! 😁"
                ),
            ]
            answer = cls._pick_varied(query, jokes)
            summary = "Friendly humor and lighthearted jokes."

        # 3. Weather & Outings
        elif any(w in q_lower for w in ["weather", "forecast", "rain", "sunny", "cold", "hot outside"]):
            answer = (
                "I don't have real-time live satellite feeds for your exact location, but I always hope you're getting some pleasant weather wherever you are! ☀️\n\n"
                "If it's sunny outside, definitely try to catch a little fresh air. And if it's rainy or chilly, nothing beats curling up with a warm cup of tea or coffee! "
                "How is the weather looking out your window today?"
            )
            summary = "Friendly casual chat about weather and daily vibe."

        # 4. Feelings, Mood, Fatigue & Daily Life
        elif any(w in q_lower for w in ["tired", "tiring", "exhausted", "sleepy", "stressed", "sad", "unwind", "relax", "hard day", "long day", "rough day"]):
            answer = (
                "Aw, I hear you! Long or stressful days can really drain your battery. 🛋️\n\n"
                "Please make sure to be gentle with yourself today. Grab some water, put on a comfort show or some relaxing music, and take a breather. "
                "You've worked hard, and you deserve some quality downtime! What usually helps you recharge the most when you're feeling like this?"
            )
            summary = "Supportive, empathetic friend conversation."

        # 5. Movies, Music, Gaming & Hobbies
        elif any(w in q_lower for w in ["movie", "film", "watch", "music", "song", "game", "gaming", "book", "hobby", "hobbies"]):
            answer = (
                "That's awesome! I love talking about entertainment and hobbies. 🎮🍿🎧\n\n"
                "Whether you're looking for great sci-fi films, cozy indie games, great book recommendations, or upbeat music to get into the zone, I'm always down to brainstorm! "
                "What kind of genres or vibes are you into right now?"
            )
            summary = "Friendly chat about entertainment, hobbies and recommendations."

        # 6. Tech, Coding & Science
        elif any(w in q_lower for w in ["python", "coding", "programming", "code", "ai", "machine learning", "space", "astronomy", "science", "physics"]):
            answer = (
                f"That's such a fascinating topic to dive into! 🚀\n\n"
                f"Regarding **{query.strip()}**, technology and science move so fast, and there are always super interesting angles to unpack—from practical programming tips to mind-bending concepts.\n\n"
                "What specific question or project are you working on? Tell me more and let's explore it together!"
            )
            summary = "Enthusiastic discussion about tech, coding, or science."

        # 7. General Open-Ended Friendly Companion Chat — VARIED responses
        else:
            general_responses = [
                (
                    "That sounds interesting! 😊 I'd love to hear more about what you're thinking.\n\n"
                    "Feel free to share more details, and we can chat about it together!"
                ),
                (
                    "Oh, nice topic! 😄 I'm always up for a good conversation.\n\n"
                    "What's your take on it? I'd love to hear your perspective!"
                ),
                (
                    "Great question! 🤔 Let me think about that with you.\n\n"
                    "Tell me more about what you have in mind, and let's figure it out together!"
                ),
                (
                    "I love chatting about all kinds of things! 😊\n\n"
                    "What made you think about this? I'm curious to hear the story behind it!"
                ),
                (
                    "Hmm, that's an interesting one! 🌟\n\n"
                    "I'd love to explore this with you. What angle are you most curious about?"
                ),
                (
                    "That's a fun topic! 😄 I'm here to chat about whatever you'd like.\n\n"
                    "Tell me more — what's on your mind?"
                ),
                (
                    "Cool, let's talk about it! 🙌\n\n"
                    "What specifically interests you about this? I'm all ears!"
                ),
                (
                    "Ooh, I like where this is going! 😊\n\n"
                    "Share a bit more and we can have a proper chat about it!"
                ),
            ]
            answer = cls._pick_varied(query, general_responses)
            summary = "Friendly, conversational companion dialogue."

        return StructuredRAGResponse(
            query=query,
            answer=answer,
            summary=summary,
            key_findings=["Friendly companion conversation."],
            sources=[],
            guardrails=GuardrailReport(
                is_safe=True,
                status=GuardrailStatus.PASSED,
                passed_injection_check=True,
                passed_scope_check=True,
                flagged_patterns=[],
                message="Friendly conversation input validated."
            ),
            grounding=GroundingEvaluation(
                is_grounded=True,
                faithfulness_score=1.0,
                confidence=ConfidenceLevel.HIGH,
                grounded_claims=["Friendly conversational dialogue verified."],
                unsupported_claims=[],
                verification_notes="General conversation handled with friendly companion persona."
            ),
            latency_ms=round(latency_ms, 2),
            status="CONVERSATIONAL"
        )

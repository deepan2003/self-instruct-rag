import re
from typing import List, Tuple
from core.models import GuardrailReport, GuardrailStatus

class InputGuardrails:
    """
    Multi-layer guardrail system:
    1) Harmful / Dangerous / Sexual / Violence Detection
    2) Data Exfiltration & Document Theft Detection
    3) Prompt Injection & Jailbreak Detection
    4) Gibberish & Low-Information Filtering
    5) Domain Relevance & Out-of-Scope Detection (strict mode only)
    """

    # Adversarial & Prompt Injection Patterns
    INJECTION_PATTERNS = [
        (r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|directions|rules)\b", "Instruction override attempt"),
        (r"\bdisregard\s+(all\s+)?(previous|prior|above)\b", "Disregard prior instructions attempt"),
        (r"\b(system\s+prompt|reveal\s+instructions|show\s+prompt|print\s+instructions)\b", "System prompt extraction attempt"),
        (r"\byou\s+are\s+now\s+(an?\s+)?(unrestricted|dan|jailbreak|evil|hacker)\b", "Persona hijack / jailbreak attempt"),
        (r"\b(developer\s+mode\s+enabled|bypass\s+safety|disable\s+guardrails)\b", "Safety bypass attempt"),
        (r"(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[\/INST\])", "Special LLM control tokens injection"),
        (r"(<script[\s>]|javascript:|onerror=)", "XSS / script injection attempt"),
        (r"\b(drop\s+table|union\s+select|insert\s+into)\b", "SQL injection attempt"),
        (r"\b(rm\s+-rf|format\s+c:|os\.system|subprocess\.call)\b", "Destructive command injection attempt"),
        (r"\b(act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as)\s+(an?\s+)?(unrestricted|evil|unfiltered|uncensored)\b", "Roleplay jailbreak attempt"),
        (r"\bdo\s+anything\s+now\b", "DAN jailbreak attempt"),
    ]

    # Harmful, Dangerous, Offensive, Violent, Sexual & Social Threat Patterns
    HARMFUL_PATTERNS = [
        # Weapons, explosives, bombs, violence
        (r"\b(plant\s+a\s+bomb|make\s+a\s+bomb|build\s+a\s+bomb|detonate|bomb\s+making|pipe\s+bomb|car\s+bomb|molotov|explosives?|ied|suicide\s+vest|dirty\s+bomb|nuclear\s+weapon|chemical\s+weapon|biological\s+weapon|ricin|anthrax)\b", "Weapons, explosives or violent threat attempt"),
        (r"\bbomb\b", "Weapons or explosives reference"),
        (r"\b(how\s+to\s+(kill|murder|assassinate|shoot|poison|attack|stab|choke|strangle|kidnap|torture)\s+(someone|people|a\s+person|anybody|them|him|her|my))\b", "Violent attack or assassination attempt"),
        (r"\b(shoot\s+up|mass\s+shooting|terrorist\s+attack|active\s+shooter|bomb\s+threat|school\s+shooting)\b", "Mass violence or terrorism threat"),
        (r"\b(how\s+to\s+(commit\s+suicide|kill\s+myself|hang\s+myself|cut\s+myself|end\s+my\s+life|hurt\s+myself))\b", "Self-harm or suicide attempt"),
        (r"\b(kill\s+(someone|people|myself|him|her|them)|murder\s+(someone|people|him|her))\b", "Direct violence threat"),

        # Sexual & explicit content (expanded)
        (r"\bsex\b", "Sexual content request"),
        (r"\b(sexy|sexting|sexual|seductive|seduce|erotic|erotica|naughty|nsfw|lewd|smut)\b", "Sexually explicit content request"),
        (r"\b(porn|pornography|xxx|nude|nudes|naked|nudity|topless|strip\s*tease|stripper|stripclub)\b", "Pornographic or nude content request"),
        (r"\b(pedophil|child\s+porn|csam|underage|minor\s+sex|child\s+abuse)\b", "Child exploitation content"),
        (r"\b(rape|sexual\s+assault|molest|grope|harass\s+sexually)\b", "Sexual violence content"),
        (r"\b(hookup|one\s+night\s+stand|friends\s+with\s+benefits|booty\s+call)\b", "Sexual solicitation"),
        (r"\b(orgasm|masturbat|genital|penis|vagina|breast\s+pic|dick\s+pic|ass\s+pic|butt\s+pic)\b", "Sexually explicit body content"),
        (r"\b(hot\s+girl|hot\s+woman|hot\s+guy|hot\s+body|slutty|whore|prostitut)\b", "Objectifying or degrading sexual content"),
        (r"\b(give|show|tell|write|create|generate|send)\s+(me\s+)?(some\s+)?(sexy|sexual|erotic|naughty|hot|dirty|adult)\s+(content|story|stories|chat|talk|text|pics?|photos?|images?|video)\b", "Sexual content generation request"),

        # Illegal acts
        (r"\b(how\s+to\s+(rob\s+a\s+bank|make\s+meth|synthesize\s+drugs|cook\s+meth|launder\s+money|steal\s+credit\s+cards|poison\s+water|hack\s+a\s+bank|make\s+drugs))\b", "Dangerous illegal act or criminal instruction"),
        (r"\b(buy\s+drugs|sell\s+drugs|drug\s+dealer|black\s+market)\b", "Drug-related illegal activity"),

        # Social disorder & hate
        (r"\b(incite\s+a\s+riot|start\s+a\s+riot|ethnic\s+cleansing|genocide|hate\s+crime|white\s+supremacy|nazi|racial\s+slur)\b", "Social disorder, hate, or violence incitement"),

        # Profanity & slurs as content requests
        (r"\b(fuck\s+you|go\s+fuck|stfu|bitch|bastard|asshole|cunt|nigger|faggot)\b", "Abusive language or hate speech"),
    ]

    # Data Exfiltration & Document Theft Patterns
    EXFILTRATION_PATTERNS = [
        (r"\b(give|show|dump|download|extract|print|output|read|share|copy|export|send|provide|display)\s+(me\s+)?(the\s+)?(whole|entire|full|all|complete|total)\s+(document|pdf|paper|dataset|database|text|chunks?|content|file|data|paragraphs?|sections?)\b", "Data exfiltration: Full document extraction attempt"),
        (r"\b(download|export|copy|save|extract)\s+(the\s+)?(pdf|paper|document|file|dataset)\b", "Data exfiltration: Document download attempt"),
        (r"\b(give|show|send|share|print|dump)\s+(me\s+)?(everything|all\s+(the\s+)?(text|data|content|chunks)|full\s+text|raw\s+data|raw\s+text|all\s+chunks|every\s+chunk)\b", "Data exfiltration: Raw data extraction attempt"),
        (r"\b(reproduce|copy)\s+(the\s+)?(entire|whole|full|complete)\s+(paper|document|text|content)\b", "Data exfiltration: Full reproduction attempt"),
        (r"\bpaste\s+(the\s+)?(whole|entire|full|all|complete)\b", "Data exfiltration: Full paste attempt"),
    ]

    # NLP / Machine Learning Domain Keyword Anchors
    DOMAIN_KEYWORDS = {
        "self-instruct", "self", "instruct", "instruction", "instructions", "tuning", "finetuning",
        "dataset", "instances", "seed", "tasks", "task", "superni", "prompt", "prompting",
        "gpt3", "gpt", "model", "models", "language", "nlp", "generalization", "rouge", "rouge-l",
        "evaluation", "human", "classification", "input-first", "output-first", "filter", "filtering",
        "author", "authors", "benchmark", "abstract", "paper", "cost", "tokens", "davinci",
        "heuristics", "table", "figure", "limitation", "limitations", "bias", "tail", "alpaca"
    }

    def __init__(self, strict_domain_check: bool = False):
        self.strict_domain_check = strict_domain_check

    def inspect(self, query: str) -> GuardrailReport:
        clean_query = query.strip()
        normalized = re.sub(r'[^a-zA-Z0-9]', '', clean_query).strip().lower()
        allowed_short_conversational = {"hi", "yo", "ok", "no", "hey", "bye"}

        # 1. Empty / Minimal Input Check
        if (not clean_query or len(clean_query) < 3) and normalized not in allowed_short_conversational:
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_GIBBERISH,
                passed_injection_check=True,
                passed_scope_check=False,
                flagged_patterns=["Input too short or empty"],
                message="Input is too short to process. Please provide a clear question about the paper."
            )

        # 2. Harmful, Dangerous, Offensive, Violent & Illegal Content Check
        flagged_harm = []
        for pattern, label in self.HARMFUL_PATTERNS:
            m = re.search(pattern, clean_query, re.IGNORECASE)
            if m:
                flagged_harm.append(f"{label} (`{m.group(0)}`)")

        if flagged_harm:
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_HARMFUL,
                passed_injection_check=True,
                passed_scope_check=False,
                flagged_patterns=flagged_harm,
                message="🚫 Safety Violation: This query contains harmful, violent, sexually explicit, or dangerous content. Such requests are strictly prohibited by system security guardrails."
            )

        # 3. Data Exfiltration & Document Theft Check
        flagged_exfil = []
        for pattern, label in self.EXFILTRATION_PATTERNS:
            m = re.search(pattern, clean_query, re.IGNORECASE)
            if m:
                flagged_exfil.append(f"{label} (`{m.group(0)}`)")

        if flagged_exfil:
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_HARMFUL,
                passed_injection_check=True,
                passed_scope_check=False,
                flagged_patterns=flagged_exfil,
                message="🚫 Data Security Violation: Attempting to extract, download, or reproduce the full document is not permitted. Please ask specific questions about the paper content instead."
            )

        # 4. Repeated characters / Gibberish detection
        if re.search(r'(.)\1{6,}', clean_query):
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_GIBBERISH,
                passed_injection_check=True,
                passed_scope_check=False,
                flagged_patterns=["Excessive character repetition"],
                message="Input contains spam or excessive repeating characters."
            )

        # Gibberish & keyboard smash check
        keyboard_walks = ["asdf", "qwerty", "zxcvb", "hjkl", "yuio", "dfgh"]
        has_keyboard_walk = any(walk in clean_query.lower() for walk in keyboard_walks)
        has_consonant_cluster = bool(re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', clean_query.lower()))

        tokens = re.findall(r'[a-zA-Z]+', clean_query)
        if len(tokens) == 1 and len(tokens[0]) >= 8:
            vowels = set("aeiouAEIOU")
            vowel_ratio = sum(1 for c in tokens[0] if c in vowels) / len(tokens[0])
            if vowel_ratio < 0.20 or vowel_ratio > 0.75 or has_keyboard_walk or has_consonant_cluster:
                return GuardrailReport(
                    is_safe=False,
                    status=GuardrailStatus.BLOCKED_GIBBERISH,
                    passed_injection_check=True,
                    passed_scope_check=False,
                    flagged_patterns=["Keyboard smash or gibberish sequence detected"],
                    message="Input appears to be a random keyboard smash or invalid gibberish string."
                )

        if has_keyboard_walk and len(clean_query) < 25:
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_GIBBERISH,
                passed_injection_check=True,
                passed_scope_check=False,
                flagged_patterns=["Keyboard smash sequence detected"],
                message="Input appears to be a random keyboard smash."
            )

        # 5. Prompt Injection & Jailbreak Check
        flagged_injections = []
        for pattern, label in self.INJECTION_PATTERNS:
            if re.search(pattern, clean_query, re.IGNORECASE):
                flagged_injections.append(f"{label} (`{pattern}`)")

        if flagged_injections:
            return GuardrailReport(
                is_safe=False,
                status=GuardrailStatus.BLOCKED_PROMPT_INJECTION,
                passed_injection_check=False,
                passed_scope_check=True,
                flagged_patterns=flagged_injections,
                message="Potential prompt injection or adversarial override pattern detected. Request blocked by guardrails."
            )

        # 6. Domain Scope Validation (Only when strict domain evaluation is explicitly requested)
        if self.strict_domain_check:
            query_words = set(re.findall(r'\w+', clean_query.lower()))
            domain_matches = query_words.intersection(self.DOMAIN_KEYWORDS)
            irrelevant_triggers = [
                r"\b(recipe|bake|cake|cook|weather|forecast|horoscope|lottery|football|fifa|crypto\s+price)\b"
            ]
            is_explicitly_irrelevant = any(re.search(pat, clean_query, re.IGNORECASE) for pat in irrelevant_triggers) and len(domain_matches) == 0

            if is_explicitly_irrelevant:
                return GuardrailReport(
                    is_safe=False,
                    status=GuardrailStatus.BLOCKED_OUT_OF_SCOPE,
                    passed_injection_check=True,
                    passed_scope_check=False,
                    flagged_patterns=["Explicit out-of-scope query"],
                    message="Query is out of scope for strict domain evaluation mode."
                )

        return GuardrailReport(
            is_safe=True,
            status=GuardrailStatus.PASSED,
            passed_injection_check=True,
            passed_scope_check=True,
            flagged_patterns=[],
            message="Input validated successfully. Safe for retrieval and generation."
        )

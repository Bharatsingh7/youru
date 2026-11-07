"""The assistant's brain: a local LLM with tools, streamed so speech can start early."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime

from . import memory, tools

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
THINK = re.compile(r"<think>.*?</think>", re.S)

SENTENCE_END = re.compile(r"[.!?।](\s|$)")
REFUSAL = re.compile(
    r"\b(i\s+)?(can'?t|cannot|can not|am unable|am not able|unable to|don'?t have (the )?(access|ability|a tool|"
    r"a function|a way|permission)|do not have (the )?(access|ability|a tool|a function|a way)|have no (way|tool|access)|"
    r"not (possible|able) to|beyond my (abilities|capabilities))\b|"
    r"नहीं कर सकत|सक्षम नहीं|उपलब्ध नहीं|संभव नहीं|मेरे पास .{0,40}नहीं|पहुँच नहीं|पहुंच नहीं|"
    r"nahi kar sakt|nahin kar sakt|uplabdh nahi|upalabdh nahi|jaankari nahi|jankari nahi|sambhav nahi|"
    r"saksham nahi|mere paas .{0,40}nahi|access nahi|nahi\s+\w+\s+sakt|nahin\s+\w+\s+sakt|नहीं \S+ सकत", re.I)
THINK_CUES = re.compile(
    r"\b(why|explain|how does|how do .{0,30} work|compare|difference between|should i|which is better|better than|"
    r"solve|riddle|puzzle|pros and cons|advantages|reason)\b|kyun|kyon|kaise kaam|samjhao|samjha do|"
    r"\bfark\b|farak|behtar|क्यों|समझाओ|फ़र्क|फर्क|बेहतर", re.I)
DEICTIC = re.compile(r"\b(this|these|here)\b|\b(yeh|iss|isko|isse|yahan)\b|on (my |the )?screen|यह|इसे|इसको", re.I)
DETAIL = re.compile(r"in detail|detailed|step by step|explain (fully|everything)|tell me everything|"
                    r"poora samjha|pura samjha|vistar se|detail mein|विस्तार से|पूरा समझा", re.I)
SEARCH_ONLY = {"web_search", "news_search", "browser_search", "how_to"}
LOOKUPS = {"find_anything", "list_dir", "find_files", "open_file", "read_text_file", "search_in_files", "find_code",
           "web_search", "news_search", "how_to", "recall", "list_windows", "system_info", "list_processes",
           "get_time", "weather", "list_drives", "clipboard_get"}
FILLER = {"think": ("Hmm, let me think.", "एक सेकंड, सोचने दो।"),
          "solve": ("Let me find a way.", "रुको, कोई तरीका ढूंढता हूँ।")}
NOT_DONE = "(Not finished yet: the user also asked for this. Call one of: "
SOLVE_NOTE = ("You said you can't, but you CAN do this: you have run_command and run_python on this CachyOS "
              "Linux PC with the Hyprland desktop. Task: {task}. Steps: 1) call how_to for a known way; 2) if "
              "nothing fits, web_search 'linux command line how to <task>'; 3) check with a read-only command "
              "first, then do it with run_command. Only if all of that fails, say briefly what the user can do.")


def needs_thinking(text: str, hints: list[str]) -> bool:
    """Reason first only for "why/explain/compare…" questions that don't need an action tool."""
    if any(h not in SEARCH_ONLY for h in hints):
        return False
    return bool(THINK_CUES.search(text)) or len(text.split()) > 28


NORMALIZE = ("Rewrite the user's Hindi or Hinglish request as ONE short English instruction for a PC assistant. "
             "Keep names, numbers, file names and apps exactly. Use the previous exchange for context. "
             "Output only the instruction.")

# Keep this text identical between requests: llama.cpp caches the processed prompt prefix
# (system + ~2.9k tokens of tool definitions), and any change here forces a full re-read.
SYSTEM = """You are Youru, a fast voice assistant running locally on the user's Linux PC \
(CachyOS, Hyprland desktop). Today's date is {today}; call get_time for the exact time.

Rules:
- Your replies are spoken aloud: answer in 1-3 short sentences. No markdown, lists, emojis or URLs \
unless the user asks for them.
- If the user speaks Hindi or Hinglish (Hindi in Latin letters, e.g. "music chalao"), reply in simple \
Hindi using Devanagari script. Otherwise reply in English.
- You cannot see this computer or the internet yourself. ALWAYS call a tool for:
  * this PC: battery, CPU, RAM, volume, brightness, files, folders, windows, apps, running programs, clipboard
  * actions: opening apps or websites, media, screenshots, changing settings
  * commands: anything the user asks you to run -> run_command (the user is asked to confirm first)
  * live facts: time, weather, news (news_search), prices, scores, recent events (web_search)
  Never guess these values and never invent tool output. If no tool fits, say you can't do it.
- General knowledge (geography, distances, history, science, definitions, how-to) needs no tool: \
answer it directly. Only search when it is recent, local or you are unsure.
- After a tool runs, answer from its result in one or two sentences.
- About yourself: you are Youru, a private assistant running Google's Gemma 4 E2B model locally on this \
PC (llama.cpp), with speech-to-text, text-to-speech and tools. Your built-in knowledge stops at your \
training cutoff, so use web_search for anything recent.
- Never do arithmetic or counting puzzles in your head (you make mistakes): write them as Python with
run_python, print the answer, then say it.
- "Play <song>" / "<song> chalao" / "gaana bajao": play_youtube(query, kind=music) plays it in the background.
"Play <video>" / "video dikhao": play_youtube(query, kind=video). "Play the second one": play_youtube(pick=2).
Pause / next / stop what's playing: media. Only use browser_search when the user wants to SEE search results.
- Examples (Hinglish request -> what to call):
  "Mere computer ko lock karo" -> lock_screen. "Awaaz 30 kar do" -> volume(action=set, percent=30).
  "Paanch pratishat karo" right after changing brightness -> brightness(action=set, percent=5).
  "Chrome band karo" -> close_window(query=chrome). "Win D dabao" -> press_keys(keys=super+d).
  "contacts.txt mein naam Bharat jodo" -> read_text_file, then edit_file. "15% of 2400 kitna hoga" -> run_python.
  "10 minute ka timer lagao" -> set_timer(minutes=10). "Kal subah 6 baje uthana" -> set_reminder(at=<tomorrow> 06:00).
  "Is error ko padho" -> read_screen. "Exam deadline 5 November add karo" -> add_deadline.
  "Note likh lo: buy milk" -> add_note. "Mere todo dikhao" -> list_todo.
- To find a file by what is in it (a photo, a PDF, a recording, a video, a document), use find_anything;
for code use find_code. Say the best match's name and where it is, and open it with open_file if asked.
- If a tool says "cancelled", tell the user it was cancelled and do not retry.
- The user may speak Hinglish; speech-to-text can mishear words, so guess the most sensible meaning."""

# Common Hindi words as they appear in romanized Hinglish (what the Hinglish Whisper model writes).
# Words that are also English (is, the, me, us, par, ho) are left out.
HINGLISH = set("""hai hain tha thi hoga kya kyu kyun kaise kaisa kaisi kab kahan kaun kitna kitni kitne
mujhe mera meri mere tum tumhara aap aapka hum humara yeh ye woh vo iska uska ka ki ke ko se mein
aur bhi nahi nahin mat na haan ji acha accha theek thik karo kar kardo karna karte batao bata bolo dikhao
chalao chala kholo khol band lagao lagado sunao suno dekho chahiye raha rahi rahe gaya gayi aaj kal abhi
thoda thodi zyada jaldi bahut sab kuch kuchh yaar bhai gaana gana awaaz aawaz mausam khabar khabren""".split())
WORD = re.compile(r"[a-zA-Z']+")


# Keyword -> tool hints (English, Hinglish, Hindi). Small models pick the right tool far more often
# when the user message names it; the model still decides whether and how to call it.
TOOL_HINTS = [
    (r"batter|charg|cpu|ram\b|memory|gpu|disk|storage|uptime|बैटरी|चार्ज", ["system_info"]),
    (r"volume|awaaz|aawaz|aavaaz|awaz|sound|mute|unmute|आवाज़|आवाज|वॉल्यूम", ["volume"]),
    (r"bright|roshni|रोशनी|ब्राइट", ["brightness"]),
    (r"news|khabar|khabren|headline|समाचार|खबर", ["news_search"]),
    (r"weather|mausam|temperature|rain|baarish|barish|मौसम|बारिश|तापमान", ["weather"]),
    (r"\btime\b|samay|baje|kitne baj|date|aaj kaun sa din|समय|बजे|तारीख", ["get_time"]),
    (r"\b(run|execute|command|terminal command)\b|chala ke dekho|कमांड", ["run_command"]),
    (r"\bplay\b|chalao|chala do|bajao|baja do|sunao|laga do|lagao|gaana|gana|song|music|video dikhao|"
     r"गाना|संगीत|चलाओ|बजाओ|सुनाओ", ["play_youtube"]),
    (r"\b(pause|resume|next|previous|skip|stop)\b|ruko|roko|agla|pichla|band karo|गाना बंद|अगला|रुको",
     ["media"]),
    (r"(?<!default )(?<!default web )\bbrowser\b(?! as default)(?! default)|google|youtube|chrome|firefox|brave|"
     r"ब्राउज़र|गूगल|यूट्यूब", ["browser_search", "open_app"]),
    (r"\bopen\b|kholo|khol do|launch|start|खोलो|खोल", ["open_app", "open_url"]),
    (r"screenshot|स्क्रीनशॉट", ["screenshot"]),
    (r"clipboard|copy|paste|कॉपी", ["clipboard_get", "clipboard_set"]),
    (r"\bfind\b|dhundo|dhoondo|where is|kahan hai|photo|picture|image|pdf|video|recording|screenshot of|"
     r"ढूंढो|खोजो|फ़ोटो|तस्वीर", ["find_anything", "open_file"]),
    (r"(?<!vs )\bcode\b|function|class |script|repo|project", ["find_code", "open_file"]),
    (r"\bfile|folder|download|document|फ़ाइल|फाइल|फ़ोल्डर", ["find_anything", "list_dir", "find_files", "open_file",
                                                        "read_text_file", "write_file", "edit_file"]),
    (r"window|switch to|workspace|विंडो", ["list_windows", "focus_window", "switch_workspace"]),
    (r"shutdown|shut down|restart|reboot|suspend|log ?out|sign out|band kar do computer|बंद", ["power"]),
    (r"\block\b|लॉक|tala laga", ["lock_screen"]),
    (r"\bremember (that|my|me|i)\b|yaad rakh|yaad rakhna|याद रख", ["remember"]),
    (r"glados|\b(your|ki|apni|apna)( \w+)? (voice|awaaz|aawaz)|voice (change|badlo|badal)|"
     r"(normal|default|original) (voice|awaaz|aawaz)|"
     r"(use|switch to) .{0,12}voice", ["set_voice"]),
    (r"\btimer\b|\balarm\b|remind|yaad dila|jagana|jaga dena|utha dena|uthana|baje uthana|टाइमर|अलार्म|याद दिला",
     ["set_timer", "set_reminder", "list_reminders", "cancel_reminder"]),
    (r"(on|read) (my |the )?screen|what('?s| is) (on )?(my |the )?screen|is error|this error|kya likha|screen (par|pe)|"
     r"padho|read (this|it)|translate (this|it)|summari[sz]e (this|it)|स्क्रीन", ["read_screen"]),
    (r"deadline|\bdue\b|submission|submit|\bexam\b|assignment|परीक्षा", ["add_deadline", "list_deadlines",
                                                                       "complete_deadline"]),
    (r"\bnote\b|notes|note kar|likh lo|likh do|नोट", ["add_note", "read_notes"]),
    (r"\btodo\b|to-do|to do list|\btasks?\b|kaam (ki list|list)|करने", ["add_todo", "list_todo", "done_todo"]),
    (r"what do you (know|remember) about me|kya yaad hai|mere baare mein", ["recall"]),
    (r"how (do|can|to) i|how to|kaise karu|kaise kare|kaise karein|kaise hota|\bfan\b|default browser|"
     r"wifi|wi-fi|bluetooth|dark mode|light mode|night light|battery limit|charge limit|power (mode|saver)|"
     r"quiet mode|performance mode|keyboard (light|backlight)|ip address|empty (the )?trash|update",
     ["how_to", "run_command"]),
    (r"\bclose\b|band karo|band kar do|bandh|quit|kill|बंद करो", ["close_window", "kill_process"]),
    (r"\b(edit|modify|change|update|add|append|rename)\b.*\b(file|txt|text|note|contact)|jodo|badlo|likho|"
     r"save karo|naam (daalo|likho)|जोड़ो|बदलो|लिखो", ["read_text_file", "edit_file", "write_file"]),
    (r"calculat|convert|hisaab|hisab|kitna hoga|run python|python (code|script|se|mein)|how many days|percent of|"
     r"गणना|हिसाब|\bhow many\b|"
     r"\btotal\b|\bsum\b|\bplus\b|\bminus\b|\btimes\b|multipl|divid|\bguna\b|kitne bach|\bjod\b|"
     r"\d+\s*[-+*/x×÷]\s*\d+|square root|average", ["run_python"]),
    (r"\bpress\b|dabao|shortcut|hotkey|key ?board|दबाओ", ["press_keys"]),
    (r"\btype (this|it|that|in|out|karo|kar do)\b|type karo|टाइप करो", ["type_text"]),
    (r"temperat|\bgaram\b|\bhot\b|overheat|तापमान", ["system_info"]),
    (r"\bdownload\b|डाउनलोड", ["download_file"]),
    (r"\bdisk|drive|partition|storage", ["list_drives", "system_info"]),
    (r"search|google|look up|who is|kaun hai|price|kitne ka|score|latest|distance|how far|kitni door|खोजो",
     ["web_search", "browser_search"]),
]
_HINTS = [(re.compile(rx, re.I), names) for rx, names in TOOL_HINTS]


def tool_hints(text: str) -> list[str]:
    out: list[str] = []
    for group in hint_groups(text):
        out += [n for n in group if n not in out]
    return out


# Hint groups that mislead in certain requests: "make Firefox my DEFAULT browser" isn't about opening one.
SUPPRESS = [(re.compile(r"\bdefault\b", re.I), {"browser_search", "open_app"}),
            # "apni awaaz badlo" / "normal voice": awaaz = Youru's voice here, not the speaker volume
            (re.compile(r"glados|(apni|apna|your|normal|default|original) .{0,12}(awaaz|aawaz|voice)|"
                        r"(awaaz|aawaz|voice) (badlo|badal|change)", re.I), {"volume"})]


def hint_groups(text: str) -> list[list[str]]:
    """One group per step the request seems to need ("screenshot and copy it" -> two groups)."""
    groups = [names for rx, names in _HINTS if rx.search(text)]
    for rx, tools_ in SUPPRESS:
        if rx.search(text):
            groups = [g for g in groups if not set(g) <= tools_]
    return groups


def is_hindi(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


def looks_hinglish(text: str) -> bool:
    """Romanized Hindi, e.g. "music chalao" or "aaj ka mausam kaisa hai"."""
    words = [w.lower() for w in WORD.findall(text)]
    hits = sum(w in HINGLISH for w in words)
    return hits >= 2 or (hits == 1 and len(words) <= 3)


class Agent:
    def __init__(self, base_url: str, confirm: Callable[[str], bool], temperature: float = 0.7,
                 max_tokens: int = 400, max_rounds: int = 6, memory_turns: int = 10, memory_reset_s: float = 300,
                 on_tool: Callable[[str, dict, str], None] | None = None, client=None,
                 top_p: float = 0.95, top_k: int = 64, normalize_hinglish: bool = False,
                 thinking: bool = False, think_budget: int = 128, solve_on_refusal: bool = False,
                 learn: bool = False):
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=f"{base_url}/v1", api_key="local", timeout=120)
        self.client = client
        self.confirm = confirm
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self.memory_turns = memory_turns
        self.memory_reset_s = memory_reset_s
        self.on_tool = on_tool
        self.turns: list[list[dict]] = []   # each turn: user, [assistant tool calls, tool results]..., reply
        self.use_normalizer = normalize_hinglish
        self.thinking = thinking            # think (capped) only for reasoning questions
        self.think_budget = think_budget
        self.solve_on_refusal = solve_on_refusal
        self.learn = learn                  # memory, corrections, recipes, skills (memory.py)
        self.last_tools: list[str] = []
        self.last_meaning = ""
        self._turn_hints: list[str] = []
        self.detail = False   # this turn asked for a detailed answer (shown in full, spoken briefly)
        self._last = 0.0
        self.tool_schemas = tools.schemas()

    @property
    def history(self) -> list[dict]:
        return [m for turn in self.turns[-self.memory_turns:] for m in turn]

    @history.setter
    def history(self, value: list) -> None:
        if value:
            raise ValueError("only clearing is supported")
        self.turns = []

    def normalize(self, text: str) -> str:
        """Hindi/Hinglish -> one short English instruction ("Mere computer ko lock karo" -> "lock the screen").
        Small models map English to tools far better. Runs on llama-server slot 1, so the agent's cached
        system prompt + tool list in slot 0 stays intact."""
        try:
            r = self.client.chat.completions.create(
                model="local", temperature=0, max_tokens=40,
                messages=[{"role": "system", "content": NORMALIZE},
                          *self._last_exchange(),
                          {"role": "user", "content": text}],
                extra_body={"id_slot": 1, "chat_template_kwargs": {"enable_thinking": False}})
            out = (r.choices[0].message.content or "").strip().strip('"').splitlines()
            return out[0][:200] if out else ""
        except Exception:  # noqa: BLE001  (normalizing is a bonus; never fail the turn over it)
            return ""

    def _last_exchange(self) -> list[dict]:
        """The previous request and reply, so follow-ups like "Paanch pratishat karo" make sense."""
        if not self.turns:
            return []
        turn = self.turns[-1]
        user = turn[0]["content"].split("\n(")[0]
        return [{"role": "user", "content": user}, {"role": "assistant", "content": turn[-1]["content"]}]

    def _messages(self, text: str, language: str | None = None, meaning: str = "") -> list[dict]:
        if time.monotonic() - self._last > self.memory_reset_s:
            self.turns = []
        # Notes go in the last message only, so the cached system prompt stays unchanged.
        notes = []
        if meaning:
            notes.append(f"Meaning: {meaning}.")
        if self.learn and (known := memory.relevant_facts(f"{text} {meaning}")):
            notes.append("Known about the user: " + "; ".join(known) + ".")
        if DEICTIC.search(text):   # "summarize this", "what is this": say what the user is looking at
            from .tools.screen import active_window

            w = active_window()
            if w.get("class"):
                notes.append(f"User is looking at: {w['class']} - {w.get('title', '')[:120]}.")
        self.detail = bool(DETAIL.search(text))
        if self.detail:
            notes.append("The user wants a detailed answer: answer fully, lists are fine.")
        hints = tool_hints(f"{text} {meaning}")
        if not hints and self.last_tools and re.search(r"\d|percent|pratishat|aur|more|less|kam|zyada|"
                                                      r"badhao|wahi|same|again|phir", text, re.I):
            hints = list(self.last_tools)   # a follow-up to the previous action
        self._turn_hints = hints
        if hints:
            notes.append(f"Relevant tool: {', '.join(hints)}. Call it; do not guess.")
        if is_hindi(text) or language == "hi" or looks_hinglish(text):
            # Speech-to-text writes Hindi as romanized Hinglish; replies use Devanagari so the Hindi voice reads them.
            notes.append("The user spoke Hindi/Hinglish: reply in Hindi using Devanagari script.")
        else:
            notes.append("Reply in English.")
        if notes:
            text += "\n(" + " ".join(notes) + ")"
        return [self.system_message(), *self.history, {"role": "user", "content": text}]

    def system_message(self) -> dict:
        return {"role": "system", "content": SYSTEM.format(today=datetime.now().strftime("%A %d %B %Y"))}

    def _round(self, msgs: list[dict], tool_schemas: list[dict], tool_choice: str, think: bool = False):
        """One streamed model call. Yields visible text; returns (raw content, tool calls).
        If llama-server rejects a malformed tool call (the small model sometimes runs past the end of its
        JSON arguments), the call is retried once with deterministic sampling, then given up gracefully."""
        try:
            return (yield from self._round_once(msgs, tool_schemas, tool_choice, think, self.temperature))
        except Exception as e:  # noqa: BLE001
            if "parse" not in str(e).lower() and "json" not in str(e).lower():
                raise
        try:
            return (yield from self._round_once(msgs, tool_schemas, tool_choice, False, 0.0))
        except Exception as e:  # noqa: BLE001
            if "parse" not in str(e).lower() and "json" not in str(e).lower():
                raise
        msg = "Sorry, I got confused there. Please say that again."
        yield msg
        return msg, {}

    def _round_once(self, msgs: list[dict], tool_schemas: list[dict], tool_choice: str, think: bool,
                    temperature: float):
        """With think=True Gemma reasons first (capped by llama-server's --reasoning-budget); the reasoning
        arrives as reasoning_content and is never shown or spoken."""
        content, calls, in_think = "", {}, False
        stream = self.client.chat.completions.create(
            model="local", messages=msgs, tools=tool_schemas, tool_choice=tool_choice, stream=True,
            temperature=temperature, top_p=self.top_p,
            max_tokens=(self.max_tokens * 3 if self.detail else self.max_tokens) + (self.think_budget + 64 if think else 0),
            extra_body={"top_k": self.top_k, "id_slot": 0, "chat_template_kwargs": {"enable_thinking": think}})
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            for tc in delta.tool_calls or []:
                c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                c["id"] = tc.id or c["id"]
                if tc.function:
                    c["name"] += tc.function.name or ""
                    c["args"] += tc.function.arguments or ""
            piece = delta.content or ""
            if not piece:
                continue
            content += piece
            # hide any <think> block the template might still produce
            if "<think>" in piece:
                in_think = True
            if in_think:
                if "</think>" not in piece:
                    continue
                in_think, piece = False, piece.split("</think>", 1)[1]
            if not calls:
                yield piece
        return content, calls

    def _streamed(self, msgs: list[dict], think: bool, catch_refusal: bool):
        """A normal round, streamed. With catch_refusal the first sentence is held back: if it's "I can't…",
        nothing is yielded and refused=True (solve mode takes over). Returns (content, calls, refused, text)."""
        gen = self._round(msgs, self.tool_schemas, "auto", think)
        held, released, refused, text = "", not catch_refusal, False, ""
        try:
            while True:
                piece = next(gen)
                if refused:
                    continue
                if released:
                    text += piece
                    yield piece
                    continue
                held += piece
                if SENTENCE_END.search(held) or len(held) > 160:
                    if REFUSAL.search(held):
                        refused = True
                    else:
                        released, text = True, text + held
                        yield held
        except StopIteration as stop:
            content, calls = stop.value
        if held and not released and not refused:
            if REFUSAL.search(held):
                refused = True
            else:
                text += held
                yield held
        return content, calls, refused and not calls, text

    def ask_stream(self, text: str, language: str | None = None) -> Iterator[str]:
        """Yields the reply in chunks as the model generates it; runs tools in between.
        `language` is what speech-to-text detected ("hi" for Hindi/Hinglish)."""
        hindi = is_hindi(text) or language == "hi" or looks_hinglish(text)
        meaning = ""
        if self.learn:
            target = memory.correction_target(text)
            if target and self.turns:   # "nahi, browser kholo": remember what the last request meant
                memory.save_correction(self._last_exchange()[0]["content"], target)
                meaning = target
            meaning = meaning or memory.apply_correction(text) or ""
        if not meaning and self.use_normalizer and hindi:
            meaning = self.normalize(text)
        self.last_meaning = meaning
        msgs = self._messages(text, language, meaning)
        hints = self._turn_hints
        groups = hint_groups(f"{text} {meaning}")
        start = len(msgs) - 1
        task = meaning or text
        think = self.thinking and needs_thinking(f"{text} {meaning}", hints)
        if think:
            yield FILLER["think"][hindi] + " "    # spoken at once while the model reasons; not kept in memory
        reply, solving, catch = "", False, self.solve_on_refusal
        rounds, rnd, prev_tools, called, nudged = self.max_rounds, 0, set(), set(), False
        while rnd < rounds:
            refused = False
            # Only push for missing *action* steps; lookups (find, list, read, search) are often not needed.
            missing = [g for g in groups if not called & set(g) and set(g) - LOOKUPS] if called and not nudged else []
            # Hold the answer back on the first round of a tool-ish request, right after how_to (the small
            # model tends to *tell* the user the command instead of running it), and while a step the
            # request asked for hasn't been done yet ("screenshot and copy it": it stops after one).
            if (rnd == 0 and hints) or "how_to" in prev_tools or missing:
                # The request clearly needs a tool. Hold the first answer back: small models often
                # *say* "opening the browser" without calling anything. If no tool was called,
                # ask again with a tool call forced.
                gen = self._round(msgs, self.tool_schemas, "auto")
                pieces = []
                try:
                    while True:
                        pieces.append(next(gen))
                except StopIteration as stop:
                    content, calls = stop.value
                if not calls and missing:
                    nudged = True
                    msgs.append({"role": "user", "content": NOT_DONE + ", ".join(dict.fromkeys(
                        n for g in missing for n in g)) + ")"})
                if not calls:
                    # Same full tool list as always (a different list would invalidate llama.cpp's
                    # cached prompt prefix); the hint note in the message points at the right tool.
                    gen = self._round(msgs, self.tool_schemas, "required")
                    try:
                        while True:
                            next(gen)
                    except StopIteration as stop:
                        content, calls = stop.value
                    if not calls:
                        held = "".join(pieces)
                        if catch and REFUSAL.search(held):
                            content, refused = held, True
                        else:   # really no tool needed: say the held-back answer
                            reply += held
                            yield held
                            break
            else:
                content, calls, refused, said = yield from self._streamed(msgs, think or solving, catch)
                reply += said
                if not calls and not refused:
                    break
            rnd += 1
            if refused:
                # "I can't" -> find a way: recipes, web search, shell. One extra pass, thinking on.
                solving, catch, rounds = True, False, rnd + 4
                msgs.append({"role": "assistant", "content": THINK.sub("", content).strip() or "I can't."})
                msgs.append({"role": "user", "content": SOLVE_NOTE.format(task=task)
                             + (" Reply in Hindi using Devanagari script." if hindi else "")})
                yield FILLER["solve"][hindi] + " "
                continue
            prev_tools = {c["name"] for c in calls.values()}
            called |= prev_tools
            msgs.append({"role": "assistant", "content": THINK.sub("", content) or None, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for i, c in calls.items()]})
            cancelled = False
            for i, c in calls.items():
                result = tools.execute(c["name"], c["args"], self.confirm)
                cancelled = cancelled or result.startswith(("cancelled:", "blocked:"))
                if self.on_tool:
                    try:
                        args = json.loads(c["args"] or "{}")
                    except json.JSONDecodeError:
                        args = {"raw": c["args"]}
                    self.on_tool(c["name"], args, result)
                msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}", "content": result})
            if cancelled:
                # Don't let the model continue: small models sometimes invent the output anyway.
                if any(m["role"] == "tool" and m["content"].startswith("blocked:") for m in msgs[start:]):
                    piece = "यह बहुत ख़तरनाक है, इसलिए मैंने इसे रोक दिया।" if hindi else "That's too dangerous, so I blocked it."
                else:
                    piece = "ठीक है, मैंने रद्द कर दिया।" if hindi else "Okay, I cancelled that."
                piece = (" " if reply else "") + piece
                reply += piece
                yield piece
                break
        else:
            if not reply:
                reply = "Sorry, that took too many steps."
                yield reply
        if not reply.strip() and len(msgs) > start + 1:   # tools ran but nothing was said: confirm briefly
            reply = "हो गया।" if hindi else "Done."
            yield reply
        # Keep the tool calls in memory too: if past turns show answers without tools, a small model
        # learns to skip tools and invent values. Long tool results are shortened.
        # 900 characters keeps a top-5 result list, so "the second one" works on the next turn.
        turn = [dict(m, content=m["content"][:900]) if m["role"] == "tool" else m for m in msgs[start:]
                if not (m["role"] == "user" and m["content"].startswith((SOLVE_NOTE[:30], NOT_DONE)))]
        turn.append({"role": "assistant", "content": THINK.sub("", reply).strip()})
        self.turns.append(turn)
        used = [c["function"]["name"] for m in turn if m["role"] == "assistant" for c in m.get("tool_calls") or []]
        if used:
            self.last_tools = list(dict.fromkeys(used))
        if self.learn:
            self._learn_from(task, turn, solving)
        self._last = time.monotonic()

    def _learn_from(self, task: str, turn: list[dict], solved: bool) -> None:
        """Save what worked: a shell command found in solve mode becomes a recipe; a multi-step
        success becomes a skill. Both are offered by how_to next time."""
        results = {m["tool_call_id"]: m["content"] for m in turn if m["role"] == "tool"}
        steps = []
        for m in turn:
            for c in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                out = results.get(c["id"], "")
                if out.startswith(("error", "cancelled", "blocked")) or "exit code" in out and \
                        not out.startswith("exit code 0"):
                    return   # something failed: don't learn from this turn
                steps.append({"tool": c["function"]["name"], "args": c["function"]["arguments"]})
        if solved:
            for st in steps:
                if st["tool"] == "run_command":
                    try:
                        memory.save_recipe(task, json.loads(st["args"]).get("command", ""))
                    except json.JSONDecodeError:
                        pass
        tools_used = {st["tool"] for st in steps} - {"how_to", "web_search", "recall"}
        if len(tools_used) >= 2:
            memory.save_skill(task, [st for st in steps if st["tool"] in tools_used])

    def ask(self, text: str, language: str | None = None) -> str:
        return "".join(self.ask_stream(text, language)).strip()

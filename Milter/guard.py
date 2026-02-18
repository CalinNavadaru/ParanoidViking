import email
import logging.handlers
import os
import time

import Milter
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "https://phish-nginx/api/analyze-email/")
VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() in ("1", "true", "yes")
MILTER_PORT = int(os.getenv("MILTER_PORT", 10025))
TIMEOUT = int(os.getenv("TIMEOUT", 600))
LOG_FILE = os.getenv("LOG_FILE", "/var/log/milter/milter.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

level = getattr(logging, LOG_LEVEL, logging.INFO)

logger = logging.getLogger("guard")
logger.setLevel(level)

fmt = (
    "%(asctime)s %(levelname)s "
    "MessageID=%(message_id)s "
    "VERDICT=%(verdict)s "
    "URL_SCORE=%(url_score).2f "
    "TEXT_SCORE=%(text_score).2f "
    "LATENCY=%(latency)dms "
    "ACTION=%(action)s"
)
formatter = logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")

handler = logging.handlers.TimedRotatingFileHandler(
    LOG_FILE, when="D", interval=1, backupCount=7, encoding="utf-8"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

console = logging.StreamHandler()
console.setLevel(level)
console.setFormatter(formatter)
logger.addHandler(console)


default_extra = {
    "message_id": "<none>",
    "verdict":    "-",
    "url_score":  0.0,
    "text_score": 0.0,
    "latency":    0,
    "action":     "-",
}

class DefaultExtraFilter(logging.Filter):
    def filter(self, record):
        for key, value in default_extra.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True

handler.addFilter(DefaultExtraFilter())
console.addFilter(DefaultExtraFilter())

class GuardMilter(Milter.Base):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.headers = []

    def header(self, name, value):
        self.headers.append(f"{name}: {value}")
        return Milter.CONTINUE

    @Milter.noreply
    def body(self, chunk):
        text = chunk.decode(errors='ignore') if isinstance(chunk, (bytes, bytearray)) else chunk
        self.chunks.append(text)
        return Milter.CONTINUE

    def eom(self):
        raw = "\r\n".join(self.headers) + "\r\n\r\n" + "".join(self.chunks)
        msg = email.message_from_string(raw)

        subject = msg.get("Subject", "")

        if msg.is_multipart():
            body_parts = []
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(part.get_payload(decode=True).decode(charset, errors="ignore"))
            body = "\n".join(body_parts)
        else:
            charset = msg.get_content_charset() or "utf-8"
            body = msg.get_payload(decode=True).decode(charset, errors="ignore")

        email_content = f"{subject} [SUBJECT] {body}"

        payload = {
            "email_body": email_content
        }

        start = time.time()
        try:
            r = requests.post(
                BACKEND_URL,
                json=payload,
                verify=VERIFY_SSL
            )
            r.raise_for_status()
            data = r.json()

            block_reasons = []
            if data.get("message_analysis", {}).get("label") == 1:
                conf = data["message_analysis"]["confidence"]
                block_reasons.append(f"message(conf={conf:.2f})")
            for u in data.get("url_analysis", []):
                if u.get("label") == 1:
                    block_reasons.append(f"{u['url']}(conf={u['confidence']:.2f})")

            elapsed = time.time() - start
            latency_ms = int(elapsed * 1000)

            should_block = bool(block_reasons)
            verdict = "phishing" if should_block else "clean"
            action = "reject" if should_block else "accept"
            url_score = max((u["confidence"] for u in data.get("url_analysis", [])
                             if u["label"] == 1), default=0.0)
            text_score = data.get("message_analysis", {}).get("confidence", 0.0)

            logger.info(
                "",
                extra={
                    "message_id": msg.get("Message-ID", "<unknown>"),
                    "verdict": verdict,
                    "url_score": url_score,
                    "text_score": text_score,
                    "latency": latency_ms,
                    "action": action,
                }
            )

            return Milter.REJECT if block_reasons else Milter.ACCEPT
        except requests.RequestException as e:
            logger.error("HTTP error contacting backend", exc_info=True)
            return Milter.TEMPFAIL

    def envclose(self):
        self.headers.clear()
        self.chunks.clear()
        return Milter.CONTINUE

    def abort(self):
        return self.envclose()


def main():
    logger.info("Starting Guard.")
    Milter.factory = GuardMilter
    Milter.set_flags(Milter.ADDHDRS | Milter.CHGBODY)
    Milter.runmilter("Guard", f"inet:{MILTER_PORT}@0.0.0.0", TIMEOUT)


if __name__ == "__main__":
    main()

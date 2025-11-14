import subprocess
import shlex
import json
from typing import Optional, Generator

def call_ollama(prompt: str, model: str = "phi3:mini", timeout: int = 20) -> Optional[str]:
    """
    Call Ollama and stream output in real time (non-blocking).
    Returns the full response text.
    """
    try:
        cmd = f"ollama run {shlex.quote(model)} {shlex.quote(prompt)}"
        print(f"[Ollama Command]: {cmd}")
        process = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        full_output = ""
        for line in iter(process.stdout.readline, ''):
            print(line.strip(), end='', flush=True)  # real-time token streaming
            full_output += line

        process.stdout.close()
        process.wait(timeout=timeout)
        return full_output.strip() if full_output.strip() else None

    except Exception as e:
        print(f"⚠️ Ollama call failed: {e}")
        return None


def call_ollama_stream(prompt: str, model: str = "phi3:mini", timeout: int = 20) -> Generator[str, None, None]:
    """
    Generator that yields streaming text from Ollama.
    Example usage:
        for token in call_ollama_stream("hello", "phi3:mini"):
            print(token, end="", flush=True)
    """
    try:
        cmd = f"ollama run {shlex.quote(model)} {shlex.quote(prompt)}"
        process = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            yield line
        process.stdout.close()
        process.wait(timeout=timeout)
    except Exception as e:
        yield f"[Streaming error: {e}]"


def call_subprocess_local_model(prompt: str, cmd_template: str = None, timeout: int = 12) -> Optional[str]:
    """
    Generic local fallback, runs a command-line model or script.
    Example: cmd_template='python local_model.py "{prompt}"'
    """
    if not cmd_template:
        return None
    try:
        cmd = cmd_template.format(prompt=prompt)
        res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception as e:
        print(f"⚠️ Local model call failed: {e}")
    return None

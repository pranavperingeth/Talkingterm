from translator import translate
import subprocess


# ── Blocked dangerous commands ─────────────────────────────────────────────────
BLOCKED = [
    "rm -rf", "rmdir", "del /f", "format", "mkfs",
    "dd if=", ":(){:|:&};:", "shutdown", "reboot",
    "reg delete", "rd /s", "powershell -enc",
]

def is_dangerous(command: str) -> bool:
    cmd_lower = command.lower()
    return any(b in cmd_lower for b in BLOCKED)


def run_command(command: str) -> None:
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=15,        # kill if it hangs
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print("[stderr]", result.stderr, end="")
        if result.returncode != 0:
            print(f"[exited with code {result.returncode}]")
    except subprocess.TimeoutExpired:
        print("[error] Command timed out after 15 seconds.")
    except Exception as e:
        print(f"[error] {e}")


# ── Main REPL ──────────────────────────────────────────────────────────────────
print("TalkingTerm  —  type 'exit' to quit\n")

while True:
    try:
        query = input("tt> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break

    if not query:
        continue

    if query.lower() == "exit":
        print("Bye!")
        break

    # Translate natural language → shell command
    command = translate(query)

    if not command.strip():
        print("[error] Could not translate that. Try rephrasing.\n")
        continue

    print(f"  Suggested: {command}")

    # Safety check before asking user
    if is_dangerous(command):
        print("  [blocked] This command is potentially destructive and won't be run.\n")
        continue

    try:
        confirm = input("  Execute? (y/n): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
        break

    if confirm == "y":
        run_command(command)
    else:
        print("  Skipped.")

    print()
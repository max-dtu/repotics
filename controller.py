import urllib.request
import sys

# Change this to match your robot's ID
ROBOT_UNIQUE_ID = "kals-ev3"
URL = "https://ntfy.sh/{}".format(ROBOT_UNIQUE_ID)

def send_command(cmd: str):
    try:
        req = urllib.request.Request(
            URL,
            data=cmd.encode("utf-8"),
            headers={"User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
        print("  -> Sent: {}".format(cmd))
    except Exception as e:
        print("  -> Error sending command {}: {}".format(cmd, e))

def main():
    print("=" * 50)
    print("  EV3 Internet Controller (Robot ID: {})".format(ROBOT_UNIQUE_ID))
    print("=" * 50)
    print("Quick Controls:")
    print("  w — forward")
    print("  s — backward")
    print("  a — left")
    print("  d — right")
    print("  o — open gripper")
    print("  c — close gripper")
    print("\n* You can also type raw commands directly.")
    print("* Type 'exit' to quit.\n")

    KEY_MAP = {
        'w': 'forward',
        's': 'backward',
        'a': 'left',
        'd': 'right',
        'o': 'open_gripper',
        'c': 'close_gripper'
    }

    while True:
        try:
            val = input("Enter key/command > ").strip().lower()
            if not val:
                continue
            if val == "exit":
                break
            
            # Match shortcut key, or send the typed text command directly
            cmd = KEY_MAP.get(val, val)
            send_command(cmd)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()

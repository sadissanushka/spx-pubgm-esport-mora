import os
import sys
import cv2
import numpy as np
import mss
import pygetwindow as gw

lib_path = r'C:\Users\ASUS-TUF\lib'
if os.path.exists(lib_path):
    sys.path.insert(0, lib_path)

EMULATOR_TITLES = ["Gameloop", "PUBG MOBILE", "BlueStacks", "LDPlayer"]


def run_selector():
    win = None
    for title in EMULATOR_TITLES:
        wins = gw.getWindowsWithTitle(title)
        if wins:
            win = wins[0]
            break

    if not win:
        print("Error: No game/emulator window found.")
        return

    print(f"Window: '{win.title}'  {win.width}x{win.height}  @ ({win.left},{win.top})")

    region = {"top": win.top, "left": win.left, "width": win.width, "height": win.height}

    with mss.mss() as sct:
        shot  = sct.grab(region)
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)

    print("\nDrag to select the TIMER area, then press ENTER/SPACE. Press 'c' to cancel.\n")

    x, y, w, h = cv2.selectROI("Select Timer Region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("Selection cancelled.")
        return

    top_pct    = y / win.height
    left_pct   = x / win.width
    width_pct  = w / win.width
    height_pct = h / win.height

    snippet = (
        f'self.region = {{\n'
        f'    "top":    win.top  + int(win.height * {top_pct:.4f}),\n'
        f'    "left":   win.left + int(win.width  * {left_pct:.4f}),\n'
        f'    "width":  int(win.width  * {width_pct:.4f}),\n'
        f'    "height": int(win.height * {height_pct:.4f}),\n'
        f'}}'
    )

    print(f"\n{'='*40}")
    print(f"  top={top_pct:.4f}  left={left_pct:.4f}  w={width_pct:.4f}  h={height_pct:.4f}")
    print(f"  Pixel coords: x={x} y={y} w={w} h={h}")
    print(f"{'='*40}")
    print("\nPaste into tracker.py → find_window():\n")
    print(snippet)


if __name__ == "__main__":
    run_selector()

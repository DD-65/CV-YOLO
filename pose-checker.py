import argparse
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Confidence cutoff for the keypoints we care about
CONF_TH = 0.25
# Smoothing strength for noisy frame-to-frame pose changes
EMA_ALPHA = 0.20
# First few seconds are used to learn your "upright" baseline
CALIB_SECS = 4.0
# If head height drops below this part of baseline, treat as slouching
SLOUCH_RATIO = 0.78
# Don't warn instantly; only warn if bad posture lasts this long
WARN_AFTER_SECS = 1.5

# Light rhythmic timing for alert sounds.
CALIB_LOW_TIMES = [0.00, CALIB_SECS / 3.0, 2.0 * CALIB_SECS / 3.0]
WARN_BEEP_SECS = 0.60


def parse_args():
    # CLI flags for quiet, GUI, or popup-only mode
    parser = argparse.ArgumentParser(description="Posture checker")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-g", "--gui", action="store_true", help="show main camera GUI window")
    mode.add_argument(
        "-p",
        "--popup",
        action="store_true",
        help="show popup warning window when posture is bad (cannot be used with --gui)",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="disable all sounds (calibration + warning beeps)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto, cpu, mps, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def resolve_device(requested):
    req = requested.lower()

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        print("CUDA requested, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("MPS requested, but MPS is unavailable. Falling back to CPU.")
        return "cpu"
    if req == "cpu":
        return "cpu"
    return requested


class SoundPlayer:
    def __init__(self, silent=False):
        self.silent = silent
        # On macOS we can play nicer system sounds through afplay
        self.is_macos = sys.platform == "darwin" and shutil.which("afplay") is not None
        self.low_sound = "/System/Library/Sounds/Pop.aiff"
        self.high_sound = "/System/Library/Sounds/Glass.aiff"
        self.warn_sound = "/System/Library/Sounds/Ping.aiff"
        self.mode = "off"
        self.calib_t0 = 0.0
        self.calib_low_idx = 0
        self.next_warn = 0.0

    def _play(self, tone):
        if self.silent:
            return
        if self.is_macos:
            # Pick a sound + volume based on event type
            if tone == "warn":
                sound_file = self.warn_sound
                vol = "0.24"
            elif tone == "high":
                sound_file = self.high_sound
                vol = "0.25"
            else:
                sound_file = self.low_sound
                vol = "0.20"
            subprocess.Popen(
                ["afplay", "-v", vol, sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Fallback bell character for non-mac systems
            print("\a", end="", flush=True)

    def set_mode(self, mode, now):
        # Ignore mode changes if we're already in that state
        if mode == self.mode:
            return
        self.mode = mode
        if mode == "calibrating":
            self.calib_t0 = now
            self.calib_low_idx = 0
        elif mode == "warning":
            self.next_warn = now

    def play_calibration_done(self):
        self._play("high")

    def play_posture_ok(self):
        self._play("high")

    def update(self, now):
        if self.silent or self.mode == "off":
            return
        if self.mode == "warning":
            if now >= self.next_warn:
                # Keep warning beeps steady even if frame timing jitters (Doesnt work :( )
                self._play("warn")
                self.next_warn = now + WARN_BEEP_SECS
            return
        if self.mode == "calibrating":
            elapsed = now - self.calib_t0
            if self.calib_low_idx < len(CALIB_LOW_TIMES) and elapsed >= CALIB_LOW_TIMES[self.calib_low_idx]:
                self._play("low")
                self.calib_low_idx += 1


class PopupWarning:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.available = False
        self.visible = False
        self.root = None
        self.win = None
        self.tk = None
        if not enabled:
            return
        try:
            # Import lazily so popup is optional and doesn't break CLI-only use
            import tkinter as tk

            self.tk = tk
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.attributes("-topmost", True)

            self.win = tk.Toplevel(self.root)
            self.win.withdraw()
            self.win.title("Posture Warning")
            self.win.configure(bg="#b00020")
            self.win.attributes("-topmost", True)
            self.win.geometry("380x120+60+60")
            self.win.resizable(False, False)

            label = tk.Label(
                self.win,
                text="WARNING: Sit upright",
                fg="white",
                bg="#b00020",
                font=("Helvetica", 19, "bold"),
            )
            label.pack(expand=True, fill="both", padx=8, pady=8)
            self.available = True
        except Exception as exc:
            print(f"Popup disabled: {exc}")
            self.enabled = False

    def show(self):
        if not self.available or self.visible:
            return
        self.win.deiconify()
        self.win.lift()
        self.visible = True

    def hide(self):
        if not self.available or not self.visible:
            return
        self.win.withdraw()
        self.visible = False

    def pump(self):
        if not self.available:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.available = False

    def close(self):
        if not self.available:
            return
        try:
            self.win.destroy()
            self.root.destroy()
        except self.tk.TclError:
            pass
        self.available = False


def main():
    args = parse_args()
    device = resolve_device(args.device)
    # Load the pose model and webcam stream
    model = YOLO("yolo26s-pose.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    sound = SoundPlayer(silent=args.silent)
    popup = PopupWarning(enabled=args.popup)

    # Smoothed pose features and baseline stats
    ema_head = None
    ema_forward = None
    baseline_head = None
    baseline_forward = None
    calib_head = []
    calib_forward = []
    slouch_since = None
    # Helps detect the "bad -> good" transition to use for confirmation sound
    last_posture_bad = False
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            r = model(frame, verbose=False, device=device)[0]
            vis = r.plot()
            msg = "No person detected"
            color = (255, 255, 255)
            has_pose = False
            posture_bad = False
            posture_ok = False

            if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
                # If there are multiple people, track the largest (-> closest) one in frame
                boxes = r.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                i = int(np.argmax(areas))

                kxy = r.keypoints.xy[i].cpu().numpy()
                if r.keypoints.conf is not None:
                    kcf = r.keypoints.conf[i].cpu().numpy()
                else:
                    kcf = np.ones(17, dtype=np.float32)

                # COCO keypoints needed: nose + both shoulders
                need = [0, 5, 6]
                if np.all(kcf[need] > CONF_TH):
                    has_pose = True
                    nose = kxy[0]
                    l_sh, r_sh = kxy[5], kxy[6]

                    neck = (l_sh + r_sh) / 2.0
                    # Tiny epsilon avoids divide-by-zero on weird detections
                    shoulder_w = np.linalg.norm(l_sh - r_sh) + 1e-6

                    # Bigger value = head is higher above shoulders
                    head_height = (neck[1] - nose[1]) / shoulder_w
                    # Bigger value = head drifting away from shoulder center
                    forward_offset = abs(nose[0] - neck[0]) / shoulder_w

                    if ema_head is None:
                        ema_head = head_height
                        ema_forward = forward_offset
                    else:
                        ema_head = (1 - EMA_ALPHA) * ema_head + EMA_ALPHA * head_height
                        ema_forward = (1 - EMA_ALPHA) * ema_forward + EMA_ALPHA * forward_offset

                    now = time.time()
                    if baseline_head is None:
                        # During calibration, store stable smoothed values
                        calib_head.append(ema_head)
                        calib_forward.append(ema_forward)
                        msg = "Calibrating... sit upright"
                        color = (0, 255, 255)
                        # Need both enough time and enough samples.
                        if (now - t0) >= CALIB_SECS and len(calib_head) >= 15:
                            baseline_head = float(np.median(calib_head))
                            baseline_forward = float(np.median(calib_forward))
                            sound.play_calibration_done()
                    else:
                        # Slouch if head drops too much or drifts too far forward
                        slouch = (
                            ema_head < (SLOUCH_RATIO * baseline_head)
                            or ema_forward > (baseline_forward + 0.22)
                        )
                        if slouch:
                            if slouch_since is None:
                                # Start a timer so short dips don't trigger warnings
                                slouch_since = now
                        else:
                            slouch_since = None

                        if slouch_since is not None and (now - slouch_since) >= WARN_AFTER_SECS:
                            msg = "Warning: Sit upright"
                            color = (0, 0, 255)
                            posture_bad = True
                        else:
                            msg = "Posture OK"
                            color = (0, 200, 0)
                            posture_ok = True
                else:
                    msg = "Pose unclear (need nose + both shoulders)"
                    color = (0, 165, 255)

            now = time.time()
            # Play a quick "ok again" tone when posture recovers
            if last_posture_bad and posture_ok:
                sound.play_posture_ok()
            last_posture_bad = posture_bad

            # Keep sound mode aligned with current posture state
            if baseline_head is None and has_pose:
                sound.set_mode("calibrating", now)
            elif posture_bad:
                sound.set_mode("warning", now)
            else:
                sound.set_mode("off", now)
            sound.update(now)

            if args.popup:
                # Popup should only stay visible while warning is active
                if posture_bad:
                    popup.show()
                else:
                    popup.hide()
                popup.pump()

            if args.gui:
                cv2.putText(
                    vis,
                    msg,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,  # bumped text size a bit so status is easier to read
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Posture Coach", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        popup.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

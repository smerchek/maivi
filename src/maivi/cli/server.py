"""
Streaming STT Server with real-time transcription.
Processes audio chunks as they're recorded using a sliding window.
"""
import os
import time
import threading
import subprocess
import shutil
from pathlib import Path

import nemo.collections.asr as nemo_asr
import pyperclip
import soundfile as sf
from pynput import keyboard
from pynput.keyboard import Key

from maivi.core.streaming_recorder import StreamingRecorder
from maivi.core.chunk_merger import SimpleChunkMerger  # New simple merger
from maivi.cli.terminal_ui import create_streaming_ui
from maivi.core.pause_detector import PauseDetector

# Cross-platform notifications
try:
    from plyer import notification
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False


class StreamingSTTServer:
    def __init__(
        self,
        auto_paste=False,
        window_seconds=7.0,  # Chunk size (more context = better quality)
        slide_seconds=3.0,   # Slide interval (larger overlap = better merging)
        start_delay_seconds=2.0,  # Start processing after this delay
        speed=1.0,
        toggle_mode=False,
        output_file=None,
        show_ui=False,
        ui_width=30,
        pause_paragraph_breaks=True,
        pause_threshold_seconds=1.0,
    ):
        self.auto_paste = auto_paste
        self.speed = speed
        self.toggle_mode = toggle_mode
        self.output_file = output_file
        self.model = None

        # Simple fixed overlapping chunks
        self.recorder = StreamingRecorder(
            window_seconds=window_seconds,
            slide_seconds=slide_seconds,
            start_delay_seconds=start_delay_seconds,
            speed=speed,
        )
        self.is_shutting_down = False

        # Detect Wayland and choose appropriate paste method
        self._detect_paste_method()

        # Track which keys are pressed for hotkey detection
        self.current_keys = set()
        self.hotkey_pressed = False
        self.is_recording = False  # For toggle mode

        # Transcription state
        self.transcription_thread = None
        self.chunk_counter = 0
        self.chunk_merger = SimpleChunkMerger()  # Simple overlap-based merging
        self.is_transcribing = False

        # Output file handle
        self.output_stream = None
        if self.output_file:
            self.output_stream = open(self.output_file, 'w', buffering=1)  # Line buffered

        # Streaming UI
        self.show_ui = show_ui
        self.streaming_ui = None
        if self.show_ui:
            self.streaming_ui = create_streaming_ui(width_chars=ui_width, prefer_gui=True)

        # Pause detection for paragraph breaks
        self.pause_paragraph_breaks = pause_paragraph_breaks
        self.pause_detector = PauseDetector(
            silence_threshold_db=-40.0,
            min_pause_duration=pause_threshold_seconds,
            sample_rate=16000
        )
        self.last_chunk_audio = None  # Store last chunk audio for pause detection
        self.recording_start_time = None  # Track when recording started

    def _detect_paste_method(self):
        """Detect display server and available paste tools."""
        self.paste_method = None

        # Check if we're on Wayland
        wayland_display = os.environ.get('WAYLAND_DISPLAY')
        xdg_session_type = os.environ.get('XDG_SESSION_TYPE', '').lower()
        is_wayland = wayland_display or xdg_session_type == 'wayland'

        if is_wayland:
            # On Wayland, try wtype first, then ydotool
            if shutil.which('wtype'):
                self.paste_method = 'wtype'
            elif shutil.which('ydotool'):
                self.paste_method = 'ydotool'
            else:
                self.paste_method = None
                if self.auto_paste:
                    print("⚠️  Warning: Auto-paste not available on Wayland")
                    print("    Install wtype: sudo pacman -S wtype")
        else:
            # On X11, use pynput
            self.paste_method = 'pynput'

    def _auto_paste_text(self):
        """Paste text using the detected method."""
        if not self.auto_paste:
            return

        time.sleep(0.2)  # Small delay

        try:
            if self.paste_method == 'wtype':
                # Wayland with wtype
                subprocess.run(['wtype', '-M', 'ctrl', '-M', 'shift', 'v'], check=False)
            elif self.paste_method == 'ydotool':
                # Wayland with ydotool
                subprocess.run(['ydotool', 'key', '29:1', '42:1', '47:1', '47:0', '42:0', '29:0'], check=False)
            elif self.paste_method == 'pynput':
                # X11 with pynput
                from pynput.keyboard import Controller
                kb = Controller()
                kb.press(Key.ctrl)
                kb.press(Key.shift)
                kb.press('v')
                kb.release('v')
                kb.release(Key.shift)
                kb.release(Key.ctrl)
            else:
                print("⚠️  Auto-paste not available")
                return

            print(f"✓ Auto-pasted")

        except Exception as e:
            print(f"⚠️  Auto-paste failed: {e}")

    def _show_notification(self, title: str, message: str, timeout: int = 2):
        """Show cross-platform notification (non-blocking)."""
        if not NOTIFICATIONS_AVAILABLE:
            return

        try:
            notification.notify(
                title=title,
                message=message,
                app_name='STT Server',
                timeout=timeout  # seconds
            )
        except Exception as e:
            # Silently fail if notifications don't work
            pass

    def load_model(self):
        """Load the STT model."""
        print("Loading Parakeet TDT 0.6B v3 model...")
        print("This may take a few minutes on first run (downloading model)...")

        # Force CPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        start_time = time.time()
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        self.model = self.model.cpu()
        self.model.eval()

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds\n")

    def transcribe_chunk(self, chunk_np, chunk_id, is_last_chunk=False):
        """Transcribe a single audio chunk and merge with existing result."""
        try:
            # Save chunk to file
            chunk_file = self.recorder.save_chunk_to_file(chunk_np, chunk_id)
            if not chunk_file:
                return None

            # Transcribe
            start_time = time.time()
            output = self.model.transcribe([chunk_file], timestamps=False)
            text = output[0].text.strip()
            transcribe_time = time.time() - start_time

            if text:
                # Merge with existing result using overlap detection
                merged = self.chunk_merger.add_chunk(text, is_final=is_last_chunk)

                # Show progress
                marker = "🏁" if is_last_chunk else "⚡"
                print(f"  {marker} Chunk {chunk_id}: {text[:50]}...")
                print(f"     Merged ({len(merged.split())} words): ...{merged[-80:]}")

                # Stream to file if enabled
                if self.output_stream:
                    self.output_stream.write(f"{merged}\n")
                    self.output_stream.flush()

                # Update streaming UI if enabled
                if self.streaming_ui:
                    self.streaming_ui.update_text(merged)

                return text
            return None

        except Exception as e:
            print(f"Error transcribing chunk {chunk_id}: {e}")
            return None

    def streaming_transcription_loop(self):
        """
        Process audio chunks with simple fixed overlapping windows.

        Strategy:
        - Fixed 7s windows sliding every 3s (4s overlap)
        - Merge using overlap detection (simple and reliable)
        - Process chunks as they arrive during recording
        """
        print("🔄 Streaming processor started")
        print(f"   Window: {self.recorder.window_seconds}s, Slide: {self.recorder.slide_seconds}s")
        print(f"   Overlap: {self.recorder.window_seconds - self.recorder.slide_seconds}s")

        self.chunk_counter = 0
        self.chunk_merger.reset()

        while self.is_transcribing or not self.recorder.processing_queue.empty():
            chunk_np = self.recorder.get_next_chunk()

            if chunk_np is not None:
                self.chunk_counter += 1
                is_last = not self.is_transcribing and self.recorder.processing_queue.empty()

                # Transcribe this chunk
                self.transcribe_chunk(chunk_np, self.chunk_counter, is_last_chunk=is_last)
            else:
                # No chunk available yet
                time.sleep(0.1)

        print("🔄 Streaming processor stopped")

    def _normalize_text(self, text):
        """Normalize text (same as chunk_merger)."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def finalize_transcription(self):
        """Finalize and output the complete transcription."""
        # Get the smartly merged result
        final_text = self.chunk_merger.get_result()

        if not final_text:
            print("No text transcribed")
            self._show_notification(
                "STT Server",
                "No text transcribed",
                timeout=2
            )
            return

        print(f"\n{'=' * 60}")
        print(f"📝 Final Transcription:")
        print(f"{'=' * 60}")
        print(final_text)
        print(f"{'=' * 60}\n")

        # Copy to clipboard
        pyperclip.copy(final_text)
        print(f"✓ Copied to clipboard")

        # Show notification IMMEDIATELY - don't wait!
        preview = final_text[:50] + "..." if len(final_text) > 50 else final_text
        self._show_notification(
            "Copied to clipboard!",
            preview,
            timeout=2
        )

        # Small delay to ensure notification shows before auto-paste
        time.sleep(0.1)

        # Auto-paste if enabled
        self._auto_paste_text()

        print()

    def on_press(self, key):
        """Handle key press events."""
        if self.is_shutting_down:
            return False

        # Track pressed keys
        try:
            self.current_keys.add(key)
        except:
            pass

        # Check for Alt+Q combination (simple and uncommon)
        alt_pressed = Key.alt_l in self.current_keys or Key.alt in self.current_keys or Key.alt_r in self.current_keys

        # Check for 'q' key (or 'œ' on macOS when Option+Q is pressed)
        try:
            q_pressed = (keyboard.KeyCode.from_char('q') in self.current_keys or
                        keyboard.KeyCode.from_char('œ') in self.current_keys)
        except:
            q_pressed = False

        hotkey_combo = alt_pressed and q_pressed

        if self.toggle_mode:
            # Toggle mode: press once to start, press again to stop
            if hotkey_combo and not self.hotkey_pressed:
                self.hotkey_pressed = True  # Debounce

                if not self.is_recording:
                    # Start recording
                    print("🔴 Recording started (press again to stop)")
                    self.is_recording = True
                    self.chunk_counter = 0
                    self.recording_start_time = time.time()  # Track start time for hybrid mode

                    # Start streaming UI if enabled
                    if self.streaming_ui:
                        self.streaming_ui.start()

                    self.recorder.start_recording()
                    self.is_transcribing = True
                    self.transcription_thread = threading.Thread(
                        target=self.streaming_transcription_loop
                    )
                    self.transcription_thread.start()
                else:
                    # Stop recording
                    self._stop_recording()
        else:
            # Hold mode: hold to record, release to stop
            if hotkey_combo:
                if not self.hotkey_pressed and not self.recorder.is_recording:
                    self.hotkey_pressed = True
                    self.chunk_counter = 0
                    self.recording_start_time = time.time()  # Track start time for hybrid mode

                    # Start streaming UI if enabled
                    if self.streaming_ui:
                        self.streaming_ui.start()

                    # Start recording
                    self.recorder.start_recording()

                    # Start transcription loop
                    self.is_transcribing = True
                    self.transcription_thread = threading.Thread(
                        target=self.streaming_transcription_loop
                    )
                    self.transcription_thread.start()

    def _stop_recording(self):
        """Internal method to stop recording and complete transcription."""
        print("\n🛑 Recording stopped, processing audio...")
        self.is_recording = False

        # Stop microphone input
        audio_file = self.recorder.stop_recording()

        # Get recording duration
        try:
            data, samplerate = sf.read(audio_file)
            duration = len(data) / samplerate
        except:
            duration = 0

        # Check if recording was too short for streaming
        if duration < self.recorder.start_delay_seconds:
            print(f"⚡ Short recording ({duration:.1f}s) - processing entire clip...")

            # Stop any streaming that might have started
            self.is_transcribing = False
            if self.transcription_thread:
                self.transcription_thread.join(timeout=2.0)

            # Transcribe the whole recording at once
            try:
                output = self.model.transcribe([audio_file], timestamps=False)
                text = output[0].text.strip()

                if text:
                    print(f"\n📝 Transcribed: {text}\n")

                    # Copy to clipboard immediately
                    pyperclip.copy(text)
                    print(f"✓ Copied to clipboard")

                    # Show notification RIGHT AWAY
                    preview = text[:50] + "..." if len(text) > 50 else text
                    self._show_notification(
                        "Copied to clipboard!",
                        preview,
                        timeout=2
                    )

                    # Auto-paste if enabled
                    self._auto_paste_text()
                else:
                    print("No text transcribed")
                    self._show_notification("STT Server", "No text transcribed", timeout=2)
            except Exception as e:
                print(f"Error transcribing: {e}")

        else:
            # Normal streaming mode - process buffered chunks
            # Show notification that we're processing
            self._show_notification(
                "STT Server",
                "Processing transcription...",
                timeout=3
            )

            # Signal transcription thread to finish processing queue
            self.is_transcribing = False

            # Wait for transcription thread to process ALL remaining chunks
            if self.transcription_thread:
                print("⏳ Processing remaining chunks...")
                self.transcription_thread.join(timeout=30.0)

            # Finalize and copy to clipboard
            self.finalize_transcription()

    def on_release(self, key):
        """Handle key release events."""
        if self.is_shutting_down:
            return False

        # Check for Esc to exit
        if key == Key.esc:
            print("\n👋 Shutting down...")
            self.is_shutting_down = True
            if self.output_stream:
                self.output_stream.close()
            return False

        # Track released keys
        try:
            if key in self.current_keys:
                self.current_keys.remove(key)
        except:
            pass

        # Check if hotkey is released
        alt_pressed = Key.alt_l in self.current_keys or Key.alt in self.current_keys or Key.alt_r in self.current_keys

        try:
            q_pressed = (keyboard.KeyCode.from_char('q') in self.current_keys or
                        keyboard.KeyCode.from_char('œ') in self.current_keys)
        except:
            q_pressed = False

        hotkey_combo = alt_pressed and q_pressed

        # Reset debounce when keys released (for toggle mode)
        if self.toggle_mode and not hotkey_combo:
            self.hotkey_pressed = False

        # Hold mode: stop recording when hotkey released
        if not self.toggle_mode:
            if self.hotkey_pressed and not hotkey_combo:
                self.hotkey_pressed = False

                if self.recorder.is_recording:
                    self._stop_recording()

    def run(self):
        """Start the streaming STT server."""
        print("=" * 60)
        print("Streaming STT Server - Real-time Voice to Text")
        print("=" * 60)
        print(f"Auto-paste: {'Enabled' if self.auto_paste else 'Disabled'}")
        print(f"Speed: {self.speed}x")
        print(f"Chunk size: {self.recorder.window_seconds}s (context window)")
        print(f"Slide interval: {self.recorder.slide_seconds}s")
        print(f"Overlap: {self.recorder.window_seconds - self.recorder.slide_seconds}s (for merging)")
        print(f"Start delay: {self.recorder.start_delay_seconds}s")
        if self.toggle_mode:
            print(f"Hotkey: Alt+Q (Option+Q on macOS) - press once to start, again to stop")
        else:
            print(f"Hotkey: Alt+Q (Option+Q on macOS) - hold to record")
        if self.output_file:
            print(f"Output file: {self.output_file} (streaming)")
        print(f"Exit: Press Esc")
        print("=" * 60)

        # Load model
        self.load_model()

        # Start keyboard listener
        try:
            if self.toggle_mode:
                print(f"\n✓ Ready! Press Alt+Q (Option+Q on macOS) once to start recording.")
            else:
                print(f"\n✓ Ready! Hold Alt+Q (Option+Q on macOS) to start recording.")
            print(f"  Streaming will start after {self.recorder.start_delay_seconds}s\n")

            with keyboard.Listener(
                on_press=self.on_press, on_release=self.on_release
            ) as listener:
                listener.join()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            self.recorder.cleanup()
            if self.output_stream:
                self.output_stream.close()
            if self.streaming_ui:
                self.streaming_ui.stop()
            print("✓ Cleaned up")

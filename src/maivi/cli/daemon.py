"""
Daemon mode for maivi - keeps model loaded and responds to IPC commands.
"""
import os
import json
import socket
import threading
import signal
import sys
from pathlib import Path

from maivi.cli.server import StreamingSTTServer


class MaiviDaemon:
    """Daemon that keeps STT model loaded and responds to IPC commands."""

    def __init__(
        self,
        auto_paste=False,
        window_seconds=7.0,
        slide_seconds=3.0,
        start_delay_seconds=2.0,
        speed=1.0,
        output_file=None,
        show_ui=False,
        ui_width=30,
        pause_paragraph_breaks=True,
        pause_threshold_seconds=1.0,
    ):
        # Get socket path from XDG_RUNTIME_DIR or fallback
        runtime_dir = os.environ.get('XDG_RUNTIME_DIR', f'/tmp/maivi-{os.getuid()}')
        os.makedirs(runtime_dir, exist_ok=True)
        self.socket_path = Path(runtime_dir) / 'maivi.sock'

        # Remove stale socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Create STT server (toggle mode, no keyboard listener)
        self.server = StreamingSTTServer(
            auto_paste=auto_paste,
            window_seconds=window_seconds,
            slide_seconds=slide_seconds,
            start_delay_seconds=start_delay_seconds,
            speed=speed,
            toggle_mode=True,  # Always use toggle mode for daemon
            output_file=output_file,
            show_ui=show_ui,
            ui_width=ui_width,
            pause_paragraph_breaks=pause_paragraph_breaks,
            pause_threshold_seconds=pause_threshold_seconds,
        )

        self.socket = None
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📡 Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def _get_status(self):
        """Get current recording status."""
        if self.server.is_recording:
            return "recording"
        elif self.server.is_transcribing:
            return "processing"
        else:
            return "idle"

    def _handle_command(self, command_data):
        """Handle incoming command from client."""
        try:
            command = command_data.get('command')

            if command == 'toggle':
                if self.server.is_recording:
                    # Stop recording
                    self.server._stop_recording()
                    return {'status': 'stopped', 'message': 'Recording stopped'}
                else:
                    # Start recording
                    print("🔴 Recording started (via IPC)")
                    self.server.is_recording = True
                    self.server.chunk_counter = 0
                    self.server.recording_start_time = __import__('time').time()

                    # Start streaming UI if enabled
                    if self.server.streaming_ui:
                        self.server.streaming_ui.start()

                    self.server.recorder.start_recording()
                    self.server.is_transcribing = True
                    self.server.transcription_thread = threading.Thread(
                        target=self.server.streaming_transcription_loop
                    )
                    self.server.transcription_thread.start()

                    return {'status': 'recording', 'message': 'Recording started'}

            elif command == 'status':
                status = self._get_status()
                return {'status': status}

            elif command == 'stop':
                self.running = False
                return {'status': 'shutting_down', 'message': 'Daemon shutting down'}

            else:
                return {'error': f'Unknown command: {command}'}

        except Exception as e:
            return {'error': str(e)}

    def _handle_client(self, client_socket):
        """Handle individual client connection."""
        try:
            # Receive command (max 1KB)
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                return

            command_data = json.loads(data)
            response = self._handle_command(command_data)

            # Send response
            client_socket.send(json.dumps(response).encode('utf-8'))

        except Exception as e:
            error_response = {'error': str(e)}
            try:
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass

        finally:
            client_socket.close()

    def run(self, detach=False):
        """Start the daemon."""
        if detach:
            # Fork to background
            pid = os.fork()
            if pid > 0:
                # Parent process
                print(f"✓ Daemon started with PID {pid}")
                print(f"✓ Socket: {self.socket_path}")
                print(f"✓ Use 'maivi-cli toggle' to start recording")
                sys.exit(0)

            # Child process - detach from terminal
            os.setsid()
            os.chdir('/')

            # Redirect stdout/stderr to log file
            log_dir = Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / 'maivi'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / 'daemon.log'

            sys.stdout = open(log_file, 'a')
            sys.stderr = open(log_file, 'a')

        print("=" * 60)
        print("Maivi Daemon - Voice to Text")
        print("=" * 60)
        print(f"Socket: {self.socket_path}")
        print(f"Auto-paste: {'Enabled' if self.server.auto_paste else 'Disabled'}")
        print(f"Speed: {self.server.speed}x")
        print("=" * 60)

        # Load model
        print("\n📦 Loading model...")
        self.server.load_model()
        print("✓ Model loaded, daemon ready\n")

        # Create Unix socket
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(str(self.socket_path))
        self.socket.listen(5)
        self.socket.settimeout(1.0)  # Timeout for clean shutdown

        # Set socket permissions (user only)
        os.chmod(self.socket_path, 0o600)

        print(f"📡 Listening on {self.socket_path}")
        print("✓ Ready for commands (use 'maivi-cli toggle' to start)\n")

        self.running = True

        try:
            while self.running:
                try:
                    client_socket, _ = self.socket.accept()
                    # Handle client in separate thread for responsiveness
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()

                except socket.timeout:
                    # Normal timeout, continue loop
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")

        finally:
            self.stop()

    def stop(self):
        """Stop the daemon."""
        print("\n🛑 Stopping daemon...")
        self.running = False

        # Stop recording if active
        if self.server.is_recording:
            try:
                self.server._stop_recording()
            except:
                pass

        # Cleanup
        if self.socket:
            self.socket.close()

        if self.socket_path.exists():
            self.socket_path.unlink()

        self.server.recorder.cleanup()
        if self.server.output_stream:
            self.server.output_stream.close()
        if self.server.streaming_ui:
            self.server.streaming_ui.stop()

        print("✓ Daemon stopped")

"""
IPC client for communicating with maivi daemon.
"""
import os
import json
import socket
import sys
from pathlib import Path


class MaiviClient:
    """Client for sending commands to maivi daemon."""

    def __init__(self):
        # Get socket path from XDG_RUNTIME_DIR or fallback
        runtime_dir = os.environ.get('XDG_RUNTIME_DIR', f'/tmp/maivi-{os.getuid()}')
        self.socket_path = Path(runtime_dir) / 'maivi.sock'

    def _send_command(self, command):
        """Send command to daemon and return response."""
        if not self.socket_path.exists():
            return {
                'error': 'Daemon not running',
                'help': 'Start daemon with: maivi-cli daemon'
            }

        try:
            # Connect to daemon
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            client_socket.connect(str(self.socket_path))

            # Send command
            command_json = json.dumps({'command': command})
            client_socket.send(command_json.encode('utf-8'))

            # Receive response
            response_data = client_socket.recv(4096).decode('utf-8')
            response = json.loads(response_data)

            client_socket.close()
            return response

        except socket.timeout:
            return {'error': 'Daemon not responding (timeout)'}
        except ConnectionRefusedError:
            return {'error': 'Cannot connect to daemon'}
        except Exception as e:
            return {'error': f'Communication error: {str(e)}'}

    def toggle(self):
        """Toggle recording on/off."""
        response = self._send_command('toggle')

        if 'error' in response:
            print(f"❌ Error: {response['error']}", file=sys.stderr)
            if 'help' in response:
                print(f"💡 {response['help']}", file=sys.stderr)
            return 1

        status = response.get('status')
        message = response.get('message', '')

        if status == 'recording':
            print(f"🔴 Recording started")
        elif status == 'stopped':
            print(f"🛑 Recording stopped")
            print(f"⏳ Processing...")
        else:
            print(f"Status: {status}")
            if message:
                print(f"Message: {message}")

        return 0

    def status(self):
        """Get daemon status."""
        response = self._send_command('status')

        if 'error' in response:
            print(f"❌ Error: {response['error']}", file=sys.stderr)
            if 'help' in response:
                print(f"💡 {response['help']}", file=sys.stderr)
            return 1

        status = response.get('status', 'unknown')

        status_icons = {
            'idle': '⚪',
            'recording': '🔴',
            'processing': '⏳'
        }
        icon = status_icons.get(status, '❓')

        print(f"{icon} Status: {status}")
        return 0

    def stop_daemon(self):
        """Stop the daemon."""
        response = self._send_command('stop')

        if 'error' in response:
            print(f"❌ Error: {response['error']}", file=sys.stderr)
            if 'help' in response:
                print(f"💡 {response['help']}", file=sys.stderr)
            return 1

        print(f"✓ Daemon shutting down...")
        return 0

    def is_running(self):
        """Check if daemon is running."""
        return self.socket_path.exists()

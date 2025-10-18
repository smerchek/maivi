#!/usr/bin/env python3
"""
CLI interface for STT server.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Voice-to-Text STT Server with keyboard shortcuts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start daemon (keeps model loaded, responds to IPC)
  maivi-cli daemon

  # Start daemon in background
  maivi-cli daemon --detach

  # Toggle recording (daemon must be running)
  maivi-cli toggle

  # Check daemon status
  maivi-cli status

  # Stop daemon
  maivi-cli daemon stop

  # Legacy: Start server (clipboard only)
  maivi-cli server

  # Legacy: Start server with auto-paste
  maivi-cli server --auto-paste

  # Legacy: Toggle mode (press once to start, again to stop)
  maivi-cli server --toggle

  # Legacy: Show live UI window with transcription
  maivi-cli server --toggle --show-ui

Daemon Mode (Recommended):
  - Keeps model loaded in background (no 10s startup delay)
  - Use 'maivi-cli toggle' from Hyprland keybind
  - Instant response to recording commands
  - Bind in Hyprland: bind = SUPER, R, exec, maivi-cli toggle

Streaming Mode (SIMPLE OVERLAPPING CHUNKS):
  - Fixed 7s chunks with 4s overlap (3s slide)
  - Processes chunks in parallel during recording
  - Merges using overlap detection (simple and reliable)
  - Nearly instant completion when you stop recording
  - No complex algorithms, just clean overlap-based merging!
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Daemon subcommand
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon (keeps model loaded)')
    daemon_parser.add_argument(
        '--detach',
        action='store_true',
        help='Run daemon in background'
    )
    daemon_parser.add_argument(
        'daemon_action',
        nargs='?',
        choices=['start', 'stop'],
        default='start',
        help='Daemon action (default: start)'
    )
    _add_server_args(daemon_parser)

    # Toggle subcommand
    subparsers.add_parser('toggle', help='Toggle recording (daemon must be running)')

    # Status subcommand
    subparsers.add_parser('status', help='Check daemon status')

    # Server subcommand (legacy mode)
    server_parser = subparsers.add_parser('server', help='Run in server mode (legacy, with keyboard listener)')
    _add_server_args(server_parser)

    # Also support old behavior (no subcommand = server mode)
    _add_server_args(parser)

    args = parser.parse_args()

    # Handle commands
    if args.command == 'daemon':
        from maivi.cli.client import MaiviClient
        from maivi.cli.daemon import MaiviDaemon

        if args.daemon_action == 'stop':
            client = MaiviClient()
            sys.exit(client.stop_daemon())
        else:
            # Start daemon
            daemon = MaiviDaemon(
                auto_paste=getattr(args, 'auto_paste', False),
                window_seconds=getattr(args, 'window', 7.0),
                slide_seconds=getattr(args, 'slide', 3.0),
                start_delay_seconds=getattr(args, 'delay', 2.0),
                speed=getattr(args, 'speed', 1.0),
                output_file=getattr(args, 'output_file', None),
                show_ui=getattr(args, 'show_ui', False),
                ui_width=getattr(args, 'ui_width', 30),
                pause_paragraph_breaks=not getattr(args, 'no_pause_breaks', False),
                pause_threshold_seconds=getattr(args, 'pause_threshold', 1.0),
            )
            daemon.run(detach=args.detach)
            sys.exit(0)

    elif args.command == 'toggle':
        from maivi.cli.client import MaiviClient
        client = MaiviClient()
        sys.exit(client.toggle())

    elif args.command == 'status':
        from maivi.cli.client import MaiviClient
        client = MaiviClient()
        sys.exit(client.status())

    elif args.command == 'server' or args.command is None:
        # Legacy server mode or no subcommand
        _run_server(args)
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


def _add_server_args(parser):
    """Add server-specific arguments to parser."""
    parser.add_argument(
        "-p",
        "--auto-paste",
        action="store_true",
        help="Automatically paste transcribed text after copying to clipboard",
    )

    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Enable streaming mode for real-time transcription",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=7.0,
        help="Chunk size in seconds (default: 7.0, larger = better quality)",
    )

    parser.add_argument(
        "--slide",
        type=float,
        default=3.0,
        help="Slide interval in seconds (default: 3.0, window-slide = overlap for merging)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay before starting streaming in seconds (default: 2.0)",
    )

    parser.add_argument(
        "--hotkey",
        default="<ctrl>+<alt>",
        help="Hotkey combination (default: <ctrl>+<alt>)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for audio (1.25 = 25%% faster, 2.0 = 2x faster, default: 1.0)",
    )

    parser.add_argument(
        "-t",
        "--toggle",
        action="store_true",
        help="Toggle mode: press once to start, press again to stop (default: hold mode)",
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=None,
        help="Stream transcription to file (for real-time voice command detection)",
    )

    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Show live transcription UI window (streaming mode only)",
    )

    parser.add_argument(
        "--ui-width",
        type=int,
        default=30,
        help="Width of live UI in characters (default: 30)",
    )

    parser.add_argument(
        "--no-pause-breaks",
        action="store_true",
        help="Disable automatic paragraph breaks on long pauses (default: enabled)",
    )

    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=1.0,
        help="Minimum pause duration for paragraph break in seconds (default: 1.0)",
    )


def _run_server(args):
    """Run the server in legacy mode."""
    from maivi.cli.server import StreamingSTTServer
    from maivi.utils.ffmpeg_installer import ensure_ffmpeg_installed

    # Check for FFmpeg (optional but recommended for advanced audio processing)
    ensure_ffmpeg_installed(silent=False)
    print()

    # Create and run streaming STT server
    server = StreamingSTTServer(
        auto_paste=args.auto_paste,
        window_seconds=args.window,
        slide_seconds=args.slide,
        start_delay_seconds=args.delay,
        speed=args.speed,
        toggle_mode=args.toggle,
        output_file=args.output_file,
        show_ui=args.show_ui,
        ui_width=args.ui_width,
        pause_paragraph_breaks=not args.no_pause_breaks,
        pause_threshold_seconds=args.pause_threshold,
    )

    server.run()


if __name__ == "__main__":
    main()

"""
TUI slash-command handler for LocMemory.

Parses `/command [args]` input and renders output using Rich.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from core.memory.graph import GraphManager, TIER_NAMES


@dataclass
class CommandResult:
    """Outcome of a command invocation, returned to the chat loop."""
    handled: bool = False       # True if input was a slash command
    should_exit: bool = False   # True if chat loop should terminate
    skip_pipeline: bool = True  # True if run_pipeline should be skipped this turn


class CommandHandler:
    """Dispatch slash commands against the graph + extractor."""

    def __init__(
        self,
        graph_manager: GraphManager,
        extractor,
        console: Optional[Console] = None,
        on_clear: Optional[Callable[[], None]] = None,
    ):
        self.gm = graph_manager
        self.extractor = extractor
        self.console = console or Console()
        self.on_clear = on_clear
        self.extraction_enabled = True

    # ─────────────────────────── public api ───────────────────────────

    def is_command(self, text: str) -> bool:
        return text.startswith("/")

    def handle(self, text: str) -> CommandResult:
        parts = text[1:].strip().split(maxsplit=1)
        if not parts:
            self._err("empty command. try /help")
            return CommandResult(handled=True)

        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        dispatch = {
            "help":       self._cmd_help,
            "clear":      self._cmd_clear,
            "stats":      self._cmd_stats,
            "list":       self._cmd_list,
            "activate":   self._cmd_activate,
            "deactivate": self._cmd_deactivate,
            "exit":       self._cmd_exit,
            "quit":       self._cmd_exit,
        }
        handler = dispatch.get(cmd)
        if handler is None:
            self._err(f"unknown command: /{cmd}. try /help")
            return CommandResult(handled=True)

        return handler(args)

    # ─────────────────────────── commands ───────────────────────────

    def _cmd_help(self, _: str) -> CommandResult:
        table = Table(
            title="LocMemory commands",
            title_style="bold gold1",
            border_style="grey39",
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("command", style="bold blue")
        table.add_column("description")

        rows = [
            ("/help",              "show this help"),
            ("/clear",             "clear the screen"),
            ("/stats",             "show graph statistics (nodes, edges, tiers, domains)"),
            ("/list [category]",   "list memories; optional category filters by domain"),
            ("/activate",          "enable background fact extraction (default)"),
            ("/deactivate",        "pause background fact extraction"),
            ("/exit  |  /quit",    "leave the chat"),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)

        self.console.print(table)
        return CommandResult(handled=True)

    def _cmd_clear(self, _: str) -> CommandResult:
        os.system("cls" if os.name == "nt" else "clear")
        if self.on_clear:
            self.on_clear()
        return CommandResult(handled=True)

    def _cmd_stats(self, _: str) -> CommandResult:
        g = self.gm.graph
        tier_counts: dict[int, int] = {}
        domain_counts: dict[str, int] = {}
        for _nid, data in g.nodes(data=True):
            tier_counts[data.get("tier", 0)] = tier_counts.get(data.get("tier", 0), 0) + 1
            dom = data.get("domain", "") or "(none)"
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold blue")
        summary.add_column()
        summary.add_row("nodes",     str(g.number_of_nodes()))
        summary.add_row("edges",     str(g.number_of_edges()))
        summary.add_row("extraction", "on" if self.extraction_enabled else "off")

        tiers = Table(title="by tier", border_style="grey39", header_style="bold cyan")
        tiers.add_column("tier", style="bold blue")
        tiers.add_column("count", justify="right")
        for t in sorted(tier_counts):
            tiers.add_row(f"{t} ({TIER_NAMES.get(t, '?')})", str(tier_counts[t]))

        domains = Table(title="by domain", border_style="grey39", header_style="bold cyan")
        domains.add_column("domain", style="bold blue")
        domains.add_column("count", justify="right")
        for d, c in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
            domains.add_row(d, str(c))

        self.console.print(Panel(summary, title="[bold gold1]stats[/]", border_style="grey39"))
        if tier_counts:
            self.console.print(tiers)
        if domain_counts:
            self.console.print(domains)
        return CommandResult(handled=True)

    def _cmd_list(self, args: str) -> CommandResult:
        category = args.strip().lower() or None
        g = self.gm.graph

        rows = []
        for nid, data in g.nodes(data=True):
            dom = (data.get("domain") or "").lower()
            if category and dom != category:
                continue
            rows.append((
                nid[:8],
                TIER_NAMES.get(data.get("tier", 0), "?"),
                data.get("domain", "") or "-",
                str(data.get("text", ""))[:80],
            ))

        if not rows:
            msg = f"no memories in category '{category}'" if category else "no memories yet"
            self.console.print(f"[yellow]{msg}[/]")
            return CommandResult(handled=True)

        rows.sort(key=lambda r: (r[1], r[2]))

        title = f"memories (category: {category})" if category else "memories"
        table = Table(title=title, title_style="bold gold1", border_style="grey39", header_style="bold cyan")
        table.add_column("id", style="dim")
        table.add_column("tier", style="bold blue")
        table.add_column("domain")
        table.add_column("text")
        for r in rows:
            table.add_row(*r)

        self.console.print(table)
        self.console.print(f"[dim]{len(rows)} node(s)[/]")
        return CommandResult(handled=True)

    def _cmd_activate(self, _: str) -> CommandResult:
        self.extraction_enabled = True
        self.console.print("[bold green]✓[/] background extraction [bold]enabled[/]")
        return CommandResult(handled=True)

    def _cmd_deactivate(self, _: str) -> CommandResult:
        self.extraction_enabled = False
        self.console.print("[bold yellow]·[/] background extraction [bold]paused[/]")
        return CommandResult(handled=True)

    def _cmd_exit(self, _: str) -> CommandResult:
        return CommandResult(handled=True, should_exit=True)

    # ─────────────────────────── helpers ───────────────────────────

    def _err(self, msg: str):
        self.console.print(f"[bold red]error:[/] {msg}")

"""Pick a measurement script (and its toml), run it, watch the output.

    uv run spec-launcher [scripts folder]

Scripts run as separate processes with the same Python, so a crash never takes
the launcher down and every script also works from the terminal. "Stop" sends
Ctrl+C (the script's ``with setup:`` block then closes all devices); pressing
it again kills the process.
"""

import queue
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_FOLDER = Path(__file__).resolve().parents[3] / "scripts"


def find_scripts(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*.py") if not p.name.startswith("_")),
        key=lambda p: (len(p.relative_to(folder).parts), str(p)),
    )


def default_toml(script: Path) -> Path | None:
    toml = script.with_suffix(".toml")
    return toml if toml.is_file() else None


def command(script: Path, toml: str, arguments: str) -> list[str]:
    result = [sys.executable, "-u", str(script)]
    if toml.strip():
        result.append(toml.strip())
    return result + shlex.split(arguments)


class Launcher:
    def __init__(self, root, folder: Path):
        import tkinter as tk
        from tkinter import scrolledtext

        self.root, self.folder = root, folder
        self.process: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.scripts = find_scripts(folder)
        root.title("SPEC launcher")
        root.protocol("WM_DELETE_WINDOW", self.quit)

        left = tk.Frame(root)
        left.pack(side="left", fill="y", padx=6, pady=6)
        tk.Label(left, text=f"Scripts in {folder}", anchor="w").pack(fill="x")
        self.listbox = tk.Listbox(left, width=34, exportselection=False)
        self.listbox.pack(fill="y", expand=True)
        for script in self.scripts:
            self.listbox.insert("end", str(script.relative_to(folder)))
        self.listbox.bind("<<ListboxSelect>>", self.selected)

        right = tk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        form = tk.Frame(right)
        form.pack(fill="x")
        tk.Label(form, text="Measurement toml").grid(row=0, column=0, sticky="w")
        self.toml = tk.StringVar()
        tk.Entry(form, textvariable=self.toml).grid(row=0, column=1, sticky="ew")
        tk.Button(form, text="Browse…", command=self.browse).grid(row=0, column=2)
        tk.Label(form, text="Arguments").grid(row=1, column=0, sticky="w")
        self.arguments = tk.StringVar()
        tk.Entry(form, textvariable=self.arguments).grid(row=1, column=1, columnspan=2, sticky="ew")
        form.columnconfigure(1, weight=1)

        buttons = tk.Frame(right)
        buttons.pack(fill="x", pady=4)
        self.run_button = tk.Button(buttons, text="Run", width=10, command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = tk.Button(buttons, text="Stop", width=10, command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        self.status = tk.Label(buttons, text="idle", anchor="w")
        self.status.pack(side="left", padx=8)

        self.output = scrolledtext.ScrolledText(right, width=100, height=30, state="disabled")
        self.output.pack(fill="both", expand=True)
        if self.scripts:
            self.listbox.selection_set(0)
            self.selected()
        self.poll()

    @property
    def script(self) -> Path | None:
        selection = self.listbox.curselection()
        return self.scripts[selection[0]] if selection else None

    def selected(self, _event=None):
        toml = default_toml(self.script) if self.script else None
        self.toml.set(str(toml) if toml else "")
        self.arguments.set("")

    def browse(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            initialdir=self.folder, filetypes=[("Measurement toml", "*.toml"), ("All", "*")]
        )
        if path:
            self.toml.set(path)

    def write(self, text: str):
        self.output.config(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.config(state="disabled")

    def run(self):
        if self.process is not None or self.script is None:
            return
        cmd = command(self.script, self.toml.get(), self.arguments.get())
        self.write(f"\n$ {shlex.join(cmd[2:])}\n")
        self.process = subprocess.Popen(
            cmd,
            cwd=self.folder.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self.read, args=(self.process,), daemon=True).start()
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal", text="Stop")
        self.status.config(text=f"running {self.script.name}")

    def read(self, process):
        for line in process.stdout:
            self.lines.put(line)
        process.wait()
        self.lines.put(None)

    def poll(self):
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    code = self.process.returncode
                    self.write(f"[finished with exit code {code}]\n")
                    self.status.config(text="idle" if code == 0 else f"failed ({code})")
                    self.process = None
                    self.run_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                else:
                    self.write(line)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def stop(self):
        if self.process is None:
            return
        if self.stop_button.cget("text") == "Stop":
            self.process.send_signal(signal.SIGINT)  # devices close in the script
            self.stop_button.config(text="Kill")
            self.status.config(text="stopping…")
        else:
            self.process.kill()

    def quit(self):
        if self.process is not None:
            self.status.config(text="a script is running; stop it first")
            return
        self.root.destroy()


def main():
    import tkinter as tk

    folder = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_FOLDER
    root = tk.Tk()
    Launcher(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()

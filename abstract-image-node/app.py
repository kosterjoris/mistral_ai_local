from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from abstract_image_node import AbstractImageNode

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
PREVIEW_MAX = 460


class SettingsDialog(tk.Toplevel):
    """Tandwiel-dialoog: system prompt en alle node-instellingen aanpasbaar."""

    def __init__(self, master, node: AbstractImageNode):
        super().__init__(master)
        self.node = node
        self.title("Instellingen")
        self.resizable(False, False)
        s = node.settings

        form = ttk.Frame(self, padding=10)
        form.grid(row=0, column=0, sticky="nsew")

        ttk.Label(form, text="Model (Hugging Face id):").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=s.model_id)
        ttk.Entry(form, textvariable=self.model_var, width=44).grid(row=0, column=1, sticky="ew")

        ttk.Label(form, text="Device:").grid(row=1, column=0, sticky="w")
        self.device_var = tk.StringVar(value=s.device)
        ttk.Combobox(form, textvariable=self.device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(form, text="System prompt:").grid(row=2, column=0, sticky="nw")
        self.system_text = tk.Text(form, width=44, height=3, wrap="word")
        self.system_text.insert("1.0", s.system_prompt)
        self.system_text.grid(row=2, column=1, sticky="ew")

        ttk.Label(form, text="Negative prompt:").grid(row=3, column=0, sticky="nw")
        self.negative_text = tk.Text(form, width=44, height=2, wrap="word")
        self.negative_text.insert("1.0", s.negative_prompt)
        self.negative_text.grid(row=3, column=1, sticky="ew")

        ttk.Label(form, text="Stappen:").grid(row=4, column=0, sticky="w")
        self.steps_var = tk.StringVar(value=str(s.steps))
        ttk.Spinbox(form, from_=4, to=150, textvariable=self.steps_var, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(form, text="Guidance scale:").grid(row=5, column=0, sticky="w")
        self.guidance_var = tk.StringVar(value=str(s.guidance_scale))
        ttk.Entry(form, textvariable=self.guidance_var, width=8).grid(row=5, column=1, sticky="w")

        ttk.Label(form, text="Seed (-1 = willekeurig):").grid(row=6, column=0, sticky="w")
        self.seed_var = tk.StringVar(value=str(s.seed))
        ttk.Spinbox(form, from_=-1, to=2147483647, textvariable=self.seed_var, width=12).grid(row=6, column=1, sticky="w")

        ttk.Label(form, text="Output map:").grid(row=7, column=0, sticky="w")
        self.output_var = tk.StringVar(value=s.output_dir)
        ttk.Entry(form, textvariable=self.output_var, width=44).grid(row=7, column=1, sticky="ew")

        buttons = ttk.Frame(self, padding=(10, 0, 10, 10))
        buttons.grid(row=1, column=0, sticky="e")
        ttk.Button(buttons, text="Annuleren", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Opslaan", command=self.on_save).grid(row=0, column=1, padx=4)

        self.transient(master)
        self.grab_set()

    def on_save(self):
        try:
            updates = {
                "model_id": self.model_var.get().strip(),
                "device": self.device_var.get(),
                "system_prompt": self.system_text.get("1.0", "end").strip(),
                "negative_prompt": self.negative_text.get("1.0", "end").strip(),
                "steps": int(self.steps_var.get()),
                "guidance_scale": float(self.guidance_var.get()),
                "seed": int(self.seed_var.get()),
                "output_dir": self.output_var.get().strip() or "outputs",
            }
        except ValueError:
            messagebox.showerror("Ongeldige invoer", "Stappen, guidance en seed moeten getallen zijn.", parent=self)
            return
        if not updates["model_id"]:
            messagebox.showerror("Ongeldige invoer", "Model id mag niet leeg zijn.", parent=self)
            return
        self.node.update_settings(**updates)
        self.node.save_settings()
        self.destroy()


class TileApp(tk.Tk):
    def __init__(self, node: AbstractImageNode):
        super().__init__()
        self.node = node
        self.title("Abstracte plaatjes — test-node")
        self.minsize(580, 660)
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._preview_image = None

        s = node.settings
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Prompt:").pack(anchor="w")
        self.prompt_text = tk.Text(main, height=3, wrap="word")
        self.prompt_text.pack(fill="x")
        self.prompt_text.insert("1.0", "casual smartphone snapshot of an average person in a cluttered room, awkward angle, ungraded")

        row = ttk.Frame(main)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text="Breedte:").grid(row=0, column=0)
        self.width_var = tk.StringVar(value="512")
        ttk.Spinbox(row, from_=64, to=2048, increment=8, textvariable=self.width_var, width=7).grid(row=0, column=1)
        ttk.Label(row, text="Hoogte:").grid(row=0, column=2)
        self.height_var = tk.StringVar(value="512")
        ttk.Spinbox(row, from_=64, to=2048, increment=8, textvariable=self.height_var, width=7).grid(row=0, column=3)
        ttk.Label(row, text="Seed:").grid(row=0, column=4)
        self.seed_var = tk.StringVar(value=str(s.seed))
        ttk.Spinbox(row, from_=-1, to=2147483647, textvariable=self.seed_var, width=10).grid(row=0, column=5)
        ttk.Label(row, text="Stappen:").grid(row=0, column=6)
        self.steps_var = tk.StringVar(value=str(s.steps))
        ttk.Spinbox(row, from_=4, to=150, textvariable=self.steps_var, width=6).grid(row=0, column=7)

        btnrow = ttk.Frame(main)
        btnrow.pack(fill="x")
        self.gen_button = ttk.Button(btnrow, text="Genereer plaatje", command=self.on_generate)
        self.gen_button.pack(side="left")
        ttk.Button(btnrow, text="⚙", width=3, command=self.on_settings).pack(side="left", padx=6)
        ttk.Label(btnrow, text="tandwiel = instellingen (incl. system prompt)").pack(side="left")

        self.preview_label = ttk.Label(main, text="Nog geen plaatje", anchor="center", background="#dddddd")
        self.preview_label.pack(fill="both", expand=True, pady=6)

        self.status_var = tk.StringVar(value="Klaar. Kies resolutie, vul een prompt in en klik op Genereer.")
        ttk.Label(main, textvariable=self.status_var, wraplength=540).pack(anchor="w")

        self.after(100, self._poll)

    def on_settings(self):
        SettingsDialog(self, self.node)

    def on_generate(self):
        if self._busy:
            return
        prompt = self.prompt_text.get("1.0", "end").strip()
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            seed = int(self.seed_var.get())
            steps = int(self.steps_var.get())
        except ValueError:
            messagebox.showerror("Ongeldige invoer", "Resolutie, seed en stappen moeten getallen zijn.")
            return
        self._busy = True
        self.gen_button.config(state="disabled")
        self.status_var.set("Model laden / plaatje genereren…")

        def work():
            try:
                result = self.node.generate(
                    prompt,
                    width=width,
                    height=height,
                    seed=seed,
                    steps=steps,
                    progress_cb=lambda done, total: self._queue.put(("progress", done, total)),
                )
                self._queue.put(("done", result))
            except Exception as exc:
                self._queue.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                message = self._queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self.status_var.set(f"Stap {message[1]} van {message[2]}…")
                elif kind == "done":
                    self._show_result(message[1])
                elif kind == "error":
                    self._show_error(message[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _show_result(self, result):
        self._busy = False
        self.gen_button.config(state="normal")
        self.status_var.set(f"Klaar in {result.elapsed:.1f}s — opgeslagen: {result.path} (seed {result.seed})")
        try:
            image = tk.PhotoImage(file=str(result.path))
            largest = max(image.width(), image.height())
            if largest > PREVIEW_MAX:
                factor = largest // PREVIEW_MAX + 1
                image = image.subsample(factor, factor)
            self._preview_image = image
            self.preview_label.config(image=image, text="")
        except tk.TclError:
            self.preview_label.config(image="", text=f"Preview niet beschikbaar (Tk < 8.6?). Bestand: {result.path}")

    def _show_error(self, exc):
        self._busy = False
        self.gen_button.config(state="normal")
        self.status_var.set(f"Fout: {exc}")
        messagebox.showerror("Genereren mislukt", str(exc))


def main():
    node = AbstractImageNode.from_config(CONFIG_PATH)
    TileApp(node).mainloop()


if __name__ == "__main__":
    main()

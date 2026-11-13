from __future__ import annotations

import argparse
from pathlib import Path

from abstract_image_node import AbstractImageNode

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def main():
    parser = argparse.ArgumentParser(description="Genereer een abstract plaatje met een klein lokaal model (headless test, bijv. Linux).")
    parser.add_argument("prompt", help="Prompt voor het plaatje")
    parser.add_argument("--width", type=int, default=512, help="Breedte (wordt afgerond op veelvoud van 8)")
    parser.add_argument("--height", type=int, default=512, help="Hoogte (wordt afgerond op veelvoud van 8)")
    parser.add_argument("--seed", type=int, default=None, help="-1 of weglaten = willekeurig")
    parser.add_argument("--steps", type=int, default=None, help="Aantal diffusie-stappen (default uit config)")
    parser.add_argument("--system-prompt", default=None, help="Overschrijft de system prompt uit config")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Pad naar config.json")
    args = parser.parse_args()

    node = AbstractImageNode.from_config(args.config)
    result = node.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps,
        system_prompt=args.system_prompt,
    )
    print(f"Opgeslagen: {result.path}")
    print(f"Prompt: {result.prompt}")
    print(f"Seed: {result.seed} | {result.width}x{result.height} | {result.steps} stappen | {result.elapsed:.1f}s")


if __name__ == "__main__":
    main()

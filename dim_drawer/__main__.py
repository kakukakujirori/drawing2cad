"""CLI: python -m dim_drawer DXF... --out DIR"""

import argparse

from dim_drawer.pipeline import annotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxf", nargs="+", help="source DXF files")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="base style seed")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--text-height",
        type=float,
        default=None,
        help="dimension text height in drawing units",
    )
    args = parser.parse_args()

    for i, path in enumerate(args.dxf):
        _, png, placed, views, thick, thin = annotate(
            path,
            args.out,
            seed=args.seed + i,
            text_height=args.text_height,
            dpi=args.dpi,
        )
        print(
            f"{png}  views={views} dims={placed} "
            f"lw={thick / 100:.2f}/{thin / 100:.2f}mm"
        )


if __name__ == "__main__":
    main()

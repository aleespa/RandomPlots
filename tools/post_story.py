"""
Publish a set of Story cards to Instagram through the Instagram Graph API.

    uv run python -m tools.post_story 20261102_story --dry-run  # list + convert only
    uv run python -m tools.post_story 20261102_story            # publish, in order

Posts every `outputs/<folder>/story_<n>.png` as one Story, in order of n. The API takes
only a JPEG `image_url`, so each card is converted to a full-size quality-95 JPEG (in
`outputs/<folder>/jpeg/`) and served for a few seconds through the same temporary
Cloudflare tunnel as `tools.post_reel`. Credentials: the same `.env`.
"""

import argparse
import re
import sys

from loguru import logger
from PIL import Image

from tools.post_reel import ROOT, client, read_env, tunnel

SIZE = (1080, 1920)


def cards(folder):
    found = sorted(
        folder.glob("story_*.png"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])
    )
    if not found:
        raise SystemExit(f"No story_<n>.png in {folder}")
    return found


def to_jpeg(png, out_dir):
    out_dir.mkdir(exist_ok=True)
    image = Image.open(png).convert("RGB")
    if image.size != SIZE:
        raise SystemExit(f"{png.name} is {image.size}, not {SIZE}")
    jpg = out_dir / f"{png.stem}.jpg"
    image.save(jpg, quality=95, subsampling=0)
    return jpg


@logger.catch(reraise=False, onerror=lambda _: sys.exit(1))
def main():
    parser = argparse.ArgumentParser(description="Publish Story cards to Instagram.")
    parser.add_argument("folder", help="output folder name, e.g. 20261102_story")
    parser.add_argument(
        "--dry-run", action="store_true", help="convert only, no upload"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    folder = ROOT / "outputs" / args.folder
    jpegs = [to_jpeg(png, folder / "jpeg") for png in cards(folder)]
    for jpg in jpegs:
        logger.info(f"{jpg.relative_to(ROOT)}: {jpg.stat().st_size / 1024:.0f} KB")
    if args.dry_run:
        logger.success(f"Dry run OK: {len(jpegs)} Story cards, nothing uploaded")
        return

    ig = client(read_env())
    for n, jpg in enumerate(jpegs, 1):
        with tunnel(jpg) as url:
            cid = ig.post(f"{ig.user_id}/media", media_type="STORIES", image_url=url)[
                "id"
            ]
            ig.wait_until_finished(cid)
        media_id = ig.post(f"{ig.user_id}/media_publish", creation_id=cid)["id"]
        logger.success(f"Story {n}/{len(jpegs)} published: {media_id}")
    logger.success(f"All {len(jpegs)} Story cards are live")


if __name__ == "__main__":
    main()

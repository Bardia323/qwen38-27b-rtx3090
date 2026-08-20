"""Render the side-by-side demo from two demo_capture.py recordings.

One GPU means the two configurations cannot run simultaneously, so each lane was
recorded separately with per-token arrival times and this replays them together at
their real speed. Nothing is sped up, slowed down or interpolated: at video time t
a lane shows exactly the tokens whose recorded arrival was <= t.

  python3 bench/demo_render.py baseline.json dflash2.json --out docs/media/demo

Writes <out>.gif (for inline README rendering) and <out>.mp4. Needs pillow + ffmpeg.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

FPS = 12
W, H = 940, 530
PAD = 16
COL_W = (W - PAD * 3) // 2
HOLD_S = 1.6          # freeze at the end of each prompt so the result is readable
TITLE_S = 1.0         # title card between prompts, so the change of prompt is obvious
BODY_LINES = 13

BG = (13, 17, 23)
CARD = (22, 27, 34)
EDGE = (48, 54, 61)
FG = (201, 209, 217)
DIM = (110, 118, 129)
CYAN = (86, 211, 220)
GREEN = (63, 185, 80)
AMBER = (210, 153, 34)


def font(sz, bold=False):
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf"):
        try:
            return ImageFont.truetype(p, sz, index=1 if (bold and p.endswith("ttc")) else 0)
        except Exception:
            continue
    return ImageFont.load_default()


F_H1, F_H2, F_BODY, F_BIG, F_SM = font(17, True), font(12), font(12), font(26, True), font(11)


def prepare(p):
    """Annotate each streamed chunk with how many output tokens it carried.

    A chunk is not a token. With a 16-token verify block DFlash2 delivers a whole
    accepted run in one SSE event, so counting chunks would show 29 where 400 tokens
    arrived. The server only reports the total, so split it across chunks by character
    count -- monotone, sums to the real total, and exact for the one-token-per-chunk
    case (the unspeculated lane).
    """
    chunks = p["tokens"]
    total_chars = sum(len(c[1]) for c in chunks) or 1
    cum_c, cum = 0, []
    for t_ms, txt in chunks:
        cum_c += len(txt)
        cum.append((t_ms, cum_c * p["n_out"] / total_chars))
    p["_cum"] = cum
    return p


def visible(p, t_ms):
    """Text, token count and index visible at t_ms, by recorded arrival time."""
    chunks = p["tokens"]
    lo, hi = 0, len(chunks)
    while lo < hi:                      # chunks are in arrival order
        mid = (lo + hi) // 2
        if chunks[mid][0] <= t_ms:
            lo = mid + 1
        else:
            hi = mid
    n_tok = p["_cum"][lo - 1][1] if lo else 0.0
    return "".join(c[1] for c in chunks[:lo]), int(round(n_tok)), lo


def rate_at(p, i, t_ms, window=800.0):
    """Trailing-window tok/s, so the readout moves like a live meter."""
    cum = p["_cum"]
    if i < 2:
        return 0.0
    t0 = max(0.0, t_ms - window)
    first = 0
    for j in range(i - 1, -1, -1):
        if cum[j][0] < t0:
            first = j + 1
            break
    span = (cum[i - 1][0] - cum[first][0]) / 1000.0
    if span <= 0.05:
        return cum[i - 1][1] / max(t_ms / 1000.0, 1e-3)
    return (cum[i - 1][1] - cum[first][1]) / span


def wrap(text, cols):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, cols) or [""])
    return out


def panel(d, x, lane, prompt, t_ms, cols):
    title, sub, accent = lane
    # A lane that has stopped generating must stop its clock too, or it keeps
    # counting while the other lane finishes and the elapsed time is a lie.
    end_ms = prompt["tokens"][-1][0]
    done = t_ms >= end_ms
    t_ms = min(t_ms, end_ms)

    d.rounded_rectangle([x, 92, x + COL_W, H - 46], 8, fill=CARD, outline=EDGE)
    d.text((x + 14, 104), title, font=F_H1, fill=accent)
    d.text((x + 14, 126), sub, font=F_SM, fill=DIM)

    text, n, i = visible(prompt, t_ms)
    lines = wrap(text, cols)[-BODY_LINES:]
    y = 152
    for ln in lines:
        d.text((x + 14, y), ln, font=F_BODY, fill=FG)
        y += 15

    rate = prompt["decode_tok_s"] if done else rate_at(prompt, i, t_ms)
    ry = H - 118
    big = f"{rate:.0f}"
    d.text((x + 14, ry), big, font=F_BIG, fill=accent)
    d.text((x + 20 + d.textlength(big, font=F_BIG), ry + 12), "tok/s", font=F_SM, fill=DIM)
    d.text((x + COL_W - 150, ry + 2), f"{n:4d} / {prompt['n_out']} tokens", font=F_H2, fill=DIM)
    d.text((x + COL_W - 150, ry + 18), f"{t_ms / 1000.0:5.2f} s", font=F_H2, fill=DIM)
    if done:
        d.text((x + 14, ry + 34), f"done — avg {prompt['decode_tok_s']:.1f} tok/s",
               font=F_H2, fill=accent)
    return rate


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = os.path.expanduser(sys.argv[sys.argv.index("--out") + 1]
                             if "--out" in sys.argv else "demo")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames_dir = out + "_frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    lanes = (("Stock vLLM", "vllm serve, no speculative decoding", AMBER),
             ("This repo", "SPEC=dflash2  DFLASH_TOKENS=15  PREFIX_CACHE=1", GREEN))
    cols = (COL_W - 28) // 7

    prompts = list(zip(map(prepare, a["prompts"]), map(prepare, b["prompts"])))
    n_prompts = len(prompts)
    frame = 0

    def save(img):
        nonlocal frame
        img.save(f"{frames_dir}/f{frame:05d}.png")
        frame += 1

    for idx, (pa, pb) in enumerate(prompts):
        # Title card: without a hard break between prompts the panels just refill
        # and it is not obvious a new prompt started.
        for _ in range(int(TITLE_S * FPS)):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            tag = f"PROMPT {idx + 1} OF {n_prompts}"
            d.text((W / 2 - d.textlength(tag, font=F_H2) / 2, H / 2 - 54), tag,
                   font=F_H2, fill=CYAN)
            d.text((W / 2 - d.textlength(pa["label"], font=F_BIG) / 2, H / 2 - 26),
                   pa["label"], font=F_BIG, fill=FG)
            sub = f"{pa['prompt_tokens']:,} prompt tokens"
            d.text((W / 2 - d.textlength(sub, font=F_H2) / 2, H / 2 + 18), sub,
                   font=F_H2, fill=DIM)
            save(img)

        span = max(pa["tokens"][-1][0], pb["tokens"][-1][0])
        total = span / 1000.0 + HOLD_S
        for f in range(int(total * FPS) + 1):
            t_ms = f * 1000.0 / FPS
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)

            d.text((PAD, 16), "Qwen3.8-27B on one RTX 3090 @ 250 W", font=F_H1, fill=FG)
            pill = f" PROMPT {idx + 1}/{n_prompts} · {pa['label'].upper()} "
            pw = d.textlength(pill, font=F_H2)
            d.rounded_rectangle([PAD, 44, PAD + pw + 8, 66], 6, fill=CARD, outline=CYAN)
            d.text((PAD + 4, 48), pill, font=F_H2, fill=CYAN)
            d.text((PAD + pw + 20, 48), f"{pa['prompt_tokens']:,} prompt tokens",
                   font=F_H2, fill=DIM)

            ra = panel(d, PAD, lanes[0], pa, t_ms, cols)
            rb = panel(d, PAD * 2 + COL_W, lanes[1], pb, t_ms, cols)

            # Only once BOTH lanes have finished: a live ratio would divide the
            # finished lane's final average by the running lane's instantaneous dip
            # and read high (11.4x where the honest answer is 8.9x).
            if t_ms >= span:
                note = f"{pb['decode_tok_s'] / pa['decode_tok_s']:.1f}x faster"
                d.text((W // 2 - d.textlength(note, font=F_H1) / 2, H - 36),
                       note, font=F_H1, fill=CYAN)
            d.text((PAD, H - 32), "recorded separately, replayed at true speed",
                   font=F_SM, fill=DIM)
            save(img)

    pat = f"{frames_dir}/f*.png"
    pal = f"{frames_dir}/pal.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat,
                    "-vf", "palettegen=max_colors=64:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-i", pal,
                    "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", out + ".gif"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "23",
                    out + ".mp4"], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    for f in (out + ".gif", out + ".mp4"):
        print(f"{f}: {os.path.getsize(f) / 1e6:.2f} MB")


main()

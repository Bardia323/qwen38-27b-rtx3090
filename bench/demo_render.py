"""Render the side-by-side demo from two demo_capture.py recordings, plus a
third reference lane for a competing engine we did not and cannot run here.

One GPU means the two configurations cannot run simultaneously, so each lane was
recorded separately with per-token arrival times and this replays them together at
their real speed. Nothing is sped up, slowed down or interpolated: at video time t
a lane shows exactly the tokens whose recorded arrival was <= t.

The third lane (ninfer-3090) is not a recording: it is their own published number.
We do not have their proprietary .ninfer container and have never run it on Linux,
so that lane never streams fabricated text -- see ninfer_panel().

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

FPS = 15
W, H = 1280, 600
PAD = 18
N_COLS = 3
COL_W = (W - PAD * (N_COLS + 1)) // N_COLS
HOLD_S = 1.6          # freeze at the end of each prompt so the result is readable
TITLE_S = 1.1          # title card between prompts, so the change of prompt is obvious
FINAL_S = 3.4          # dedicated closing beat after the last prompt
BODY_LINES = 15
PEAK_WINDOW_MS = 2000.0

# -- the published reference lane ------------------------------------------
# We cannot run ninfer-3090: it needs a proprietary 18.21 GB .ninfer container
# and has never been run on Linux against a real model. This is their own
# number, not a measurement, and the lane must never look like one.
NINFER_RATE = 71.0
NINFER_SOURCE = "ninfer-3090 README · default branch release/v0.6.0-rtx3090"
NINFER_PROTOCOL = ("their own protocol: short prompts, thinking on, C1 only — "
                    "they publish no per-prompt-type numbers, so this same figure "
                    "is shown for all 3 prompts")

# -- palette --------------------------------------------------------------
# Self-contained: the frame paints its own background, so it reads the same
# whether the surrounding page is GitHub light or dark. Blue/orange/teal is a
# colourblind-safe triad (Okabe-Ito), unlike the amber/green this replaces.
BG = (10, 11, 13)
CARD = (19, 21, 25)
EDGE = (40, 43, 49)
EDGE_SOFT = (29, 31, 36)
FG = (233, 235, 238)
DIM = (142, 148, 158)
DIM2 = (92, 97, 106)
NEUTRAL = (150, 155, 245)      # violet-blue accent for chrome (pills, titles)
STOCK = (92, 165, 235)          # blue   -- "Stock vLLM"
REPO = (255, 158, 68)           # orange -- "This repo"
NINFER = (73, 197, 168)         # teal   -- "ninfer-3090" (published, not measured)
MUTED_TAG = (150, 155, 163)     # neutral grey for the "not measured" disclosure


def font(path, sz, index=0):
    try:
        return ImageFont.truetype(path, sz, index=index)
    except Exception:
        return ImageFont.load_default()


SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"

F_TITLE = font(SANS, 21, 1)     # header title (bold)
F_SUBTITLE = font(SANS, 12, 7)  # header subtitle (light)
F_PILL = font(SANS, 12, 10)     # prompt pill (medium)
F_LANE = font(SANS, 15, 1)      # lane name (bold)
F_LANESUB = font(MONO, 10, 0)   # lane command, kept mono -- it *is* a command
F_BODY = font(MONO, 12, 0)      # streaming text
F_HERO = font(SANS, 36, 1)      # tok/s hero number (bold)
F_UNIT = font(SANS, 12, 10)     # "tok/s" (medium)
F_STAT = font(SANS, 11, 0)      # token count / elapsed
F_DONE = font(SANS, 10, 10)     # done / peak / published badges
F_STAMP = font(SANS, 10, 7)     # honesty stamp
F_CARD_TAG = font(SANS, 13, 10)
F_CARD_TITLE = font(SANS, 38, 1)
F_CARD_SUB = font(SANS, 13, 7)
F_FINAL_KICKER = font(SANS, 13, 10)
F_FINAL_HERO = font(SANS, 68, 1)
F_FINAL_SUB = font(SANS, 15, 0)
F_FINAL_ROW = font(SANS, 13, 0)
F_FINAL_ROW_B = font(SANS, 13, 10)
F_NOTE_TAG = font(SANS, 11, 10)   # ninfer's "published" pill
F_NOTE = font(SANS, 10, 7)        # ninfer's protocol note
F_NOTE_BODY = font(SANS, 13, 0)   # ninfer's static placeholder message

GLOBAL_MAX = 0.0   # set in main(): shared scale so bar length is always comparable


def prepare(p):
    """Annotate each streamed chunk with how many output tokens it carried, and
    find the best sustained rate over any window >= PEAK_WINDOW_MS.

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
    p["_peak"] = peak_rate(cum)
    return p


def peak_rate(cum, min_ms=PEAK_WINDOW_MS):
    """Best sustained rate over any window >= min_ms, from real recorded arrivals.

    None when the whole run is shorter than the window -- there is nothing
    honest to report, and printing 0 would look like a real (bad) result.
    For a fixed start i the fastest qualifying window is always the shortest
    one that still clears min_ms (stretching it further can only dilute the
    rate with a slower stretch), so a single two-pointer sweep finds the max.
    """
    n = len(cum)
    if n < 2:
        return None
    t = [c[0] for c in cum]
    tok = [c[1] for c in cum]
    if t[-1] - t[0] < min_ms:
        return None
    best, j = 0.0, 0
    for i in range(n):
        if j < i:
            j = i
        while j < n - 1 and t[j] - t[i] < min_ms:
            j += 1
        if t[j] - t[i] >= min_ms:
            best = max(best, (tok[j] - tok[i]) / ((t[j] - t[i]) / 1000.0))
    return best


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


def wrap_px(d, text, fnt, max_w):
    """Greedy word-wrap by measured pixel width, for the proportional (sans) font."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if not cur or d.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# Menlo has no glyph for these two emoji (renders as a blank box) -- swap in a
# dingbat it does support. Display-only: token counts are computed from the
# original, unmodified text before this ever runs.
GLYPH_FALLBACK = str.maketrans({"✅": "✓", "❌": "✗"})


def rrect(d, box, r, **kw):
    d.rounded_rectangle(box, r, **kw)


def meter(d, x, y, w, h, value, vmax, color, track=EDGE):
    """A horizontal bar on a fixed 0..vmax scale, so bars drawn at the same
    scale in different columns still show the true ratio between them -- no
    arithmetic needed."""
    rrect(d, [x, y, x + w, y + h], h / 2, fill=track)
    frac = max(0.0, min(1.0, value / vmax)) if vmax else 0.0
    fw = int(w * frac)
    if fw > h:  # avoid a squashed pill shorter than it is tall
        rrect(d, [x, y, x + fw, y + h], h / 2, fill=color)
    elif fw > 0:
        d.ellipse([x, y, x + h, y + h], fill=color)


def dots(d, cx, y, n, idx, active, inactive):
    r, gap = 3, 12
    total = n * gap - (gap - r * 2)
    x = cx - total / 2
    for i in range(n):
        c = active if i == idx else inactive
        d.ellipse([x, y - r, x + r * 2, y + r], fill=c)
        x += gap


def stamp(d):
    d.text((PAD, H - 26), "recorded separately, replayed at true speed",
           font=F_STAMP, fill=DIM2)


def badge_pill(d, right_x, y, text, fnt, fg, outline, fill=EDGE_SOFT):
    """A small right-aligned rounded tag, e.g. DONE / PEAK / PUBLISHED badges."""
    w = d.textlength(text, font=fnt)
    rrect(d, [right_x - w - 16, y - 4, right_x, y + 15], 9, fill=fill, outline=outline)
    d.text((right_x - w - 8, y), text, font=fnt, fill=fg)
    return y + 15


def panel(d, x, lane, prompt, t_ms, cols, top, bottom):
    title, sub, accent = lane
    # A lane that has stopped generating must stop its clock too, or it keeps
    # counting while the other lanes finish and the elapsed time is a lie.
    end_ms = prompt["tokens"][-1][0]
    done = t_ms >= end_ms
    t_ms = min(t_ms, end_ms)

    rrect(d, [x, top, x + COL_W, bottom], 10, fill=CARD, outline=EDGE)
    d.ellipse([x + 16, top + 19, x + 24, top + 27], fill=accent)
    d.text((x + 32, top + 14), title, font=F_LANE, fill=FG)
    d.text((x + 16, top + 40), sub, font=F_LANESUB, fill=DIM2)
    d.line([x + 16, top + 60, x + COL_W - 16, top + 60], fill=EDGE_SOFT, width=1)

    text, n, i = visible(prompt, t_ms)
    lines = wrap(text.translate(GLYPH_FALLBACK), cols)[-BODY_LINES:]
    y = top + 72
    for ln in lines:
        d.text((x + 16, y), ln, font=F_BODY, fill=FG)
        y += 17

    rate = prompt["decode_tok_s"] if done else rate_at(prompt, i, t_ms)
    ry = bottom - 108
    big = f"{rate:.0f}"
    bw = d.textlength(big, font=F_HERO)
    d.text((x + 16, ry), big, font=F_HERO, fill=accent)
    d.text((x + 20 + bw, ry + 21), "tok/s", font=F_UNIT, fill=DIM)
    meter(d, x + 16, ry + 42, 120, 6, rate, GLOBAL_MAX, accent)

    stat_x = x + COL_W - 16
    line1 = f"{n:,} / {prompt['n_out']:,} tokens"
    d.text((stat_x - d.textlength(line1, font=F_STAT), ry + 2), line1, font=F_STAT, fill=DIM)
    line2 = f"{t_ms / 1000.0:.2f}s elapsed"
    d.text((stat_x - d.textlength(line2, font=F_STAT), ry + 16), line2, font=F_STAT, fill=DIM2)
    if done:
        by = badge_pill(d, stat_x, ry + 36, f"DONE  ·  avg {prompt['decode_tok_s']:.1f} tok/s",
                         F_DONE, accent, accent)
        # The peak is a different, larger number from a different definition
        # (best short burst vs. whole-run average) -- its own labelled row so
        # it is never mistaken for the average right above it.
        if prompt["_peak"] is not None:
            badge_pill(d, stat_x, by + 6, f"PEAK  ·  {prompt['_peak']:.1f} tok/s (2s window)",
                       F_DONE, accent, MUTED_TAG)
    return rate


def ninfer_panel(d, x, t_ms, span_ms, top, bottom):
    """The reference lane. There is no recording: no per-token arrivals exist to
    replay, so this never streams text -- real or invented. It shows only their
    published number, a constant meter/counter derived from it, and a
    disclosure that stays on screen the whole time this lane is visible."""
    rrect(d, [x, top, x + COL_W, bottom], 10, fill=CARD, outline=EDGE)
    d.ellipse([x + 16, top + 19, x + 24, top + 27], fill=NINFER)
    d.text((x + 32, top + 14), "ninfer-3090", font=F_LANE, fill=FG)

    tag = "PUBLISHED FIGURE — NOT RUN ON THIS BOX"
    tw = d.textlength(tag, font=F_NOTE_TAG)
    rrect(d, [x + 16, top + 36, x + 16 + tw + 16, top + 58], 9, fill=EDGE_SOFT, outline=NINFER)
    d.text((x + 24, top + 42), tag, font=F_NOTE_TAG, fill=NINFER)

    ny = top + 66
    for ln in wrap_px(d, NINFER_PROTOCOL, F_NOTE, COL_W - 32)[:2]:
        d.text((x + 16, ny), ln, font=F_NOTE, fill=DIM)
        ny += 13

    d.line([x + 16, top + 96, x + COL_W - 16, top + 96], fill=EDGE_SOFT, width=1)

    # static placeholder in place of streaming text -- see module docstring
    note = ["No streaming output for this lane.", "",
            "Their own published number, replayed here", "as a constant rate for visual comparison only."]
    body_top, body_bottom = top + 106, bottom - 116
    ty = body_top + (body_bottom - body_top - len(note) * 18) / 2
    for ln in note:
        if ln:
            tw2 = d.textlength(ln, font=F_NOTE_BODY)
            d.text((x + COL_W / 2 - tw2 / 2, ty), ln, font=F_NOTE_BODY, fill=DIM)
        ty += 18

    t_ms = min(t_ms, span_ms)
    rate = NINFER_RATE
    ry = bottom - 108
    big = f"{rate:.0f}"
    bw = d.textlength(big, font=F_HERO)
    d.text((x + 16, ry), big, font=F_HERO, fill=NINFER)
    d.text((x + 20 + bw, ry + 21), "tok/s", font=F_UNIT, fill=DIM)
    meter(d, x + 16, ry + 42, 120, 6, rate, GLOBAL_MAX, NINFER)

    n_tok = int(rate * t_ms / 1000.0)
    stat_x = x + COL_W - 16
    line1 = f"~{n_tok:,} tokens (extrapolated)"
    d.text((stat_x - d.textlength(line1, font=F_STAT), ry + 2), line1, font=F_STAT, fill=DIM)
    line2 = f"{t_ms / 1000.0:.2f}s elapsed"
    d.text((stat_x - d.textlength(line2, font=F_STAT), ry + 16), line2, font=F_STAT, fill=DIM2)
    by = badge_pill(d, stat_x, ry + 36, "CONSTANT · NOT MEASURED", F_DONE, NINFER, NINFER)
    badge_pill(d, stat_x, by + 6, NINFER_SOURCE, F_DONE, DIM, MUTED_TAG)


def header(d, idx, n_prompts, pa):
    d.text((PAD, 16), "Qwen3.8-27B decode speed", font=F_TITLE, fill=FG)
    d.text((PAD, 42), "one RTX 3090 @ 250 W · same prompts · lanes recorded separately",
           font=F_SUBTITLE, fill=DIM)

    pill = f"PROMPT {idx + 1}/{n_prompts}  ·  {pa['label'].upper()}"
    pw = d.textlength(pill, font=F_PILL)
    py = 68
    rrect(d, [PAD, py, PAD + pw + 24, py + 26], 13, fill=CARD, outline=NEUTRAL)
    d.text((PAD + 12, py + 6), pill, font=F_PILL, fill=NEUTRAL)
    tok_lbl = f"{pa['prompt_tokens']:,} prompt tokens"
    d.text((PAD + pw + 40, py + 6), tok_lbl, font=F_PILL, fill=DIM)
    dots(d, W - PAD - 40, py + 13, n_prompts, idx, NEUTRAL, EDGE)


def title_card(d, idx, n_prompts, pa):
    tag = f"PROMPT {idx + 1} OF {n_prompts}"
    d.text((W / 2 - d.textlength(tag, font=F_CARD_TAG) / 2, H / 2 - 78), tag,
           font=F_CARD_TAG, fill=NEUTRAL)
    label = pa["label"]
    d.text((W / 2 - d.textlength(label, font=F_CARD_TITLE) / 2, H / 2 - 52),
           label, font=F_CARD_TITLE, fill=FG)
    d.line([W / 2 - 60, H / 2 + 12, W / 2 + 60, H / 2 + 12], fill=EDGE, width=1)
    sub = f"{pa['prompt_tokens']:,} prompt tokens"
    d.text((W / 2 - d.textlength(sub, font=F_CARD_SUB) / 2, H / 2 + 26), sub,
           font=F_CARD_SUB, fill=DIM)
    dots(d, W / 2, H / 2 + 60, n_prompts, idx, NEUTRAL, EDGE)


def arrow(d, x, y, w, color):
    """A small drawn arrow -- HelveticaNeue has no usable glyph for U+2192."""
    yc = y
    d.line([x, yc, x + w - 6, yc], fill=color, width=2)
    d.polygon([(x + w - 10, yc - 5), (x + w, yc), (x + w - 10, yc + 5)], fill=color)


def final_card(d, results, hero, top):
    key, label, ra, rb, _n_rate, ratio = hero
    kicker = "SAME GPU · SAME MODEL · THIS REPO'S SERVING STACK"
    y = top
    d.text((W / 2 - d.textlength(kicker, font=F_FINAL_KICKER) / 2, y), kicker,
           font=F_FINAL_KICKER, fill=DIM)

    y += 36
    big = f"{ratio:.1f}x faster"
    d.text((W / 2 - d.textlength(big, font=F_FINAL_HERO) / 2, y), big,
           font=F_FINAL_HERO, fill=REPO)

    y += 94
    a_txt, b_txt, unit = f"{ra:.1f}", f"{rb:.1f}", " tok/s"
    parts = [(label + "  ", FG, F_FINAL_SUB), (a_txt, STOCK, F_FINAL_ROW_B)]
    w1 = sum(d.textlength(t, font=fn) for t, _, fn in parts)
    aw = 26
    w2 = d.textlength(b_txt + unit, font=F_FINAL_ROW_B)
    total = w1 + 10 + aw + 10 + w2
    x = W / 2 - total / 2
    for t, c, fn in parts:
        d.text((x, y), t, font=fn, fill=c)
        x += d.textlength(t, font=fn)
    arrow(d, x + 4, y + 9, aw, DIM)
    x += aw + 14
    d.text((x, y), b_txt, font=F_FINAL_ROW_B, fill=REPO)
    x += d.textlength(b_txt, font=F_FINAL_ROW_B)
    d.text((x, y), unit, font=F_FINAL_ROW_B, fill=FG)

    # every result, on the same fixed scale used throughout the video. Three
    # stacked bars per row: stock, this repo, and ninfer's published figure --
    # the last one identical in every row, because that is the honest fact.
    rows_w = 620
    rx = W / 2 - rows_w / 2
    y += 50
    for k, lbl, a_rate, b_rate, n_rate, r in results:
        row_hi = (k == key)
        name_col = FG if row_hi else DIM
        if row_hi:  # this is the number the whole demo is building to -- call it out
            rrect(d, [rx - 14, y - 8, rx + rows_w + 14, y + 62], 8,
                  fill=EDGE_SOFT, outline=REPO)
        d.text((rx, y), lbl, font=F_FINAL_ROW_B if row_hi else F_FINAL_ROW, fill=name_col)
        rr = f"{r:.1f}x"
        d.text((rx + rows_w - d.textlength(rr, font=F_FINAL_ROW_B), y), rr,
               font=F_FINAL_ROW_B, fill=REPO)
        by = y + 22
        bar_w = rows_w - 60
        meter(d, rx, by, bar_w, 7, a_rate, GLOBAL_MAX, STOCK)
        meter(d, rx, by + 12, bar_w, 7, b_rate, GLOBAL_MAX, REPO)
        meter(d, rx, by + 24, bar_w, 7, n_rate, GLOBAL_MAX, NINFER)
        y += 74

    legend_y = y + 4
    lx = rx
    for name, c in (("stock vLLM", STOCK), ("this repo", REPO), ("ninfer-3090 (published)", NINFER)):
        d.ellipse([lx, legend_y, lx + 8, legend_y + 8], fill=c)
        d.text((lx + 14, legend_y - 3), name, font=F_STAMP, fill=DIM)
        lx += 22 + d.textlength(name, font=F_STAMP) + 20
    note = "ninfer's 71.0 tok/s is their own published figure (short prompts, C1 only) — not measured here, and the same number every time."
    d.text((rx, legend_y + 20), note, font=F_STAMP, fill=DIM2)


def main():
    global GLOBAL_MAX
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = os.path.expanduser(sys.argv[sys.argv.index("--out") + 1]
                             if "--out" in sys.argv else "demo")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames_dir = out + "_frames"
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir)

    lanes = (("Stock vLLM", "vllm serve  (no speculative decoding)", STOCK),
             ("This repo", "SPEC=dflash2  DFLASH_TOKENS=15  PREFIX_CACHE=1", REPO))

    prompts = list(zip(map(prepare, a["prompts"]), map(prepare, b["prompts"])))
    n_prompts = len(prompts)
    peaks = [pa["_peak"] for pa, pb in prompts if pa["_peak"]] + \
            [pb["_peak"] for pa, pb in prompts if pb["_peak"]]
    GLOBAL_MAX = max([NINFER_RATE] + peaks +
                      [max(pa["decode_tok_s"], pb["decode_tok_s"]) for pa, pb in prompts]) * 1.08

    tmp = Image.new("RGB", (1, 1))
    dtmp = ImageDraw.Draw(tmp)
    char_w = dtmp.textlength("0" * 20, font=F_BODY) / 20
    cols = int((COL_W - 32) / char_w)

    frame = 0

    def save(img):
        nonlocal frame
        img.save(f"{frames_dir}/f{frame:05d}.png")
        frame += 1

    col_x = [PAD + i * (COL_W + PAD) for i in range(N_COLS)]

    for idx, (pa, pb) in enumerate(prompts):
        # Title card: without a hard break between prompts the panels just refill
        # and it is not obvious a new prompt started.
        for _ in range(int(TITLE_S * FPS)):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            title_card(d, idx, n_prompts, pa)
            stamp(d)
            save(img)

        span = max(pa["tokens"][-1][0], pb["tokens"][-1][0])
        total = span / 1000.0 + HOLD_S
        for f in range(int(total * FPS) + 1):
            t_ms = f * 1000.0 / FPS
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)

            header(d, idx, n_prompts, pa)
            panel(d, col_x[0], lanes[0], pa, t_ms, cols, top=110, bottom=H - 40)
            panel(d, col_x[1], lanes[1], pb, t_ms, cols, top=110, bottom=H - 40)
            ninfer_panel(d, col_x[2], t_ms, span, top=110, bottom=H - 40)

            # Only once BOTH *measured* lanes have finished: a live ratio would
            # divide the finished lane's final average by the running lane's
            # instantaneous dip and read high (11.4x where honest is 8.9x).
            # ninfer is a published constant, not a measurement, so it plays
            # no part in this badge.
            if t_ms >= span:
                note = f"{pb['decode_tok_s'] / pa['decode_tok_s']:.1f}x faster"
                nw = d.textlength(note, font=F_PILL)
                rrect(d, [W / 2 - nw / 2 - 12, 68, W / 2 + nw / 2 + 12, 94], 13,
                      fill=CARD, outline=REPO)
                d.text((W / 2 - nw / 2, 74), note, font=F_PILL, fill=REPO)
            stamp(d)
            save(img)

    # closing beat: land the headline number the whole demo was building to
    results = [(pa["key"], pa["label"], pa["decode_tok_s"], pb["decode_tok_s"], NINFER_RATE,
                pb["decode_tok_s"] / pa["decode_tok_s"]) for pa, pb in prompts]
    hero = next((r for r in results if r[0] == "copy"), max(results, key=lambda r: r[5]))
    for _ in range(int(FINAL_S * FPS)):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        final_card(d, results, hero, top=48)
        stamp(d)
        save(img)

    pat = f"{frames_dir}/f*.png"
    pal = f"{frames_dir}/pal.png"
    # The design is flat UI, not photographic, so a bigger palette with no
    # dithering renders crisper text *and* compresses smaller than a small
    # dithered one: dither noise is what was costing bytes and blurring glyphs.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat,
                    "-vf", "palettegen=max_colors=192:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-i", pal,
                    "-lavfi", "paletteuse=dither=none", out + ".gif"], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-pattern_type", "glob", "-i", pat, "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "23",
                    out + ".mp4"], check=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    for f in (out + ".gif", out + ".mp4"):
        print(f"{f}: {os.path.getsize(f) / 1e6:.2f} MB")


main()

# NutriFit AI
# Setup: put config.toml inside a .streamlit/ folder next to this file,
# and install the packages in requirements.txt.

import re
import json
import html
import base64
import hashlib
import pandas as pd
import os
import time
import urllib.request
from datetime import date, timedelta
import streamlit as st
from groq import Groq

import db


def get_secret(name, default=None):
    """Read a setting from Streamlit secrets, then from environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


st.set_page_config(page_title="NutriFit AI", page_icon="🥗", layout="wide")

if not get_secret("GROQ_API_KEY"):
    st.error("This app isn't set up yet: the GROQ_API_KEY secret is missing.")
    st.stop()

groq_client = Groq(api_key=get_secret("GROQ_API_KEY"))
AI_MODEL = get_secret("AI_MODEL", "qwen/qwen3.8-27b")
FOOD_VISION_MODEL = get_secret("VISION_MODEL", "qwen/qwen3.8-27b")
GEMINI_KEY = get_secret("GEMINI_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_THINKING = str(get_secret("GEMINI_THINKING", "medium")).lower()   # low, medium, high or off
VISION_FALLBACK = str(get_secret("VISION_FALLBACK", "off")).lower() == "on"   # use Groq if Gemini fails

# ---------- State ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user" not in st.session_state:
    st.session_state.user = None
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "profile" not in st.session_state:
    st.session_state.profile = {}


def go_to_page(target):
    st.session_state.page = target


def clean(text):
    """Remove <think> blocks and stray <br> tags some models include."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"\s*<br\s*/?>\s*", "; ", text, flags=re.I)
    return text.strip()


# ---------- PDF export ----------
def plan_to_pdf(plan, profile):
    """Turn the markdown plan (tables, headings, lists) into a styled PDF."""
    from io import BytesIO
    from datetime import date
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    ink, muted = colors.HexColor("#14332A"), colors.HexColor("#5B6F66")
    tint, line_c = colors.HexColor("#F1F4EA"), colors.HexColor("#DDE3D5")
    avail = A4[0] - 40 * mm

    def fmt(t):
        t = t.encode("cp1252", "ignore").decode("cp1252").replace("`", "")
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(t)).replace("*", "")

    base = ParagraphStyle("base", fontName="Helvetica", fontSize=10.5, leading=15, textColor=ink)
    h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=22, leading=26)
    h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=14,
                        leading=18, spaceBefore=16, spaceAfter=6)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12, textColor=muted)
    cell = ParagraphStyle("cell", parent=base, fontSize=10, leading=13)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    item = ParagraphStyle("item", parent=base, leftIndent=14, bulletIndent=2, spaceAfter=3)

    story = [Paragraph("Your weekly activity plan", h1),
             Paragraph(f"NutriFit AI  |  {date.today():%d %B %Y}", small), Spacer(1, 12)]

    rows = [[Paragraph(f"<b>{fmt(k)}</b>", cell), Paragraph(fmt(v), cell)] for k, v in profile.items()]
    box = Table(rows, colWidths=[42 * mm, avail - 42 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint), ("BOX", (0, 0), (-1, -1), 0.75, line_c),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [box, Spacer(1, 10)]

    lines, i = plan.splitlines(), 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            grid = [[c.strip() for c in r.strip("|").split("|")]
                    for r in block if not re.fullmatch(r"[\s|:\-]+", r)]
            n = max(len(r) for r in grid)
            grid = [r + [""] * (n - len(r)) for r in grid]
            widths = [avail * w for w in (0.2, 0.62, 0.18)] if n == 3 else [avail / n] * n
            data = [[Paragraph(fmt(c), head if k == 0 else cell) for c in r] for k, r in enumerate(grid)]
            tbl = Table(data, colWidths=widths, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ink),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, tint]),
                ("GRID", (0, 0), (-1, -1), 0.5, line_c), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
            story += [tbl, Spacer(1, 8)]
            continue
        if ln.startswith("#"):
            story.append(Paragraph(fmt(ln.lstrip("#").strip()), h2))
        elif re.match(r"[-*+]\s+", ln):
            story.append(Paragraph(fmt(re.sub(r"^[-*+]\s+", "", ln)), item, bulletText="\u2022"))
        elif re.match(r"\d+[.)]\s+", ln):
            num, txt = re.split(r"[.)]\s+", ln, maxsplit=1)
            story.append(Paragraph(fmt(txt), item, bulletText=f"{num}."))
        elif ln:
            story.append(Paragraph(fmt(ln), base))
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(20 * mm, 10 * mm, "NutriFit AI - general guidance, not medical advice")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=18 * mm,
                      title="NutriFit weekly plan").build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#F6F8F5;--ink:#14332A;--muted:#566B62;--brand:#2E8B57;--accent:#F4A93B;--line:#E2E8E0;--tint:#EEF3EA;}

/* Base */
.stApp{background:var(--bg);font-family:'Plus Jakarta Sans',sans-serif;line-height:1.55;}
.stApp p,.stApp li,.stApp label,.stApp span{color:var(--ink);}
.main .block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1100px;}
footer,#MainMenu{visibility:hidden;}
h1,h2,h3,h4{font-family:'Plus Jakarta Sans',sans-serif;color:var(--ink)!important;letter-spacing:-.01em;}
h3{font-size:1.15rem!important;font-weight:700!important;}

/* Page header */
.page-head{margin-bottom:1.6rem;padding-bottom:1.1rem;border-bottom:1px solid var(--line);}
.page-head h1{font-size:2rem;font-weight:800;letter-spacing:-.02em;margin:0;}
.page-head p{margin:.35rem 0 0;color:var(--muted)!important;font-size:1.02rem;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--ink);}
[data-testid="stSidebar"] *{color:#E4EFE8!important;}
.brand{display:flex;align-items:center;gap:.75rem;margin:.4rem 0 1.4rem;}
.brand-mark{width:38px;height:38px;flex:none;}
.brand-name{font-weight:800;font-size:1.15rem;line-height:1.1;}
.brand-sub{font-size:.8rem;opacity:.75;}
.side-note{font-size:.82rem;line-height:1.45;padding:.8rem .9rem;margin-top:1.5rem;
  border:1px solid rgba(255,255,255,.2);border-radius:10px;}
[data-testid="stSidebar"] [role="radiogroup"]{gap:.2rem;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.6rem .85rem;border-radius:10px;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--accent);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) *{color:var(--ink)!important;font-weight:700;}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none;}

/* The plate: brand mark and home hero */
.plate,.brand-mark{border-radius:50%;
  background:radial-gradient(circle at 36% 40%,#2E8B57 0 17%,transparent 18%),
             radial-gradient(circle at 66% 34%,#F4A93B 0 14%,transparent 15%),
             radial-gradient(circle at 56% 68%,#D9534F 0 12%,transparent 13%),#F5F6EF;
  box-shadow:inset 0 0 0 5px #E4EADB;}
.hero{display:flex;align-items:center;justify-content:space-between;gap:2rem;background:var(--ink);
  border-radius:20px;padding:2.2rem 2.6rem;margin-bottom:2rem;}
.hero h1{color:#F5F6EF!important;font-size:2.5rem;font-weight:800;line-height:1.1;margin:0 0 .7rem;}
.hero p{color:#BFD3C8!important;font-size:1.05rem;max-width:40ch;margin:0;}
.plate{flex:none;width:140px;height:140px;box-shadow:inset 0 0 0 10px #E4EADB;}
@media(max-width:700px){.plate{display:none;}.hero{padding:1.6rem;}.hero h1{font-size:2rem;}}

/* Cards and panels (containers with a key) */
[class*="st-key-card_"],[class*="st-key-panel_"]{background:#fff;border:1px solid var(--line);
  border-radius:14px;padding:1.3rem 1.4rem;box-shadow:0 1px 2px rgba(20,51,42,.06);}
[class*="st-key-card_"]{min-height:190px;display:flex;flex-direction:column;}
[class*="st-key-card_"]>div:last-child{margin-top:auto;}
[class*="st-key-card_"] h3{margin:0 0 .35rem;}
[class*="st-key-card_"] p{color:var(--muted)!important;margin:0 0 .6rem;}
.info,.fact{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.3rem;height:100%;}
.fact{margin-top:1rem;}
.info h4{margin:0 0 .3rem;font-size:1rem!important;font-weight:700;}
.info p{margin:0;color:var(--muted)!important;}

/* Buttons: solid for main actions, outlined inside cards and for suggestions */
.stButton button,.stDownloadButton button{background:var(--brand);border:1px solid var(--brand);
  border-radius:10px;min-height:44px;width:100%;font-weight:600;}
.stButton button p,.stDownloadButton button p{color:#fff!important;font-weight:600;}
.stButton button:hover,.stDownloadButton button:hover{background:var(--ink);border-color:var(--ink);}
.stButton button:focus-visible,.stDownloadButton button:focus-visible,
[data-testid="stSidebar"] label:focus-within{outline:3px solid var(--accent);outline-offset:2px;}
[class*="st-key-card_"] button{background:#fff;border:1px solid var(--brand);}
[class*="st-key-card_"] button p{color:var(--brand)!important;}
[class*="st-key-card_"] button:hover{background:var(--brand);}
[class*="st-key-card_"] button:hover p{color:#fff!important;}
.st-key-starters button{background:#fff;border:1px solid var(--line);text-align:left;}
.st-key-starters button p{color:var(--ink)!important;font-weight:500;}
.st-key-starters button:hover{background:var(--tint);border-color:var(--brand);}
.st-key-starters button:hover p{color:var(--ink)!important;}

/* Steps, tabs, inputs, chat, uploader */
.steps{display:flex;flex-wrap:wrap;gap:.6rem 1.6rem;margin:0 0 .6rem;}
.steps span{display:flex;align-items:center;gap:.5rem;font-weight:500;}
.steps b{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;
  background:var(--ink);color:#fff!important;font-size:.8rem;}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{font-weight:600;padding:.6rem 1rem;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{border-radius:10px!important;}
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:14px;}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:2px dashed var(--brand);border-radius:14px;}

/* Scan results */
.scan-done{display:flex;flex-direction:column;margin-bottom:.8rem;}
.scan-done b{font-size:1.25rem;color:var(--brand);}
.scan-done span{color:var(--muted);}
.kcal{display:flex;align-items:baseline;gap:.6rem;background:var(--ink);border-radius:14px;padding:1rem 1.3rem;margin-bottom:.8rem;}
.kcal b{font-size:2.6rem;font-weight:800;line-height:1;color:#F5F6EF;}
.kcal span{color:#BFD3C8;}
.macros{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;}
.macro{display:flex;flex-direction:column;background:#fff;border:1px solid var(--line);border-radius:12px;padding:.8rem .9rem;}
.macro span{display:flex;align-items:center;gap:.4rem;font-weight:600;font-size:.9rem;}
.macro u{display:inline-block;width:10px;height:10px;border-radius:50%;}
.macro b{font-size:1.5rem;font-weight:800;}
.macro i{font-style:normal;color:var(--muted);font-size:.8rem;}
.split{display:flex;height:8px;border-radius:99px;overflow:hidden;margin:.8rem 0;}
.split i{display:block;height:100%;}
.balance{background:#fff;border:1px solid var(--line);border-left:5px solid var(--brand);border-radius:12px;padding:.9rem 1.1rem;}
.balance.fair{border-left-color:var(--accent);}
.balance span{color:var(--muted);font-size:.85rem;}
.balance b{display:block;font-size:1.25rem;}
.balance p{margin:.2rem 0 0;color:var(--muted)!important;}
.scan-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin-top:.9rem;}
@media(max-width:800px){.scan-grid{grid-template-columns:1fr;}}
.scan-card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.2rem;}
.scan-card h4{margin:0 0 .6rem;font-size:1rem!important;font-weight:700;}
.scan-card .row{display:flex;justify-content:space-between;gap:1rem;padding:.35rem 0;border-bottom:1px solid var(--tint);}
.scan-card .row:last-child{border-bottom:none;}
.scan-card .bar{height:6px;background:var(--tint);border-radius:99px;overflow:hidden;margin:0 0 .3rem;}
.scan-card .bar i{display:block;height:100%;background:var(--brand);}
.scan-card p,.muted-note{margin:0;color:var(--muted)!important;font-size:.92rem;}

/* Home */
.st-key-hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#14332A 0%,#1B4A3A 55%,#256F46 100%);
  border-radius:24px;padding:2.4rem 2.8rem;margin-bottom:1.4rem;}
.st-key-hero::before{content:"";position:absolute;right:-90px;top:-90px;width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(244,169,59,.3),transparent 70%);pointer-events:none;}
.st-key-hero h1{color:#F5F6EF!important;font-size:2.7rem;font-weight:800;line-height:1.08;margin:0 0 .8rem;}
.st-key-hero p{color:#CFE3D8!important;font-size:1.08rem;max-width:42ch;margin:0 0 1.3rem;}
.st-key-hero_scan button{background:var(--accent);border-color:var(--accent);}
.st-key-hero_scan button p{color:var(--ink)!important;}
.st-key-hero_scan button:hover{background:#F7BC62;border-color:#F7BC62;}
.st-key-hero_ask button{background:transparent;border:1px solid rgba(245,246,239,.55);}
.st-key-hero_ask button:hover{background:rgba(255,255,255,.12);border-color:#fff;}
/* Hero buttons: slim, equal width, text centred */
.st-key-hero_scan,.st-key-hero_ask,.st-key-hero_scan .stButton,.st-key-hero_ask .stButton{width:100%;}
.st-key-hero_scan button,.st-key-hero_ask button{width:100%;min-height:44px;padding:.4rem 1rem;border-radius:10px;
  display:flex;align-items:center;justify-content:center;text-align:center;box-shadow:none;}
.st-key-hero_ask button{border-width:1px;}
.st-key-hero button div,.st-key-hero button [data-testid="stMarkdownContainer"]{width:100%;display:flex;justify-content:center;align-items:center;}
.st-key-hero button p{margin:0!important;max-width:none!important;font-size:.95rem;font-weight:600;line-height:1.2;text-align:center;}
.hero-art{position:relative;max-width:290px;margin:0 auto;}
.hero-art svg{display:block;width:100%;height:auto;}
.hero-art .leaf{position:absolute;animation:float 7s ease-in-out infinite;}
.hero-art .l1{width:36px;top:2%;left:0;}
.hero-art .l2{width:26px;top:12%;right:-2%;animation-delay:-2.5s;}
.hero-art .l3{width:22px;bottom:8%;left:6%;animation-delay:-4.5s;}
.tip{display:flex;align-items:center;gap:1rem;background:#fff;border:1px solid var(--line);
  border-left:5px solid var(--accent);border-radius:14px;padding:.9rem 1.2rem;margin-bottom:1.4rem;}
.tip span{flex:none;background:var(--accent);color:var(--ink)!important;font-weight:700;font-size:.74rem;
  letter-spacing:.05em;text-transform:uppercase;padding:.25rem .65rem;border-radius:99px;}
.tip p{margin:0;}
.st-key-card_scan{--c:#F4A93B;}.st-key-card_ask{--c:#2E8B57;}.st-key-card_fit{--c:#4A7FB5;}
.st-key-card_barcode{--c:#E07A3F;}.st-key-card_profile{--c:#C45C8A;}
.st-key-card_nutri{--c:#2A9D8F;}.st-key-card_allergy{--c:#D9534F;}.st-key-card_saved{--c:#8E6BB0;}
[class*="st-key-card_"]{border-top:4px solid var(--c,var(--brand));}
.icon{display:grid;place-items:center;width:46px;height:46px;border-radius:12px;margin-bottom:.8rem;
  background:#EEF3EA;background:color-mix(in srgb,var(--c) 15%,white);}
.icon svg{width:24px;height:24px;stroke:var(--c);fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:.4rem 0 1rem;}
.how>div{background:var(--tint);border-radius:14px;padding:1.1rem 1.2rem;}
.how b{display:inline-grid;place-items:center;width:28px;height:28px;border-radius:50%;background:var(--ink);
  color:#fff!important;font-size:.85rem;margin-bottom:.5rem;}
.how h4{margin:0 0 .25rem;font-size:1rem!important;font-weight:700;}
.how p{margin:0;color:var(--muted)!important;font-size:.92rem;}
.home-foot{text-align:center;color:var(--muted)!important;font-size:.88rem;margin-top:2rem;}
@media(max-width:800px){.how{grid-template-columns:1fr;}.st-key-hero{padding:1.6rem;}.st-key-hero h1{font-size:2rem;}
  .tip{flex-direction:column;align-items:flex-start;gap:.5rem;}}

/* Page backdrop */
.stApp{background-color:var(--bg);
  background-image:radial-gradient(circle at 8% -5%,rgba(46,139,87,.14),transparent 38%),
    radial-gradient(circle at 100% 12%,rgba(244,169,59,.16),transparent 34%),
    radial-gradient(rgba(20,51,42,.07) 1px,transparent 1.2px);
  background-size:auto,auto,24px 24px;background-attachment:fixed;}

/* Page banner */
.banner{position:relative;overflow:hidden;display:flex;align-items:center;gap:1.2rem;background:#fff;
  background:linear-gradient(110deg,color-mix(in srgb,var(--c) 18%,white) 0%,#fff 75%);
  border:1px solid var(--line);border-left:6px solid var(--c);border-radius:18px;
  padding:1.4rem 1.7rem;margin-bottom:1.6rem;box-shadow:0 1px 2px rgba(20,51,42,.06);}
.banner::after{content:"";position:absolute;right:-50px;top:-50px;width:190px;height:190px;border-radius:50%;background:var(--c);opacity:.1;}
.banner::before{content:"";position:absolute;right:70px;bottom:-60px;width:110px;height:110px;border-radius:50%;background:var(--c);opacity:.08;}
.banner .badge{flex:none;display:grid;place-items:center;width:56px;height:56px;border-radius:16px;background:var(--c);
  box-shadow:0 6px 14px rgba(20,51,42,.18);}
.banner .badge svg{width:28px;height:28px;stroke:#fff;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.banner h1{margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-.02em;}
.banner p{margin:.25rem 0 0;color:var(--muted)!important;font-size:1.02rem;}
@media(max-width:700px){.banner{padding:1.1rem;gap:.9rem;}.banner h1{font-size:1.5rem;}}

/* Panels: tabs, forms, chat, section titles, about cards */
.stTabs [data-baseweb="tab-list"]{background:#fff;border:1px solid var(--line);border-bottom:none;border-radius:16px 16px 0 0;padding:.35rem .6rem 0;}
.stTabs [data-baseweb="tab-panel"]{background:#fff;border:1px solid var(--line);border-top:none;border-radius:0 0 16px 16px;padding:1.4rem 1.5rem 1.6rem;}
.stTabs .info{background:var(--bg);}
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.4rem 1.5rem;}
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]){background:var(--tint);}
[data-testid="stChatInput"]{background:#fff;border:1px solid var(--line);border-radius:16px;}
h2.section{display:flex;align-items:center;gap:.6rem;font-size:1.3rem;margin:1.8rem 0 .8rem;}
h2.section::before{content:"";width:6px;height:1.15em;border-radius:4px;background:var(--brand);}
.about{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-bottom:1rem;}
.about>div{background:#fff;border:1px solid var(--line);border-top:4px solid var(--c);border-radius:14px;padding:1.2rem 1.4rem;}
.about h4{margin:0 0 .4rem;font-size:1.05rem!important;font-weight:700;}
.about p,.about li{color:var(--muted)!important;margin:0 0 .3rem;}
.about ul{margin:.2rem 0 0 1.1rem;padding:0;}
@media(max-width:800px){.about{grid-template-columns:1fr;}}

/* Product */
.prod{display:flex;gap:1rem;align-items:center;margin-bottom:.8rem;}
.prod img{width:96px;height:96px;object-fit:contain;background:#fff;border:1px solid var(--line);border-radius:12px;padding:4px;}
.prod h3{margin:0 0 .15rem;font-size:1.2rem!important;}
.prod p{margin:0;color:var(--muted)!important;}
.score-row{display:flex;flex-wrap:wrap;gap:.7rem;margin:.8rem 0;}
.score{display:flex;align-items:center;gap:.6rem;background:#fff;border:1px solid var(--line);border-radius:12px;padding:.55rem .85rem;}
.score i.grade{display:grid;place-items:center;flex:none;width:34px;height:34px;border-radius:8px;font-style:normal;font-weight:800;}
.score span{font-size:.85rem;color:var(--muted);line-height:1.25;}
.score b{display:block;color:var(--ink);}
.tagrow{display:flex;flex-wrap:wrap;gap:.4rem;margin:.2rem 0 .6rem;}
.tag{background:var(--tint);border-radius:99px;padding:.2rem .7rem;font-size:.85rem;}
.tag.warn{background:#FBE3E1;}
.scan-card .ingredients{max-height:13rem;overflow:auto;}

/* Scan accuracy aids */
.scan-done em{display:inline-block;margin-top:.3rem;font-style:normal;font-size:.78rem;font-weight:700;padding:.15rem .6rem;border-radius:99px;background:var(--tint);width:fit-content;}
.scan-done em.high{background:#DDF0E3;}.scan-done em.medium{background:#FCEFD5;}.scan-done em.low{background:#FBE3E1;}
.scan-card.items{margin-top:.8rem;}

/* Motion: answers touch, plus one gentle ambient moment (the plate) */
.stButton button,.stDownloadButton button,[class*="st-key-card_"],.info,
[data-testid="stSidebar"] [role="radiogroup"] label{
  transition:transform .18s ease,box-shadow .18s ease,background-color .18s ease,border-color .18s ease;}
.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-2px);box-shadow:0 6px 14px rgba(20,51,42,.2);}
.stButton button:active,.stDownloadButton button:active{transform:translateY(0) scale(.97);box-shadow:none;}
[class*="st-key-card_"]:hover{transform:translateY(-4px);border-color:var(--brand);box-shadow:0 10px 24px rgba(20,51,42,.12);}
.info:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(20,51,42,.08);}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{transform:translateX(3px);}
@keyframes float{0%,100%{transform:translateY(0) rotate(0deg);}50%{transform:translateY(-8px) rotate(6deg);}}
@keyframes rise{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
.plate{animation:float 6s ease-in-out infinite;}
[data-testid="stChatMessage"],[data-testid="stMarkdownContainer"] table{animation:rise .35s ease both;}
@media (prefers-reduced-motion:reduce){
  .plate,.hero-art .leaf,[data-testid="stChatMessage"],[data-testid="stMarkdownContainer"] table{animation:none!important;}
  .stButton button,.stDownloadButton button,[class*="st-key-card_"],.info,
  [data-testid="stSidebar"] [role="radiogroup"] label{transition:none!important;transform:none!important;}}

/* Tables (weekly plan) */
[data-testid="stMarkdownContainer"] table{width:100%;border-collapse:separate;border-spacing:0;background:#fff;
  border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:1rem;}
[data-testid="stMarkdownContainer"] th{background:var(--ink)!important;color:#F5F6EF!important;font-weight:600;
  text-align:left;padding:.7rem 1rem;border:none;}
[data-testid="stMarkdownContainer"] td{color:var(--ink)!important;background:#fff!important;
  padding:.7rem 1rem;border:none;border-top:1px solid var(--line);}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td{background:var(--tint)!important;}
</style>
""", unsafe_allow_html=True)


PAGE_THEME = {
    "My Profile": ("profile", "#C45C8A"),
    "AI Assistant": ("ask", "#2E8B57"),
    "Food Scanner": ("scan", "#F4A93B"),
    "Barcode Scanner": ("barcode", "#E07A3F"),
    "Nutrition Centre": ("nutri", "#2A9D8F"),
    "Fitness Planner": ("fit", "#4A7FB5"),
    "Allergy Centre": ("allergy", "#D9534F"),
    "Saved Information": ("saved", "#8E6BB0"),
    "About and Privacy": ("about", "#5B6F66"),
}


def header(title, sub):
    icon, color = PAGE_THEME.get(title, ("about", "#2E8B57"))
    st.markdown(f'<div class="banner" style="--c:{color}"><div class="badge">{ICONS[icon]}</div>'
                f'<div><h1>{title}</h1><p>{sub}</p></div></div>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<h2 class="section">{title}</h2>', unsafe_allow_html=True)


def info(title, text):
    return f'<div class="info"><h4>{title}</h4><p>{text}</p></div>'


# ---------- Accounts and saving ----------
@st.cache_resource
def _setup_db():
    db.init()


_setup_db()


# ---------- Online safeguards ----------
class VisionError(Exception):
    """Both photo AIs failed. The message already explains why, in plain words."""


def friendly(e):
    """A short, safe explanation for an AI error (the raw error stays out of the page)."""
    if isinstance(e, VisionError):
        return str(e)
    t = str(e).lower()
    if "no module named" in t:
        return "a required package isn't installed, so please tell the app owner"
    if any(w in t for w in ("perday", "per day", "daily", "limit: 0")):
        return ("the free allowance for this AI model has run out (or it has none), so please try again later "
                "or ask the app owner to switch models or turn on billing")
    if "429" in t or "rate limit" in t or "resource_exhausted" in t or "quota" in t:
        return "the AI service is busy right now, so please wait a minute and try again"
    if "404" in t or "not_found" in t or "is not found" in t:
        return "the AI model name isn't recognised, so please tell the app owner"
    if "401" in t or "api key" in t or "authentication" in t:
        return "the app's AI key isn't working, so please tell the app owner"
    return "something went wrong, so please try again"


def _retry_wait(e):
    """Seconds to wait before retrying a busy or overloaded error, or None if waiting won't help."""
    t = str(e).lower()
    if any(w in t for w in ("perday", "per day", "daily", "limit: 0")):
        return None
    if any(w in t for w in ("429", "resource_exhausted", "503", "unavailable", "overloaded")):
        m = re.search(r"retry\w*\D{0,12}(\d+(?:\.\d+)?)\s*s", t)
        wait = float(m.group(1)) + 1 if m else 4.0
        return wait if wait <= 20 else None
    return None


def limit_ok():
    """Allow a limited number of AI requests per visitor per hour, to protect the app's AI key."""
    cap = int(get_secret("RATE_LIMIT_PER_HOUR", 30))
    now = time.time()
    recent = [t for t in st.session_state.get("ai_calls", []) if now - t < 3600]
    if len(recent) >= cap:
        st.session_state.ai_calls = recent
        st.warning("You've reached the hourly limit for AI features. Please try again a little later.")
        return False
    st.session_state.ai_calls = recent + [now]
    return True


def prepare_image(photo, max_side=1280):
    """Shrink big phone photos before sending them, so scans are faster and don't fail."""
    from io import BytesIO
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(BytesIO(photo.getvalue()))).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# ---------- Profile ----------
AGE_GROUPS = ["Under 13", "13-17", "18-39", "40-59", "60+"]
GOALS = ["Eat more healthily", "Get more active", "Both", "Just learning"]
LEVELS = ["Beginner", "Active", "Experienced"]
DIETS = ["Vegetarian", "Vegan", "Pescatarian", "Halal", "Kosher"]
ALLERGEN_WORDS = {
    "Peanuts": ["peanut", "groundnut"],
    "Tree nuts": ["almond", "cashew", "walnut", "pecan", "hazelnut", "pistachio",
                  "brazil nut", "macadamia", "nut butter", "praline", "marzipan"],
    "Milk": ["milk", "cheese", "butter", "yoghurt", "yogurt", "cream", "whey", "casein", "ghee"],
    "Eggs": ["egg", "mayonnaise", "meringue", "albumin"],
    "Fish": ["fish", "salmon", "tuna", "cod", "tilapia", "anchovy", "sardine", "mackerel"],
    "Shellfish": ["shrimp", "prawn", "crab", "lobster", "crayfish", "mussel", "oyster", "clam", "squid"],
    "Wheat": ["wheat", "bread", "pasta", "flour", "couscous", "semolina", "noodle"],
    "Soy": ["soy", "tofu", "tempeh", "edamame", "miso"],
    "Sesame": ["sesame", "tahini", "hummus"],
}


def pref(key, default=None):
    return (st.session_state.get("profile") or {}).get(key, default)


def pick_valid(options, key, default):
    value = pref(key, default)
    return value if value in options else default


def profile_context():
    """A sentence for the AI prompts describing the user's profile (empty if none)."""
    parts = []
    if pref("age_group"):
        parts.append(f"age group {pref('age_group')}")
    if pref("goal"):
        parts.append(f"main goal: {pref('goal')}")
    if pref("level"):
        parts.append(f"activity level: {pref('level')}")
    if pref("diet"):
        parts.append("diet: " + ", ".join(pref("diet")))
    if pref("allergies"):
        parts.append("allergies: " + ", ".join(pref("allergies")))
    if not parts:
        return ""
    return (" About the user: " + "; ".join(parts) + ". Tailor your answers to this, keep them "
            "age-appropriate, and never suggest foods that contain their listed allergens.")


def scan_note():
    allergies = pref("allergies")
    if not allergies:
        return ""
    return (f" The user is allergic to: {', '.join(allergies)}. Include any of these in the "
            "allergens list if they are likely or possible in this food.")


def allergy_alert(text):
    """Flag words in a scan result that relate to the allergens on the user's profile."""
    allergies = pref("allergies")
    if not allergies:
        return
    low = text.lower()
    hits = {}
    for a in allergies:
        found = [w for w in ALLERGEN_WORDS.get(a, []) if re.search(rf"\b{re.escape(w)}s?\b", low)]
        if found:
            hits[a] = found[:3]
    if hits:
        st.warning("**Heads up, this may involve your allergens.** "
                   + "; ".join(f"{a} ({', '.join(w)})" for a, w in hits.items())
                   + ". Check the ingredients carefully.")
    else:
        st.info("None of your listed allergens were mentioned, but a photo can't show every "
                "ingredient. Always check the label or ask the cook.")


# ---------- Food scan results ----------
SCAN_SCHEMA = """{
  "is_food": true,
  "title": "short name of the meal",
  "confidence": "high, medium or low",
  "notes": "one sentence: what you used to judge the portions, and anything you were unsure about",
  "items": [{"name": "Jollof rice", "grams": 0, "kcal_per_100g": 0, "protein_per_100g": 0, "carbs_per_100g": 0, "fat_per_100g": 0}],
  "details": [{"label": "Saturated fat", "value": "0 g"}],
  "ingredients": ["ingredient"],
  "vitamins": [{"name": "Vitamin C", "dv_percent": 0}],
  "allergens": ["allergen that may be present"],
  "insights": ["one short, useful observation"],
  "balance": {"rating": "Well balanced, Fairly balanced or Could be more balanced", "reason": "one short sentence"}
}"""
MACRO_COLORS = ("#2E8B57", "#F4A93B", "#4A7FB5")


@st.cache_resource
def _gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


def vision_ai(system, text, photo, max_tokens=1200, temperature=0.2, max_side=1280, json_mode=False, thinking=None):
    """Ask a vision model about a photo: Google Gemini first when a key is set, Groq as the backup."""
    image = prepare_image(photo, max_side)
    st.session_state.gemini_error = ""
    if GEMINI_KEY:
        try:
            from google.genai import types
            parts = [types.Part.from_bytes(data=image, mime_type="image/jpeg"), text]
            reply, failure = None, None
            for level in ([thinking, None] if thinking else [None]):   # retry without thinking if it is rejected
                try:
                    options = {"system_instruction": system or None,
                               "response_mime_type": "application/json" if json_mode else None}
                    if level:
                        options["thinking_config"] = types.ThinkingConfig(thinking_level=level)
                    for attempt in range(3):          # a busy service often clears within seconds
                        try:
                            reply = _gemini_client(GEMINI_KEY).models.generate_content(
                                model=GEMINI_MODEL, config=types.GenerateContentConfig(**options), contents=parts)
                            break
                        except Exception as e:
                            wait = _retry_wait(e)
                            if wait is None or attempt == 2:
                                raise
                            time.sleep(wait)
                    break
                except Exception as e:
                    failure = e
                    if "thinking" not in str(e).lower():
                        break
            if reply is None:
                raise failure
            if reply.text:
                st.session_state.vision_used = "Gemini"
                return reply.text
            raise ValueError("Gemini returned an empty reply")
        except Exception as e:
            print(f"Gemini error ({GEMINI_MODEL}): {e}", flush=True)     # visible in the app logs
            st.session_state.gemini_error = friendly(e)
            if not VISION_FALLBACK:      # Gemini only: show the problem instead of a less accurate answer
                raise VisionError(f"Gemini ({GEMINI_MODEL}): {friendly(e)}") from e
    encoded = base64.b64encode(image).decode("utf-8")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]}]
    try:
        response = groq_client.chat.completions.create(
            model=FOOD_VISION_MODEL, messages=messages, temperature=temperature,
            max_tokens=max(max_tokens, 2500))      # the model may "think" before answering
    except Exception as e:
        print(f"Groq vision error ({FOOD_VISION_MODEL}): {e}", flush=True)
        if st.session_state.get("gemini_error"):
            raise VisionError(f"Gemini: {st.session_state.gemini_error}. Backup AI: {friendly(e)}") from e
        raise
    st.session_state.vision_used = "Groq"
    return clean(response.choices[0].message.content)


def vision_note():
    """A small line saying which AI read the last photo (and why the backup was used)."""
    used, problem = st.session_state.get("vision_used"), st.session_state.get("gemini_error")
    if used == "Gemini":
        return "Analysed with Google Gemini."
    if used == "Groq" and problem:
        return f"Google Gemini wasn't available ({problem}), so the backup AI was used."
    if used == "Groq" and not GEMINI_KEY:
        return "Google Gemini isn't set up yet (no GEMINI_API_KEY found), so the backup AI was used."
    return ""


def run_scan(photo, hint="", size=""):
    """Send the photo to the vision model and return its raw reply."""
    system = (
        "You are NutriFit AI Food Scanner, an expert at estimating meals from photos. Work like a "
        "dietitian and reply with ONE JSON object only: no markdown and no text before or after it.\n"
        "1. List every distinct food or drink you can see, and likely hidden ingredients such as cooking "
        "oil, sauces or sugar. Identify regional dishes by their local names (for example jollof rice, "
        "waakye, banku, fufu, kenkey, red red, ugali, injera, biryani) and allow for the oil and starch "
        "typical of them. Never invent items you cannot see.\n"
        "2. Estimate each item's weight in grams from visible scale references: the plate or bowl size "
        "(a dinner plate is about 26 cm), cutlery, hands, cups or packaging. If there is no reference, "
        "choose the middle of the plausible range, not an extreme.\n"
        "3. Give typical nutrition per 100 g for each item as it is cooked and served, using standard "
        "food composition values, including the oil and sauce it was cooked with.\n"
        "4. Lower the confidence for mixed dishes, hidden ingredients or unfamiliar foods. Never claim "
        "certainty. Never comment on bacteria, freshness or food safety. Never diagnose conditions or "
        "recommend restrictive diets.\n"
        'If the photo does not show food or a drink, reply {"is_food": false}. '
        "details, vitamins, ingredients and allergens describe the whole meal as eaten. "
        "Use exactly this structure:\n" + SCAN_SCHEMA)
    extra = ""
    if hint.strip():
        extra += (f" The user says this is: {hint.strip()}. Trust that for what the food is, "
                  "but still judge the amounts from the photo.")
    if size and size != "Not sure":
        extra += f" They describe the portion as {size.lower()}."
    thinking = GEMINI_THINKING if GEMINI_THINKING in ("low", "medium", "high") else None
    return vision_ai(system, "Analyse this photo." + extra + scan_note(), photo,
                     max_side=1600, json_mode=True, thinking=thinking)


def _num(v):
    m = re.search(r"\d+(\.\d+)?", str(v))
    return float(m.group()) if m else None


def _txt(v, n=80):
    return str(v).strip()[:n] if v is not None else ""


def _list(v, n):
    return v[:n] if isinstance(v, list) else []


def parse_scan(raw):
    m = re.search(r"\{.*\}", clean(raw), flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _item_kcal(i):
    """Calories of one item. If its calories and macros disagree a lot, trust the macros."""
    per100 = i["kcal100"]
    macro = 4 * i["protein100"] + 4 * i["carbs100"] + 9 * i["fat100"]
    if per100 <= 0 or (macro > 0 and abs(per100 - macro) > 0.25 * max(per100, macro)):
        per100 = macro or per100
    return i["grams"] / 100 * per100


def scan_totals(items):
    """Add up the items. The arithmetic is done here, not by the AI."""
    return {"calories": round(sum(_item_kcal(i) for i in items)),
            "protein": round(sum(i["grams"] / 100 * i["protein100"] for i in items)),
            "carbs": round(sum(i["grams"] / 100 * i["carbs100"] for i in items)),
            "fat": round(sum(i["grams"] / 100 * i["fat100"] for i in items))}


def _scale_text(text, ratio):
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(.*)", text or "")
    return f"{round(float(m.group(1)) * ratio, 1):g} {m.group(2)}".strip() if m else text


def apply_portions(d, grams):
    """Recalculate a scan after the user corrects the grams of each item."""
    new = []
    for item, g in zip(d["items"], grams):
        try:
            g = max(0.0, min(float(g), 2000.0))
        except (TypeError, ValueError):
            g = item["grams"]
        new.append({**item, "grams": g})
    base = sum(i["grams"] for i in d["items"]) or 1
    ratio = sum(i["grams"] for i in new) / base      # detail cards scale with the total amount
    return {**d, "items": new, **scan_totals(new),
            "details": [{**x, "value": _scale_text(x["value"], ratio)} for x in d["details"]],
            "vitamins": [{**v, "dv": v["dv"] * ratio} for v in d["vitamins"]]}


def normalise_scan(d):
    """Validate the model's JSON. Returns a clean dict, {"is_food": False}, or None if unreadable."""
    if not isinstance(d, dict):
        return None
    if d.get("is_food") is False:
        return {"is_food": False}
    items = []
    for x in _list(d.get("items"), 12):
        grams = _num(x.get("grams")) if isinstance(x, dict) else None
        if not grams:
            continue
        items.append({"name": _txt(x.get("name"), 40) or "Food", "grams": min(grams, 2000.0),
                      "kcal100": min(_num(x.get("kcal_per_100g")) or 0, 900.0),
                      "protein100": min(_num(x.get("protein_per_100g")) or 0, 100.0),
                      "carbs100": min(_num(x.get("carbs_per_100g")) or 0, 100.0),
                      "fat100": min(_num(x.get("fat_per_100g")) or 0, 100.0)})
    if not items:
        return None
    bal = d.get("balance") if isinstance(d.get("balance"), dict) else {}
    vit = [{"name": _txt(x.get("name"), 30), "dv": _num(x.get("dv_percent"))}
           for x in _list(d.get("vitamins"), 8) if isinstance(x, dict)]
    conf = _txt(d.get("confidence"), 10).lower()
    return {
        "is_food": True,
        "title": _txt(d.get("title"), 70) or "Your meal",
        "confidence": conf if conf in ("high", "medium", "low") else "",
        "portion_note": _txt(d.get("notes") or d.get("portion_note"), 180),
        "items": items, **scan_totals(items),
        "details": [{"label": _txt(x.get("label"), 30), "value": _txt(x.get("value"), 20)}
                    for x in _list(d.get("details"), 8) if isinstance(x, dict)],
        "ingredients": [_txt(x, 60) for x in _list(d.get("ingredients"), 12)],
        "vitamins": [v for v in vit if v["dv"] is not None],
        "allergens": [_txt(x, 30) for x in _list(d.get("allergens"), 9)],
        "insights": [_txt(x, 100) for x in _list(d.get("insights"), 5)],
        "balance": {"rating": _txt(bal.get("rating"), 40), "reason": _txt(bal.get("reason"), 140)},
    }


def scan_summary_html(d):
    esc = html.escape
    parts = [4 * d["protein"], 4 * d["carbs"], 9 * d["fat"]]
    total = sum(parts) or 1
    pct = [round(100 * x / total) for x in parts]
    grams = [d["protein"], d["carbs"], d["fat"]]
    names = ("Protein", "Carbs", "Fats")
    cards = "".join(
        f'<div class="macro"><span><u style="background:{col}"></u>{name}</span>'
        f'<b>{g} g</b><i>{pc}% of calories</i></div>'
        for name, g, pc, col in zip(names, grams, pct, MACRO_COLORS))
    bar = "".join(f'<i style="width:{pc}%;background:{col}"></i>' for pc, col in zip(pct, MACRO_COLORS))
    bal = d["balance"]
    rating = bal["rating"].lower()
    tone = "" if ("well" in rating or "excellent" in rating) else " fair"
    balance = (f'<div class="balance{tone}"><span>Meal balance</span><b>{esc(bal["rating"])}</b>'
               f'<p>{esc(bal["reason"])}</p></div>') if bal["rating"] else ""
    note = f'<p class="muted-note">{esc(d["portion_note"])}</p>' if d["portion_note"] else ""
    conf = d.get("confidence") or ""
    badge = f'<em class="{conf}">Confidence: {conf}</em>' if conf else ""
    if conf == "low":
        note += ('<p class="muted-note">Low confidence. Name the food in the box and scan again, '
                 "or correct the portions.</p>")
    items = ""
    if d.get("items"):
        rows = "".join(f'<div class="row"><span>{esc(i["name"])}</span>'
                       f'<b>{round(i["grams"])} g, {round(_item_kcal(i))} kcal</b></div>' for i in d["items"])
        items = f'<div class="scan-card items"><h4>What the AI saw</h4>{rows}</div>'
    return (f'<div class="scan-done"><b>Analysis complete</b><span>{esc(d["title"])}</span>{badge}</div>'
            f'<div class="kcal"><b>{d["calories"]}</b><span>kcal, estimated total</span></div>'
            f'<div class="macros">{cards}</div><div class="split">{bar}</div>{balance}{items}{note}')


def scan_detail_html(d):
    esc = html.escape

    def card(title, body):
        return f'<div class="scan-card"><h4>{title}</h4>{body or "<p>Nothing to show.</p>"}</div>'

    def rows(items):
        return "".join(f'<div class="row"><span>{esc(i)}</span></div>' for i in items)

    details = "".join(f'<div class="row"><span>{esc(x["label"])}</span><b>{esc(x["value"])}</b></div>'
                      for x in d["details"])
    vitamins = "".join(
        f'<div class="vit"><div class="row"><span>{esc(v["name"])}</span><b>{v["dv"]:g}% DV</b></div>'
        f'<div class="bar"><i style="width:{min(v["dv"], 100):g}%"></i></div></div>' for v in d["vitamins"])
    allergens = (f'<p>Possible: {esc(", ".join(d["allergens"]))}.</p>' if d["allergens"]
                 else "<p>None obvious in the photo.</p>")
    cant = ("<p>A photo can't show bacteria, freshness, hidden ingredients or exact portions. "
            "How the food was cooked and stored matters too.</p>")
    return ('<div class="scan-grid">'
            + card("Detailed nutrition", details) + card("Ingredients detected", rows(d["ingredients"]))
            + card("Vitamins and minerals", vitamins) + '</div><div class="scan-grid">'
            + card("Allergens to watch", allergens) + card("Other insights", rows(d["insights"]))
            + card("What a photo can't tell you", cant) + '</div>')


def scan_markdown(d):
    """Plain-text version of a scan, used as the saved body."""
    lines = [f"**{d['title']}**", "",
             f"About {d['calories']} kcal: protein {d['protein']} g, carbs {d['carbs']} g, fat {d['fat']} g."]
    if d["ingredients"]:
        lines += ["", "**Ingredients:** " + ", ".join(d["ingredients"])]
    if d["allergens"]:
        lines += ["", "**Possible allergens:** " + ", ".join(d["allergens"])]
    return "\n".join(lines)


def clear_profile():
    st.session_state.profile = {}
    user = st.session_state.get("user")
    if user:
        db.clear_profile(user["id"])


def save(kind, title, body, meta=None, item_id=None):
    """Save for a signed-in user. Returns the item id, or None if nothing was saved."""
    user = st.session_state.get("user")
    if not user:
        return None
    try:
        return db.save_item(user["id"], kind, title, body, meta, item_id)
    except Exception as e:
        st.toast(f"Couldn't save: {e}")
        return None


def hint_signin():
    if not st.session_state.get("user"):
        st.caption("Sign in from the sidebar to keep this in your history.")


def sign_in_as(user):
    st.session_state.user = user
    st.session_state.chat_id = None
    saved = db.get_profile(user["id"])
    if saved:
        st.session_state.profile = saved
    elif st.session_state.profile:
        db.save_profile(user["id"], st.session_state.profile)
    st.rerun()


@st.dialog("Your NutriFit account")
def account_dialog():
    st.caption("Sign in to keep your chats, food scans and plans between visits.")
    t_in, t_new = st.tabs(["Sign in", "Create account"])
    with t_in:
        with st.form("signin_form"):
            name = st.text_input("Name")
            pin = st.text_input("PIN", type="password", max_chars=8)
            submitted = st.form_submit_button("Sign in")
        if submitted:
            user, problem = db.verify_user(name, pin)
            if user:
                sign_in_as(user)
            st.error(problem)
    with t_new:
        with st.form("signup_form"):
            name = st.text_input("Choose a name", max_chars=30)
            pin = st.text_input("Choose a PIN (6 to 8 digits)", type="password", max_chars=8)
            submitted = st.form_submit_button("Create account")
        if submitted:
            created, result = db.create_user(name, pin)
            if created:
                sign_in_as(result)
            st.error(result)
        st.caption("Use a nickname rather than your full name. PINs can't be reset, so remember yours.")


def resume_chat(item_id, messages):
    st.session_state.messages = messages
    st.session_state.chat_id = item_id
    st.session_state.page = "AI Assistant"


# ---------- Sidebar ----------
st.sidebar.markdown(
    '<div class="brand"><div class="brand-mark"></div><div><div class="brand-name">NutriFit AI</div>'
    '<div class="brand-sub">Food, nutrition and movement</div></div></div>',
    unsafe_allow_html=True)

pages = ["Home", "My Profile", "AI Assistant", "Food Scanner", "Barcode Scanner", "Nutrition Centre",
         "Fitness Planner", "Allergy Centre", "Saved Information", "About and Privacy"]
page = st.sidebar.radio("Menu", pages, key="page", label_visibility="collapsed")
user = st.session_state.get("user")
if user:
    st.sidebar.markdown(f'<div class="side-note">Signed in as <b>{html.escape(user["name"])}</b></div>',
                        unsafe_allow_html=True)
    if st.sidebar.button("Sign out"):
        st.session_state.user = None
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.session_state.profile = {}
        st.rerun()
elif st.sidebar.button("Sign in to save your history"):
    account_dialog()
st.sidebar.markdown(
    '<div class="side-note">AI-generated guidance for learning. It is not medical advice.</div>',
    unsafe_allow_html=True)
st.sidebar.caption("Version 1.0")

# ---------- Barcode and packaged food ----------
OFF_FIELDS = ("code,product_name,generic_name,brands,quantity,serving_size,image_front_url,ingredients_text,"
              "nutriments,allergens_tags,traces_tags,nutriscore_grade,nova_group,labels,countries,additives_n,categories")
OFF_ALLERGENS = {
    "Peanuts": ["en:peanuts"], "Tree nuts": ["en:nuts"], "Milk": ["en:milk"], "Eggs": ["en:eggs"],
    "Fish": ["en:fish"], "Shellfish": ["en:crustaceans", "en:molluscs"], "Wheat": ["en:gluten"],
    "Soy": ["en:soybeans"], "Sesame": ["en:sesame-seeds"],
}
NOVA = {1: "Unprocessed or minimally processed", 2: "Processed culinary ingredient",
        3: "Processed food", 4: "Ultra-processed food"}
GRADE_COLORS = {"A": "#2E8B57", "B": "#7CB342", "C": "#F4C430", "D": "#EF8A17", "E": "#D9534F"}
BARCODE_MESSAGES = {
    "unreadable": "I couldn't read a barcode from that photo. Try again in good light with the bars filling "
                  "most of the frame, or type the number printed under the bars.",
    "invalid": "That doesn't look like a valid barcode number. Check the digits printed under the bars.",
    "not_found": "This product isn't in the Open Food Facts database yet. You can scan its ingredients "
                 "label with the Food Scanner instead.",
    "network": "I couldn't reach the product database just now. Please try again in a moment.",
}


def valid_gtin(code):
    """True if code is a barcode number (EAN-8, UPC-A, EAN-13 or GTIN-14) with a correct check digit."""
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    check = digits.pop()
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return (10 - total % 10) % 10 == check


def to_lookup_code(text):
    """Clean up a barcode number for lookup. Returns None if it isn't valid."""
    code = re.sub(r"\D", "", text or "")
    if not valid_gtin(code):
        return None
    if len(code) == 12:
        return "0" + code
    if len(code) == 14 and code.startswith("0"):
        return code[1:]
    return code


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, (list, tuple)):
        for item in x:
            yield from _strings(item)


def decode_barcode(photo):
    """Read a barcode from a photo with OpenCV. Returns a lookup code or None."""
    import cv2
    import numpy as np
    from io import BytesIO
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(BytesIO(photo.getvalue()))).convert("RGB")
    img.thumbnail((1600, 1600))
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    detector = cv2.barcode.BarcodeDetector() if hasattr(cv2, "barcode") else cv2.barcode_BarcodeDetector()
    # The detector is picky about how big the bars are in the frame, so try several sizes,
    # with and without a white border, and the photo turned on its side.
    for base in (gray, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)):
        for border in (1.0, 0):
            for factor in (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22, 0.15):
                view = base if factor == 1.0 else cv2.resize(base, None, fx=factor, fy=factor,
                                                             interpolation=cv2.INTER_AREA)
                if border:
                    m = int(max(view.shape) * border)
                    view = cv2.copyMakeBorder(view, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
                try:
                    found = detector.detectAndDecode(view)
                except Exception:
                    continue
                for text in _strings(found):
                    code = to_lookup_code(text)
                    if code:
                        return code
    return None


def read_digits_with_ai(photo):
    """Fallback: ask the vision model to read the number printed under the bars."""
    return to_lookup_code(vision_ai(
        "", "Read the number printed under the barcode in this photo. Reply with only the digits, "
            "or NONE if you cannot read them clearly.", photo, max_tokens=30, temperature=0, max_side=1600))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_product(code):
    """Look a product up in Open Food Facts. Returns its data, or None if it isn't listed."""
    url = f"https://world.openfoodfacts.org/api/v2/product/{code}.json?fields={OFF_FIELDS}"
    request = urllib.request.Request(url, headers={"User-Agent": "NutriFitAI/1.0 (student project)"})
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("product") if data.get("status") == 1 else None


def normalise_product(p, code):
    nut = p.get("nutriments") or {}

    def g(key):
        try:
            return float(nut[key])
        except (KeyError, TypeError, ValueError):
            return None

    kcal = g("energy-kcal_100g")
    if kcal is None and g("energy_100g") is not None:
        kcal = g("energy_100g") / 4.184          # the database also stores kilojoules
    image = str(p.get("image_front_url") or "")
    if not image.startswith(("https://images.openfoodfacts.org/", "https://static.openfoodfacts.org/")):
        image = ""
    grade = _txt(p.get("nutriscore_grade"), 1).upper()
    try:
        nova = int(p.get("nova_group"))
    except (TypeError, ValueError):
        nova = None
    tags = lambda key: [t for t in (p.get(key) or []) if isinstance(t, str)][:15]
    label = lambda tag: tag.split(":")[-1].replace("-", " ")
    return {
        "code": code,
        "name": _txt(p.get("product_name") or p.get("generic_name"), 90) or "Unnamed product",
        "brand": _txt(p.get("brands"), 60), "quantity": _txt(p.get("quantity"), 30),
        "serving": _txt(p.get("serving_size"), 30), "image": image,
        "ingredients": _txt(p.get("ingredients_text"), 1500),
        "per100": {"kcal": kcal, "protein": g("proteins_100g"), "carbs": g("carbohydrates_100g"),
                   "fat": g("fat_100g"), "sat_fat": g("saturated-fat_100g"), "sugar": g("sugars_100g"),
                   "fibre": g("fiber_100g"), "salt": g("salt_100g")},
        "kcal_serving": g("energy-kcal_serving"),
        "allergen_tags": tags("allergens_tags"), "trace_tags": tags("traces_tags"),
        "allergens": [label(t) for t in tags("allergens_tags")],
        "traces": [label(t) for t in tags("traces_tags")],
        "nutriscore": grade if grade in GRADE_COLORS else "",
        "nova": nova if nova in NOVA else None,
        "additives": p.get("additives_n") if isinstance(p.get("additives_n"), int) else None,
        "labels": _txt(p.get("labels"), 140), "countries": _txt(p.get("countries"), 100),
    }


def _fmt(v, unit="g"):
    return f"{round(v, 1):g} {unit}" if v is not None else "n/a"


def product_summary_html(d):
    esc = html.escape
    per = d["per100"]
    img = f'<img src="{esc(d["image"])}" alt="">' if d["image"] else ""
    sub = ", ".join(x for x in (d["brand"], d["quantity"]) if x)
    head = f'<div class="prod">{img}<div><h3>{esc(d["name"])}</h3><p>{esc(sub)}</p></div></div>'
    if per["kcal"] is not None:
        extra = (f", {round(d['kcal_serving'])} kcal per serving ({esc(d['serving'])})"
                 if d["kcal_serving"] is not None and d["serving"] else "")
        kcal = f'<div class="kcal"><b>{round(per["kcal"])}</b><span>kcal per 100 g{extra}</span></div>'
    else:
        kcal = '<div class="kcal"><span>Calories are not listed for this product.</span></div>'
    macro = [per["protein"], per["carbs"], per["fat"]]
    pct = [None, None, None]
    if None not in macro:
        parts = [4 * macro[0], 4 * macro[1], 9 * macro[2]]
        total = sum(parts) or 1
        pct = [round(100 * x / total) for x in parts]
    cards = "".join(
        f'<div class="macro"><span><u style="background:{col}"></u>{name}</span><b>{_fmt(v)}</b>'
        f'<i>{f"{pc}% of calories" if pc is not None else "per 100 g"}</i></div>'
        for name, v, pc, col in zip(("Protein", "Carbs", "Fats"), macro, pct, MACRO_COLORS))
    bar = ("".join(f'<i style="width:{pc}%;background:{col}"></i>' for pc, col in zip(pct, MACRO_COLORS))
           if None not in pct else "")
    scores = ""
    if d["nutriscore"]:
        color = GRADE_COLORS[d["nutriscore"]]
        ink = "#14332A" if d["nutriscore"] == "C" else "#fff"
        scores += (f'<div class="score"><i class="grade" style="background:{color};color:{ink}">'
                   f'{d["nutriscore"]}</i><span><b>Nutri-Score</b>A is best, E is lowest</span></div>')
    if d["nova"]:
        scores += (f'<div class="score"><i class="grade" style="background:#14332A;color:#fff">{d["nova"]}</i>'
                   f'<span><b>NOVA group</b>{esc(NOVA[d["nova"]])}</span></div>')
    return (head + kcal + f'<div class="macros">{cards}</div>' + (f'<div class="split">{bar}</div>' if bar else "")
            + (f'<div class="score-row">{scores}</div>' if scores else ""))


def product_detail_html(d):
    esc = html.escape
    per = d["per100"]

    def card(title, body):
        return f'<div class="scan-card"><h4>{title}</h4>{body}</div>'

    def tags(items, cls=""):
        return '<div class="tagrow">' + "".join(f'<span class="tag {cls}">{esc(x)}</span>' for x in items) + "</div>"

    rows = [("Saturated fat", per["sat_fat"]), ("Sugars", per["sugar"]), ("Fibre", per["fibre"]), ("Salt", per["salt"])]
    nutrition = "".join(f'<div class="row"><span>{label}</span><b>{_fmt(v)}</b></div>'
                        for label, v in rows if v is not None) or "<p>Not available.</p>"
    ingredients = (f'<p class="ingredients">{esc(d["ingredients"])}</p>' if d["ingredients"]
                   else "<p>Not listed in the database.</p>")
    allergens = (("<p><b>Contains</b></p>" + tags(d["allergens"], "warn")) if d["allergens"] else "<p>No allergens listed.</p>")
    if d["traces"]:
        allergens += "<p><b>May contain traces of</b></p>" + tags(d["traces"])
    origin = "".join(f'<div class="row"><span>{k}</span><b>{esc(v)}</b></div>'
                     for k, v in (("Labels", d["labels"]), ("Sold in", d["countries"]), ("Barcode", d["code"])) if v)
    processing = ((f'<div class="row"><span>Additives</span><b>{d["additives"]}</b></div>' if d["additives"] is not None else "")
                  + (f'<p>{esc(NOVA[d["nova"]])}.</p>' if d["nova"] else ""))
    about = ("<p>Product data comes from Open Food Facts, a community-built database. It can be incomplete "
             "or out of date, so check the pack. Values are per 100 g unless shown otherwise.</p>")
    return ('<div class="scan-grid">' + card("Nutrition per 100 g", nutrition) + card("Ingredients", ingredients)
            + card("Allergens", allergens) + '</div><div class="scan-grid">' + card("Labels and origin", origin)
            + card("Additives and processing", processing) + card("About this data", about) + "</div>")


def product_markdown(d, dates=None):
    per = d["per100"]
    lines = [f"**{d['name']}**" + (f" ({d['brand']})" if d["brand"] else ""), "",
             f"Per 100 g: {_fmt(per['kcal'], 'kcal')}, protein {_fmt(per['protein'])}, "
             f"carbs {_fmt(per['carbs'])}, fat {_fmt(per['fat'])}."]
    if d["ingredients"]:
        lines += ["", "**Ingredients:** " + d["ingredients"]]
    if d["allergens"]:
        lines += ["", "**Contains:** " + ", ".join(d["allergens"])]
    for label, key in (("Use by / best before", "expiry"), ("Made on", "made")):
        if dates and dates.get(key):
            lines += ["", f"**{label}:** {dates[key]}"]
    return "\n".join(lines)


def product_allergy_check(d):
    """Compare a product with the allergies on the user's profile."""
    allergies = pref("allergies")
    if not allergies:
        return
    contains, traces = [], []
    for a in allergies:
        wanted = OFF_ALLERGENS.get(a, [])
        words = [w for w in ALLERGEN_WORDS.get(a, []) if re.search(rf"\b{re.escape(w)}s?\b", d["ingredients"].lower())]
        if any(t in d["allergen_tags"] for t in wanted) or words:
            contains.append(a)
        elif any(t in d["trace_tags"] for t in wanted):
            traces.append(a)
    if contains:
        st.error("**This product lists your allergens:** " + ", ".join(contains) + ". Do not eat it if it is not safe for you.")
    elif traces:
        st.warning("**May contain traces of:** " + ", ".join(traces) + ". Check the pack carefully.")
    else:
        st.info("None of your listed allergens appear in this product's data, but the data can be incomplete. "
                "Always check the pack.")


# ---------- Dates on the pack ----------
def parse_date(text):
    """YYYY-MM-DD, or YYYY-MM (treated as the last day of that month). Returns a date or None."""
    m = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", str(text or "").strip())
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    try:
        if m.group(3):
            return date(year, month, int(m.group(3)))
        return date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    except ValueError:
        return None


def date_status(expiry, today):
    days = (expiry - today).days
    if days < 0:
        return "expired", f"Expired {abs(days)} day{'s' if abs(days) != 1 else ''} ago"
    if days == 0:
        return "today", "Expires today"
    if days <= 3:
        return "soon", f"Expires in {days} day{'s' if days != 1 else ''}"
    return "ok", f"{days} days left"


def read_dates_with_ai(photo):
    """Ask the vision model to read printed dates. Returns a dict (or None)."""
    system = ("You read printed dates on food packaging. Reply with ONE JSON object only, no other text: "
              '{"use_by": null, "best_before": null, "manufactured": null, "batch": null, "text_seen": ""}. '
              "Give dates as YYYY-MM-DD, or YYYY-MM if no day is printed. Two-digit years mean 20xx. If the "
              "order of day and month is unclear, use null and put the printed text in text_seen. "
              "text_seen is the exact date text you can read. Never guess.")
    return parse_scan(vision_ai(system, "Read the dates on this pack.", photo, max_tokens=300,
                                temperature=0, max_side=1600, json_mode=True))


# ---------- Home content ----------
TIPS = [
    "Aim to fill about half your plate with vegetables and fruit at most meals.",
    "Sip water through the day. It helps you stay focused and energised.",
    "Beans, eggs, fish and groundnuts are all good sources of protein.",
    "A short walk after a meal is an easy way to add movement to your day.",
    "A colourful plate usually means a wider mix of vitamins and minerals.",
    "Whole grains like oats and brown rice help you feel full for longer.",
    "On food labels, ingredients are listed from the largest amount to the smallest.",
    "Warm up for five minutes before exercise and stretch gently afterwards.",
    "Good sleep helps your mood, your appetite and how well you recover.",
    "Try one new fruit or vegetable this week.",
]
ICONS = {
    "scan": '<svg viewBox="0 0 24 24"><path d="M4 8h3l1.5-2.5h7L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>',
    "ask": '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H10l-4 3.5V16H4z"/></svg>',
    "fit": '<svg viewBox="0 0 24 24"><path d="M3 12h4l2.5-6 4 12 2.5-6H21"/></svg>',
    "nutri": '<svg viewBox="0 0 24 24"><path d="M5 19C5 9 11 4 20 4c0 9-5 15-15 15z"/><path d="M5 19l9-9"/></svg>',
    "allergy": '<svg viewBox="0 0 24 24"><path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.4"/></svg>',
    "saved": '<svg viewBox="0 0 24 24"><path d="M6 4h12v16l-6-4-6 4z"/></svg>',
    "barcode": '<svg viewBox="0 0 24 24"><path d="M4 5v14M7 5v14M11 5v14M14 5v14M17 5v14M20 5v14"/></svg>',
    "profile": '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 20c1-4 4-6 8-6s7 2 8 6"/></svg>',
    "about": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.4"/></svg>',
}
LEAF = ('<svg class="leaf {cls}" viewBox="0 0 24 24"><path d="M4 20C4 10 10 4 21 4c0 10-6 16-17 16z" fill="#4DB37C"/>'
        '<path d="M4 20L14 10" stroke="#14332A" stroke-width="1.2" fill="none" opacity=".35"/></svg>')
HERO_ART = (
    '<div class="hero-art" aria-hidden="true">'
    '<svg viewBox="0 0 260 260"><ellipse cx="130" cy="238" rx="86" ry="9" fill="#000" opacity=".2"/>'
    '<circle cx="130" cy="126" r="112" fill="#F5F6EF"/>'
    '<circle cx="130" cy="126" r="88" fill="#fff" stroke="#E4EADB" stroke-width="3"/>'
    '<ellipse cx="92" cy="156" rx="34" ry="26" fill="#F3E7BF"/>'
    '<g fill="#fff" opacity=".8"><ellipse cx="80" cy="150" rx="5" ry="2.2" transform="rotate(-25 80 150)"/>'
    '<ellipse cx="98" cy="146" rx="5" ry="2.2" transform="rotate(15 98 146)"/>'
    '<ellipse cx="90" cy="162" rx="5" ry="2.2"/><ellipse cx="108" cy="160" rx="5" ry="2.2" transform="rotate(-15 108 160)"/></g>'
    '<g fill="#2E8B57"><circle cx="98" cy="86" r="22"/><circle cx="124" cy="76" r="19"/>'
    '<circle cx="78" cy="102" r="16"/><circle cx="120" cy="98" r="17"/></g>'
    '<g fill="#256F46"><circle cx="92" cy="80" r="6"/><circle cx="128" cy="72" r="5"/><circle cx="116" cy="102" r="5"/></g>'
    '<g transform="rotate(-18 170 142)"><rect x="136" y="122" width="68" height="40" rx="20" fill="#C98B4B"/>'
    '<path d="M152 132v20M168 130v24M184 132v20" stroke="#8A5A2B" stroke-width="4" stroke-linecap="round"/></g>'
    '<circle cx="176" cy="82" r="18" fill="#D9534F"/><circle cx="170" cy="76" r="5" fill="#fff" opacity=".35"/>'
    '<path d="M176 62c-4 0-8 3-9 6 4-1 7 0 9 2 2-2 5-3 9-2-1-3-5-6-9-6z" fill="#2E8B57"/>'
    '<g fill="#F4A93B"><circle cx="150" cy="182" r="12"/><circle cx="176" cy="176" r="11"/><circle cx="124" cy="186" r="10"/></g>'
    '<g fill="#F9C76B"><circle cx="150" cy="182" r="5"/><circle cx="176" cy="176" r="4.5"/><circle cx="124" cy="186" r="4"/></g>'
    '</svg>' + LEAF.format(cls="l1") + LEAF.format(cls="l2") + LEAF.format(cls="l3") + '</div>')
HOW = [("Tell us about you", "Set up a short profile so answers and plans fit you. It's optional."),
       ("Scan, ask or plan", "Photograph a meal, chat with the assistant or build a weekly activity plan."),
       ("Keep what helps", "Sign in to save your chats, scans and plans and come back to them any time.")]

# ---------- Home ----------
if page == "Home":
    me = st.session_state.get("user")
    heading = f"Welcome back, {html.escape(me['name'])}" if me else "What's on your plate?"
    with st.container(key="hero"):
        text_col, art_col = st.columns([1.5, 1], gap="large", vertical_alignment="center")
        with text_col:
            st.markdown(f"<h1>{heading}</h1><p>Snap a meal, ask a question or plan your week. Clear, friendly "
                        "guidance for eating well and staying active.</p>", unsafe_allow_html=True)
            b1, b2 = st.columns(2, gap="small")
            with b1, st.container(key="hero_scan"):
                st.button("Scan a meal", key="hero_b1", on_click=go_to_page, args=("Food Scanner",))
            with b2, st.container(key="hero_ask"):
                st.button("Ask a question", key="hero_b2", on_click=go_to_page, args=("AI Assistant",))
        with art_col:
            st.markdown(HERO_ART, unsafe_allow_html=True)

    tip = TIPS[date.today().toordinal() % len(TIPS)]
    st.markdown(f'<div class="tip"><span>Tip of the day</span><p>{tip}</p></div>', unsafe_allow_html=True)

    if not st.session_state.profile:
        with st.container(key="panel_setup"):
            st.markdown("**Make NutriFit yours.** Add your age group, goals and allergies once, "
                        "and every page will use them.")
            st.button("Set up my profile", on_click=go_to_page, args=("My Profile",))
        st.write("")
    st.markdown("### What would you like to do?")

    tiles = [
        ("Food Scanner", "Photograph a meal or drink and see what it likely contains.", "Scan food", "scan"),
        ("Barcode Scanner", "Scan packaged food for nutrition, ingredients, allergens and dates.", "Scan a barcode", "barcode"),
        ("AI Assistant", "Get clear answers about food, nutrition and healthy habits.", "Ask a question", "ask"),
        ("Fitness Planner", "Build a weekly activity plan around your goal and schedule.", "Plan my week", "fit"),
        ("Nutrition Centre", "Learn what proteins, vitamins, fibre and more do for you.", "Browse nutrients", "nutri"),
        ("Allergy Centre", "See the common food allergens and where they often hide.", "View allergens", "allergy"),
        ("Saved Information", "Review your saved chats, scans, products and plans.", "Open saved items", "saved"),
        ("My Profile", "Set your age group, goals and allergies once for every page.", "Edit my profile", "profile"),
    ]
    for r in range(0, len(tiles), 4):
        cols = st.columns(4, gap="medium")
        for col, (name, text, btn, key) in zip(cols, tiles[r:r + 4]):
            with col:
                with st.container(key=f"card_{key}"):
                    st.markdown(f'<div class="icon">{ICONS[key]}</div><h3>{name}</h3><p>{text}</p>',
                                unsafe_allow_html=True)
                    st.button(btn, key=f"btn_{key}", on_click=go_to_page, args=(name,))
        st.write("")

    st.markdown("### How it works")
    st.markdown('<div class="how">' + "".join(
        f"<div><b>{i}</b><h4>{t}</h4><p>{d}</p></div>" for i, (t, d) in enumerate(HOW, 1)) + "</div>",
        unsafe_allow_html=True)
    st.markdown('<p class="home-foot">NutriFit AI is a learning tool, not medical advice.</p>',
                unsafe_allow_html=True)

# ---------- My Profile ----------
elif page == "My Profile":
    header("My Profile", "Tell NutriFit a little about you so answers, plans and food scans fit you.")

    if not st.session_state.get("user"):
        st.info("You're not signed in, so your profile lasts for this visit only. "
                "Sign in from the sidebar to keep it.")

    with st.form("profile_form"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            age_group = st.selectbox("Age group", AGE_GROUPS,
                                     index=AGE_GROUPS.index(pick_valid(AGE_GROUPS, "age_group", "18-39")))
            goal = st.selectbox("Main goal", GOALS, index=GOALS.index(pick_valid(GOALS, "goal", "Both")))
            level = st.select_slider("Activity level", LEVELS, value=pick_valid(LEVELS, "level", "Beginner"))
        with c2:
            diet = st.multiselect("Eating preferences", DIETS,
                                  default=[d for d in pref("diet", []) if d in DIETS],
                                  placeholder="None")
            allergies = st.multiselect("Food allergies", list(ALLERGEN_WORDS),
                                       default=[a for a in pref("allergies", []) if a in ALLERGEN_WORDS],
                                       placeholder="None",
                                       help="Used to tailor answers and flag these in food scans.")
        saved = st.form_submit_button("Save profile")

    if saved:
        st.session_state.profile = {"age_group": age_group, "goal": goal, "level": level,
                                    "diet": diet, "allergies": allergies}
        user = st.session_state.get("user")
        if user:
            db.save_profile(user["id"], st.session_state.profile)
            st.toast("Profile saved to your account")
        else:
            st.toast("Profile saved for this visit")

    section("Where your profile is used")
    st.markdown(
        '<div class="how">'
        "<div><h4>AI Assistant</h4><p>Answers are tailored to you and avoid foods with your allergens.</p></div>"
        "<div><h4>Food Scanner</h4><p>Scans get an allergy check and a warning if they mention your allergens.</p></div>"
        "<div><h4>Planner and Allergy Centre</h4><p>Your age group, level and allergies are filled in for you.</p></div>"
        "</div>", unsafe_allow_html=True)
    st.caption("Allergies are health information. NutriFit only uses them to tailor this app, "
               "and signed-in profiles are stored only in your account.")
    if st.session_state.profile:
        st.button("Clear my profile", on_click=clear_profile)

# ---------- AI Assistant ----------
elif page == "AI Assistant":
    header("AI Assistant", "Ask about nutrition, food and healthy living.")

    pending = st.session_state.pop("pending", None)
    if not st.session_state.messages and not pending:
        starter_qs = ["What are good sources of iron?",
                      "How much water should I drink on a hot day?",
                      "What are healthy snack ideas for school?"]
        st.markdown("**Not sure where to start?** Try one of these.")
        with st.container(key="starters"):
            for col, q in zip(st.columns(3, gap="medium"), starter_qs):
                col.button(q, key=f"starter_{q}",
                           on_click=lambda q=q: st.session_state.update(pending=q))
        st.write("")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask NutriFit AI anything...", max_chars=500) or pending
    if question and limit_ok():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = groq_client.chat.completions.create(
                        model=AI_MODEL,
                        messages=[
                            {"role": "system", "content": (
                                "You are NutriFit AI, an educational assistant that "
                                "provides reliable, age-appropriate information about "
                                "food, nutrition, allergies and physical activity. "
                                "Never diagnose medical conditions or prescribe diets. "
                                "Encourage users to consult qualified health "
                                "professionals for personal medical concerns.") + profile_context()},
                            *st.session_state.messages[-12:],
                        ],
                        temperature=0.4,
                        max_tokens=700,
                    )
                    answer = clean(response.choices[0].message.content)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_id = save(
                        "chat", st.session_state.messages[0]["content"][:60],
                        json.dumps(st.session_state.messages), item_id=st.session_state.chat_id)
                except Exception as e:
                    st.error(f"The assistant couldn't answer: {friendly(e)}.")

    if st.session_state.messages and st.button("Start a new chat"):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.rerun()

# ---------- Food Scanner ----------
elif page == "Food Scanner":
    header("Food Scanner", "Take a photo or upload one to see what your food or drink contains.")
    st.markdown('<div class="steps"><span><b>1</b>Take or upload a photo</span>'
                '<span><b>2</b>Select Analyse food</span><span><b>3</b>Read the results</span></div>',
                unsafe_allow_html=True)
    st.caption("For the best results, use good light and keep the whole meal in view.")

    mode = st.segmented_control(
        "How do you want to add a photo?", ["Take a photo", "Upload a photo"],
        default="Upload a photo", label_visibility="collapsed") or "Upload a photo"

    left, right = st.columns([1, 1.25], gap="large")
    with left, st.container(key="panel_scan_in"):
        if mode == "Take a photo":
            photo = st.camera_input("Take a picture of your food",
                                    help="Allow camera access when your browser asks.")
        else:
            photo = st.file_uploader("Upload a food image", type=["png", "jpg", "jpeg"])
            if photo:
                st.image(photo, use_container_width=True)
        hint = st.text_input("What is it? (optional)", max_chars=120,
                             placeholder="e.g. waakye with fish, gari and shito",
                             help="Naming the food helps the AI a lot, especially for local dishes.")
        size = st.selectbox("Portion size (optional)", ["Not sure", "Small", "Medium", "Large"])
        go = st.button("Analyse food") if photo else False

    key = hashlib.md5(photo.getvalue()).hexdigest() if photo else None
    if go and limit_ok():
        with right:
            try:
                with st.spinner("Analysing your food..."):
                    data = normalise_scan(parse_scan(run_scan(photo, hint, size)))
                st.session_state.scan = {"key": key, "data": data, "id": time.time()}
            except Exception as e:
                st.session_state.scan = {"key": key, "data": None, "error": friendly(e)}

    scan = st.session_state.get("scan")
    ready = bool(photo and scan and scan["key"] == key)
    shown = None
    with right:
        if not photo:
            st.info("Add a photo to get started.")
        elif not ready:
            st.info("Select Analyse food to see the results here.")
        elif scan["data"] is None and scan.get("error"):
            st.error(f"The scanner couldn't analyse this photo: {scan['error']}.")
        elif scan["data"] is None:
            st.warning("The scanner couldn't read this photo clearly. Try again with better light "
                       "or a closer shot.")
        elif not scan["data"].get("is_food"):
            st.info("This doesn't look like food or a drink. Try another photo.")
        else:
            d = scan["data"]
            if d.get("items"):
                with st.expander("Check the portions", expanded=True):
                    st.caption("Change the grams if the AI guessed wrong. The totals update.")
                    table = pd.DataFrame({"Food": [i["name"] for i in d["items"]],
                                          "Grams": [round(i["grams"]) for i in d["items"]]})
                    edited = st.data_editor(
                        table, hide_index=True, disabled=["Food"], use_container_width=True,
                        key=f"portions_{scan.get('id')}",
                        column_config={"Grams": st.column_config.NumberColumn("Grams", min_value=0,
                                                                              max_value=2000, step=5)})
                d = apply_portions(d, list(edited["Grams"]))
            shown = d
            st.markdown(scan_summary_html(d), unsafe_allow_html=True)
            allergy_alert(" ".join([d["title"], *d["ingredients"], *d["allergens"]]))

    if ready and scan["data"] and scan["data"].get("is_food"):
        d = shown or scan["data"]
        st.markdown(scan_detail_html(d), unsafe_allow_html=True)
        st.caption("All values are estimates from a photo, not lab measurements. "
                   "Actual values depend on ingredients and portion size.")
        if vision_note():
            st.caption(vision_note())
        if go:
            if save("scan", d["title"], scan_markdown(d), d):
                st.toast("Saved to your history")
            else:
                hint_signin()

# ---------- Barcode Scanner ----------
elif page == "Barcode Scanner":
    header("Barcode Scanner", "Scan a packaged food's barcode to see its nutrition, ingredients and allergens.")
    st.markdown('<div class="steps"><span><b>1</b>Scan or type the barcode</span>'
                '<span><b>2</b>Review the product</span><span><b>3</b>Check the dates</span></div>',
                unsafe_allow_html=True)
    st.caption("Hold the barcode flat and steady in good light, with the bars filling most of the frame.")

    mode = st.segmented_control(
        "How do you want to enter the barcode?", ["Take a photo", "Upload a photo", "Type the number"],
        default="Upload a photo", label_visibility="collapsed", key="bc_mode") or "Upload a photo"

    left, right = st.columns([1, 1.25], gap="large")
    photo, typed = None, ""
    with left, st.container(key="panel_bc_in"):
        if mode == "Take a photo":
            photo = st.camera_input("Take a picture of the barcode", key="bc_cam")
        elif mode == "Upload a photo":
            photo = st.file_uploader("Upload a photo of the barcode", type=["png", "jpg", "jpeg"], key="bc_up")
            if photo:
                st.image(photo, use_container_width=True)
        else:
            typed = st.text_input("Barcode number", max_chars=14, placeholder="e.g. 5449000000996",
                                  help="The digits printed under the bars.")
        go = st.button("Look up product") if (photo or typed.strip()) else False

    if go:
        with right:
            with st.spinner("Looking up the product..."):
                result = {"code": None, "product": None, "error": None}
                try:
                    if mode == "Type the number":
                        result["code"] = to_lookup_code(typed)
                        result["error"] = None if result["code"] else "invalid"
                    else:
                        result["code"] = decode_barcode(photo)
                        if not result["code"] and limit_ok():
                            result["code"] = read_digits_with_ai(photo)
                        result["error"] = None if result["code"] else "unreadable"
                    if result["code"]:
                        raw = fetch_product(result["code"])
                        if raw:
                            result["product"] = normalise_product(raw, result["code"])
                        else:
                            result["error"] = "not_found"
                except Exception:
                    result["error"] = "network"
        st.session_state.barcode = result
        if result["product"]:      # start the date checker fresh for a new product
            for k in ("exp_date", "made_date", "date_note"):
                st.session_state[k] = None

    result = st.session_state.get("barcode")
    with right:
        if not result:
            st.info("Scan or type a barcode to get started.")
        elif result.get("product"):
            st.markdown(product_summary_html(result["product"]), unsafe_allow_html=True)
            product_allergy_check(result["product"])
        else:
            st.warning(BARCODE_MESSAGES.get(result.get("error"), BARCODE_MESSAGES["network"]))

    if result and result.get("product"):
        d = result["product"]
        st.markdown(product_detail_html(d), unsafe_allow_html=True)

        section("Check the dates")
        st.caption("A barcode doesn't contain the expiry or manufacturing date. Those are printed on each "
                   "pack and differ between batches, so add them from the pack.")
        st.session_state.setdefault("exp_kind", "Best before")
        with st.expander("Read the dates from a photo of the pack"):
            if mode == "Take a photo":
                date_photo = st.camera_input("Photo of the date on the pack", key="date_cam")
            else:
                date_photo = st.file_uploader("Photo of the date on the pack", type=["png", "jpg", "jpeg"],
                                              key="date_up")
            if date_photo and st.button("Read the dates") and limit_ok():
                try:
                    with st.spinner("Reading the dates..."):
                        found = read_dates_with_ai(date_photo) or {}
                    use_by, best, made_on = (parse_date(found.get(k)) for k in ("use_by", "best_before", "manufactured"))
                    seen = _txt(found.get("text_seen"), 120)
                    if use_by or best or made_on:
                        st.session_state.exp_kind = "Use by" if use_by else "Best before"
                        st.session_state.exp_date = use_by or best
                        st.session_state.made_date = made_on
                        st.session_state.date_note = seen
                    else:
                        st.warning("I couldn't read a clear date" + (f" (I could see: {seen})" if seen else "")
                                   + ". Try a closer photo, or enter it below.")
                except Exception as e:
                    st.error(f"Couldn't read the dates: {friendly(e)}.")

        c1, c2, c3 = st.columns(3, gap="medium")
        kind = c1.radio("The expiry date on the pack is a", ["Use by", "Best before"], key="exp_kind")
        expiry = c2.date_input("Expiry date", value=None, key="exp_date", format="YYYY-MM-DD",
                               min_value=date(2000, 1, 1), max_value=date(2100, 12, 31))
        made = c3.date_input("Manufacturing date (if shown)", value=None, key="made_date", format="YYYY-MM-DD",
                             min_value=date(2000, 1, 1), max_value=date(2100, 12, 31))
        if st.session_state.get("date_note"):
            st.caption(f"Read from the photo: {st.session_state.date_note}. Please check it against the pack.")

        today = date.today()
        if expiry:
            level, text = date_status(expiry, today)
            if level == "expired":
                st.error(text + (". Don't eat food that is past its use-by date." if kind == "Use by" else
                                 ". It is past its best-before date, so quality may have dropped. Check it "
                                 "carefully and throw it out if in doubt."))
            elif level in ("today", "soon"):
                st.warning(text + ".")
            else:
                st.success(text + ".")
        if made:
            age = (today - made).days
            st.info(f"Made on {made:%d %b %Y} ({age} days ago)." if age >= 0
                    else "The manufacturing date is in the future. Please check it.")
        if expiry and made and expiry >= made:
            st.caption(f"Shelf life from manufacture to expiry: {(expiry - made).days} days.")
        st.caption("Use by is a safety date: don't eat food after it. Best before is a quality date: food is "
                   "often still safe a little after it but may not be at its best. If in doubt, throw it out.")

        if st.button("Save this product to my history"):
            dates = {"kind": kind, "expiry": expiry.isoformat() if expiry else "",
                     "made": made.isoformat() if made else ""}
            if save("product", d["name"], product_markdown(d, dates), {**d, "dates": dates}):
                st.toast("Saved to your history")
            else:
                hint_signin()

# ---------- Nutrition Centre ----------
elif page == "Nutrition Centre":
    header("Nutrition Centre", "What each nutrient does and where to find it.")

    nutrients = {
        "Proteins": ("Build and repair muscle, skin and enzymes.", "Beans, fish, eggs, chicken, groundnuts, tofu"),
        "Carbohydrates": ("The main fuel for your brain and body.", "Rice, yam, plantain, oats, whole-grain bread"),
        "Fats": ("Store energy and help absorb vitamins A, D, E and K.", "Avocado, nuts, seeds, oily fish, vegetable oils"),
        "Vitamins": ("Support immunity, vision, bones and energy release.", "Fruit, leafy greens, carrots, eggs, dairy"),
        "Minerals": ("Strengthen bones and carry oxygen, like calcium and iron.", "Milk, leafy greens, beans, fish, meat"),
        "Fibre": ("Keeps digestion regular and helps you feel full.", "Whole grains, vegetables, fruit, legumes"),
        "Water": ("Regulates temperature and moves nutrients around the body.", "Water, fruit, soups, milk"),
    }
    for tab, (name, (does, sources)) in zip(st.tabs(list(nutrients)), nutrients.items()):
        with tab:
            st.markdown(f"<h3>{name}</h3>", unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="medium")
            c1.markdown(info("What it does", does), unsafe_allow_html=True)
            c2.markdown(info("Where to find it", sources), unsafe_allow_html=True)

    st.write("")
    st.caption("Want more detail on a nutrient or a food?")
    st.button("Ask the AI Assistant", on_click=go_to_page, args=("AI Assistant",))

# ---------- Allergy Centre ----------
elif page == "Allergy Centre":
    header("Allergy Centre", "Learn the common food allergens, read labels with confidence and check a dish.")

    st.warning(
        "**Get emergency help straight away** if someone has trouble breathing, swelling of the face, "
        "lips or throat, widespread hives, dizziness or fainting after eating. Call your local "
        "emergency number. This page is educational and cannot diagnose an allergy.")

    allergens = {
        "Peanuts": {
            "found": "Groundnut paste, satay and some stews, sauces, baked goods, snack mixes",
            "labels": "Groundnut, peanut flour, arachis oil, monkey nuts, beer nuts",
            "tip": "Peanuts are legumes, not tree nuts, but both can be in the same shared kitchen or factory.",
        },
        "Tree nuts": {
            "found": "Nut butters, pesto, chocolate spreads, granola, desserts, marzipan",
            "labels": "Almond, cashew, walnut, pecan, hazelnut, pistachio, brazil nut, macadamia, praline, nougat",
            "tip": "Many people are advised to avoid all tree nuts until a doctor says which ones are safe.",
        },
        "Milk": {
            "found": "Cheese, butter, yoghurt, ice cream, chocolate, baked goods, creamy sauces",
            "labels": "Casein, caseinate, whey, lactose, ghee, buttermilk, milk solids",
            "tip": "Milk allergy is an immune reaction. Lactose intolerance is a digestive problem and is different.",
        },
        "Eggs": {
            "found": "Cakes, mayonnaise, pasta, batter and coatings, meringue, some sauces",
            "labels": "Egg white, egg yolk, albumin, ovalbumin, lysozyme, meringue, mayonnaise",
            "tip": "Eggs often hide in baked goods and in the coating on fried foods.",
        },
        "Fish": {
            "found": "Stews, stock, fish sauce, Worcestershire sauce, Caesar dressing, some soups",
            "labels": "Anchovy, cod, tilapia, tuna, salmon, fish gelatine, Worcestershire sauce",
            "tip": "A fish allergy is separate from a shellfish allergy, though some people have both.",
        },
        "Shellfish": {
            "found": "Seafood stock, stir-fries, paella, shrimp paste, mixed seafood dishes",
            "labels": "Shrimp, prawn, crab, lobster, crayfish; also clams, mussels, oysters, squid",
            "tip": "Crustaceans (like prawns) and molluscs (like clams) are different groups. Some people react to one or both.",
        },
        "Wheat": {
            "found": "Bread, pasta, flour, biscuits, sauces thickened with flour, soy sauce, beer",
            "labels": "Wheat flour, semolina, spelt, durum, couscous, bulgur, farina",
            "tip": "Wheat allergy is not the same as coeliac disease, which is a reaction to gluten.",
        },
        "Soy": {
            "found": "Tofu, soy sauce, miso, tempeh, many processed and packaged foods",
            "labels": "Soya, edamame, tempeh, textured vegetable protein, soy lecithin",
            "tip": "Soy is common in processed foods, so ingredient lists matter even for foods that don't seem to need it.",
        },
        "Sesame": {
            "found": "Tahini, hummus, bread and bun toppings, falafel, dressings, oils",
            "labels": "Tahini, sesame oil, benne seed, gingelly oil",
            "tip": "Sesame can also appear in spice mixes and breads without being obvious.",
        },
    }
    names = list(allergens)
    learn, labels_tab, check = st.tabs(["Allergens", "Reading labels", "Check a dish"])

    with learn:
        pick = st.pills("Choose an allergen", names, default=names[0],
                        label_visibility="collapsed") or names[0]
        st.write("")
        a = allergens[pick]
        c1, c2, c3 = st.columns(3, gap="medium")
        c1.markdown(info("Often found in", a["found"]), unsafe_allow_html=True)
        c2.markdown(info("Names on labels", a["labels"]), unsafe_allow_html=True)
        c3.markdown(info("Good to know", a["tip"]), unsafe_allow_html=True)

    with labels_tab:
        section("Four habits that help")
        st.markdown(
            "- **Read every label, every time.** Recipes can change without warning.\n"
            "- **Check the whole ingredient list,** not only the Contains line, because labelling "
            "rules differ between countries.\n"
            "- **Take \"may contain\" seriously.** It means the food could pick up traces from "
            "shared equipment.\n"
            "- **Ask when eating out.** Staff can tell you what is in a dish and how it is prepared.")
        section("Other names to look for")
        rows = "\n".join(f"| {n} | {a['labels']} |" for n, a in allergens.items())
        st.markdown("| Allergen | Names on labels |\n|---|---|\n" + rows)

    with check:
        st.caption("Describe a dish and the AI will point out where the allergens you choose "
                   "commonly appear. It cannot see the real recipe, so always check the label "
                   "or ask the cook.")
        dish = st.text_input("Dish or ingredients",
                             placeholder="e.g. chicken satay with peanut sauce and rice",
                             max_chars=200)
        watch = st.multiselect("Allergens to look out for", names,
                               default=[a for a in pref("allergies", []) if a in names],
                               placeholder="Leave empty to check all nine")

        if st.button("Check for allergens"):
            if not dish.strip():
                st.info("Describe a dish or list its ingredients first.")
            elif limit_ok():
                try:
                    with st.spinner("Checking the dish..."):
                        response = groq_client.chat.completions.create(
                            model=AI_MODEL,
                            messages=[
                                {"role": "system", "content": (
                                    "You are NutriFit AI Allergy Guide. You help people spot where "
                                    "common food allergens may appear in a dish. You cannot see the "
                                    "real recipe, so never say a dish is safe or free of an allergen. "
                                    "For each allergen say Likely, Possible or Unlikely and give a "
                                    "short reason. Mention common hidden sources such as sauces, "
                                    "stock, oils and batter. Never diagnose allergies or give "
                                    "treatment advice. Remind the user to check labels and ask the cook.")},
                                {"role": "user", "content": (
                                    f"Dish or ingredients: {dish}\n"
                                    f"Allergens to check: {', '.join(watch) or ', '.join(names)}\n\n"
                                    "Format: a markdown table with columns Allergen, Chance, Why, "
                                    "one line per cell and no HTML tags. Then one short line on "
                                    "hidden sources to ask about.")},
                            ],
                            temperature=0.2,
                            max_tokens=700,
                        )
                    st.session_state.allergy_check = clean(response.choices[0].message.content)
                    if save("allergy", dish.strip()[:60], st.session_state.allergy_check,
                            {"allergens": watch or "all nine"}):
                        st.toast("Saved to your history")
                    else:
                        hint_signin()
                except Exception as e:
                    st.error(f"Couldn't check this dish: {friendly(e)}.")

        result = st.session_state.get("allergy_check")
        if result:
            section("Results")
            st.markdown(result)
            st.caption("A general guide only, not a guarantee. If you have an allergy, "
                       "confirm with the label or the person who prepared the food.")

# ---------- Fitness Planner ----------
elif page == "Fitness Planner":
    header("Fitness Planner", "Choose a goal and get a simple weekly plan built around your week.")

    activities = {
        "General fitness": "Brisk walking, cycling, swimming, dancing",
        "Strength": "Squats, push-ups, planks, resistance bands",
        "Cardiovascular endurance": "Jogging, skipping rope, football, swimming laps",
        "Flexibility": "Stretching, yoga, mobility routines",
        "Balance and coordination": "Single-leg stands, tai chi, dance, skipping",
    }

    with st.container(key="panel_inputs"):
        st.markdown("### Your details")
        left, right = st.columns(2, gap="large")
        with left:
            goal = st.selectbox("Goal", list(activities), help="What you want the plan to focus on.")
            st.markdown(info("Activities to try", activities[goal]), unsafe_allow_html=True)
            st.write("")
            level = st.select_slider("Current level", LEVELS, value=pick_valid(LEVELS, "level", "Beginner"),
                                     help="A plan that starts easy is easier to stick to.")
            age_group = st.selectbox("Age group", AGE_GROUPS,
                                     index=AGE_GROUPS.index(pick_valid(AGE_GROUPS, "age_group", "18-39")),
                                     help="Used only to keep the activities age-appropriate.")
        with right:
            days = st.slider("Training days per week", 1, 6, 3,
                             help="The other days are for rest or light movement.")
            minutes = st.select_slider("Time per session", options=[15, 20, 30, 45, 60], value=30,
                                       format_func=lambda m: f"{m} min")
            equipment = st.multiselect(
                "What you have access to",
                ["Nothing (bodyweight only)", "Skipping rope", "Resistance bands",
                 "Dumbbells", "Gym", "Open space or pitch", "Pool"],
                default=["Nothing (bodyweight only)"],
            )
            notes = st.text_input("Anything to avoid or work around? (optional)",
                                  placeholder="e.g. no running, tired knees", max_chars=100)
    st.write("")

    if st.button("Build my plan") and limit_ok():
        request = (
            f"Create a one-week plan.\n"
            f"Goal: {goal}\nLevel: {level}\nAge group: {age_group}\n"
            f"Training days: {days} (the other days are rest or light movement)\n"
            f"Session length: {minutes} minutes\n"
            f"Equipment: {', '.join(equipment) or 'none'}\n"
            f"Things to avoid: {notes or 'none given'}\n\n"
            "Format: a markdown table with columns Day, Session, Minutes covering all "
            "7 days, one line per cell and no HTML tags. Then short sections: Warm-up (5 min), Cool-down (5 min), "
            "How to progress next week, and Safety. Keep the language simple."
        )
        try:
            with st.spinner("Building your plan..."):
                response = groq_client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "You are NutriFit AI Fitness Planner. Create safe, realistic, "
                            "age-appropriate activity plans for general fitness. For "
                            "under-13s use playful, low-intensity activities only. Always "
                            "include rest days, a warm-up and a cool-down. Never diagnose "
                            "or treat injuries or medical conditions, and never promise "
                            "weight loss. If the user mentions pain or a medical concern, "
                            "advise seeing a doctor or physiotherapist before training.")},
                        {"role": "user", "content": request},
                    ],
                    temperature=0.4,
                    max_tokens=900,
                )
            st.session_state.fitness_plan = clean(response.choices[0].message.content)
            st.session_state.fitness_profile = {
                "Goal": goal, "Level": level, "Age group": age_group,
                "Training days": f"{days} per week", "Session length": f"{minutes} minutes",
                "Equipment": ", ".join(equipment) or "None",
                "Avoid or work around": notes or "None given",
            }
            if save("plan", f"{goal}, {days} days a week", st.session_state.fitness_plan,
                    st.session_state.fitness_profile):
                st.toast("Plan saved to your history")
            else:
                hint_signin()
        except Exception as e:
            st.error(f"Couldn't build your plan: {friendly(e)}.")

    plan = st.session_state.get("fitness_plan")
    if plan:
        section("Your weekly plan")
        st.markdown(plan)
        try:
            pdf = plan_to_pdf(plan, st.session_state.get("fitness_profile", {}))
            st.download_button("Download plan (PDF)", pdf, file_name="nutrifit-weekly-plan.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF export isn't available ({e}). Add reportlab to requirements.txt.")
            st.download_button("Download plan (text)", plan, file_name="nutrifit-weekly-plan.md")
        st.caption("General guidance only. Stop if something hurts, and talk to a "
                   "health professional before starting if you have a medical condition.")

# ---------- Saved Information ----------
elif page == "Saved Information":
    header("Saved Information", "Your chats, food scans, fitness plans and dish checks.")
    user = st.session_state.get("user")

    if not user:
        st.info("Sign in to keep your chats, scans and plans between visits.")
        if st.button("Sign in or create an account"):
            account_dialog()
        if st.session_state.messages or st.session_state.get("fitness_plan"):
            st.markdown("### From this visit (not saved)")
            if st.session_state.messages:
                with st.expander(f"Current conversation ({len(st.session_state.messages)} messages)"):
                    for m in st.session_state.messages:
                        st.markdown(f"**{m['role'].title()}:** {m['content']}")
            if st.session_state.get("fitness_plan"):
                with st.expander("Latest fitness plan"):
                    st.markdown(st.session_state.fitness_plan)
    else:
        kinds = [("chat", "Chats", "No saved chats yet. Chat with the AI Assistant while signed in."),
                 ("scan", "Food scans", "No food scans yet. Analyse a photo in the Food Scanner."),
                 ("product", "Products", "No saved products yet. Scan a barcode and save it."),
                 ("plan", "Fitness plans", "No plans yet. Build one in the Fitness Planner."),
                 ("allergy", "Dish checks", "No dish checks yet. Try Check a dish in the Allergy Centre.")]
        totals = db.counts(user["id"])
        tabs = st.tabs([f"{label} ({totals.get(kind, 0)})" for kind, label, _ in kinds])

        for tab, (kind, label, empty) in zip(tabs, kinds):
            with tab:
                items = db.list_items(user["id"], kind)
                if not items:
                    st.info(empty)
                for it in items:
                    with st.expander(f"{it['title']}  ({it['updated']})"):
                        if kind == "chat":
                            msgs = json.loads(it["body"])
                            for m in msgs:
                                who = "You" if m["role"] == "user" else "NutriFit AI"
                                st.markdown(f"**{who}:** {m['content']}")
                        elif kind == "scan" and it["meta"] and json.loads(it["meta"]).get("is_food"):
                            saved_scan = json.loads(it["meta"])
                            st.markdown(scan_summary_html(saved_scan) + scan_detail_html(saved_scan),
                                        unsafe_allow_html=True)
                        elif kind == "product" and it["meta"]:
                            saved_product = json.loads(it["meta"])
                            st.markdown(product_summary_html(saved_product) + product_detail_html(saved_product),
                                        unsafe_allow_html=True)
                            saved_dates = saved_product.get("dates") or {}
                            if saved_dates.get("expiry"):
                                st.caption(f"{saved_dates.get('kind', 'Expiry')}: {saved_dates['expiry']}")
                            if saved_dates.get("made"):
                                st.caption(f"Made on: {saved_dates['made']}")
                        else:
                            st.markdown(it["body"])
                        c1, c2 = st.columns(2)
                        with c1:
                            if kind == "chat":
                                st.button("Continue this chat", key=f"go_{it['id']}",
                                          on_click=resume_chat, args=(it["id"], msgs))
                            elif kind == "plan":
                                try:
                                    pdf = plan_to_pdf(it["body"], json.loads(it["meta"] or "{}"))
                                    st.download_button("Download PDF", pdf, key=f"pdf_{it['id']}",
                                                       file_name="nutrifit-weekly-plan.pdf",
                                                       mime="application/pdf")
                                except Exception:
                                    pass
                        with c2:
                            st.button("Delete", key=f"del_{it['id']}",
                                      on_click=db.delete_item, args=(user["id"], it["id"]))

        st.write("")
        with st.expander("Manage my data"):
            st.caption("Deleting is permanent and can't be undone.")
            sure = st.checkbox("I understand")
            c1, c2 = st.columns(2)
            if c1.button("Delete all saved items", disabled=not sure):
                db.delete_all(user["id"])
                st.session_state.chat_id = None
                st.rerun()
            if c2.button("Delete my account", disabled=not sure):
                db.delete_account(user["id"])
                st.session_state.user = None
                st.session_state.profile = {}
                st.session_state.chat_id = None
                st.rerun()


# ---------- About and Privacy ----------
elif page == "About and Privacy":
    header("About and Privacy", "What NutriFit AI is, what it stores and how to remove your data.")

    def about_card(color, icon, title, body):
        return f'<div style="--c:{color}"><div class="icon">{ICONS[icon]}</div><h4>{title}</h4>{body}</div>'

    st.markdown('<div class="about">'
        + about_card("#2E8B57", "nutri", "What NutriFit AI is",
            "<p>A learning tool about food, nutrition and physical activity. It uses AI to answer questions, "
            "estimate what is in a meal photo and suggest activity plans.</p>"
            "<p><b>It is not medical advice</b> and cannot diagnose allergies or health conditions. Speak to a "
            "doctor, dietitian or other qualified professional about personal health concerns.</p>")
        + about_card("#4A7FB5", "saved", "What is stored",
            "<ul><li>Only if you create an account: your nickname, a scrambled (hashed) copy of your PIN, "
            "your profile, and the chats, scans, plans and dish checks you save.</li>"
            "<li>Photos are not stored. Only the written results of a scan are kept.</li>"
            "<li>Without an account, nothing is kept after you close the page.</li></ul>")
        + about_card("#F4A93B", "scan", "Who sees what you send",
            "<p>To answer you, your messages, photos and profile details (such as allergies) are sent to AI "
            "services (Groq, and Google Gemini for photos when it is set up), which write the reply.</p>"
            "<p>Barcode numbers you look up are sent to Open Food Facts, a free community database of food products.</p>"
            "<p>Please don't enter your full name, address or other personal details in chats.</p>")
        + about_card("#D9534F", "allergy", "Your choices",
            "<p>Open Saved Information, then Manage my data, to delete your saved items or your whole "
            "account at any time. Deleting is permanent.</p><p>PINs can't be reset, so keep yours safe.</p>")
        + "</div>", unsafe_allow_html=True)
    st.caption("If you are under 18, please ask a parent, guardian or teacher before creating an account.")

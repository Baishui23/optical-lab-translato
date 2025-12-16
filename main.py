import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image
import io
import re
import base64
import os
import subprocess
import tempfile
import shutil
import platform
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
import time
import random
import hashlib
import threading
from collections import OrderedDict

# ============================================================
# 0) 配置部分
# ============================================================
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    API_KEY = "sk-xxxxxxxx"

BASE_URL = "https://api.deepseek.com"

# 线程本地 client：避免多线程共享同一 client 引发偶发连接问题
_thread_local = threading.local()

def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _thread_local.client

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# ============================================================
# 1) CSS 生成器
# ============================================================
def get_css(font_size, line_height, img_width_pct):
    text_width_pct = 100 - img_width_pct - 2

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

        @page {{
            size: A4 landscape;
            margin: 15mm;
        }}

        body {{
            font-family: "Noto Serif SC", "SimSun", serif;
            font-size: {font_size}px;
            line-height: {line_height};
            color: #111;
            margin: 0;
            padding: 0;
            background-color: white;
        }}

        .page-container {{
            width: 100%;
            margin: 0 auto;
        }}

        /* 左右对照布局 */
        .split-layout {{
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 30px;
            align-items: flex-start;
            border-bottom: 1px dashed #ccc;
            padding-bottom: 30px;
            page-break-inside: avoid;
        }}

        .left-col-image {{
            width: {img_width_pct}%;
            flex-shrink: 0;
            border: 1px solid #ddd;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            border-radius: 4px;
            overflow: hidden;
        }}

        .left-col-image img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .right-col-text {{
            width: {text_width_pct}%;
            padding-left: 5px;
            text-align: justify;
            overflow-wrap: break-word;
        }}

        .MathJax {{ font-size: 100% !important; }}

        /* 纯净模式 */
        .pure-mode-container {{ max-width: 900px; margin: 0 auto; }}
        .pure-mode-container p {{ margin-bottom: 1em; text-indent: 2em; }}
        .pure-mode-container img {{ max-width: 80%; display: block; margin: 20px auto; }}

        .caption {{
            font-size: {font_size - 2}px;
            color: #555;
            text-align: center;
            font-weight: bold;
            margin-bottom: 15px;
            font-family: sans-serif;
        }}

        .page-marker {{
            text-align: center; font-size: 12px; color: #aaa;
            margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;
        }}
        .page-break {{ page-break-before: always; }}
        .page-break.first-page {{ page-break-before: avoid; }}
        @media print {{ .page-break {{ height: 0; margin: 0; }} }}
    </style>
    """

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# ============================================================
# 2) 工具函数：图片/页眉页脚/图注识别/latex清洗
# ============================================================

def image_to_base64(pil_image, fmt="PNG", jpeg_quality=85):
    buff = io.BytesIO()
    if fmt.upper() == "JPEG":
        pil_image = pil_image.convert("RGB")
        pil_image.save(buff, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        pil_image.save(buff, format="PNG")
        mime = "image/png"
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{img_str}"

def is_header_or_footer(rect, page_height):
    # 用比例比固定像素更稳
    top_cut = page_height * 0.06
    bottom_cut = page_height * 0.94
    return rect.y1 < top_cut or rect.y0 > bottom_cut

_CAPTION_RE = re.compile(
    r"""^\s*(
        (Fig\.|FIG\.|Figure|FIGURE)\s*\d+(\s*[:.])? |
        (Tab\.|TAB\.|Table|TABLE)\s*\d+(\s*[:.])? |
        (图|表)\s*\d+(\s*[:：.])?
    )""",
    re.VERBOSE
)

def is_caption_node(text):
    t = (text or "").strip()
    if not t:
        return False
    return bool(_CAPTION_RE.match(t))

def clean_latex(text):
    return (text or "").replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# ============================================================
# 3) 翻译增强：缓存 + 重试 + 分段
# ============================================================

class LRUCache:
    def __init__(self, max_size=2500):
        self.max_size = max_size
        self._data = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

_TRANSLATION_CACHE = LRUCache(max_size=2500)

def _cache_key(text, is_caption):
    h = hashlib.sha256((text + ("|cap" if is_caption else "|txt")).encode("utf-8")).hexdigest()
    return h

def _split_text_for_translation(text, max_chars=2500):
    t = (text or "").strip()
    if len(t) <= max_chars:
        return [t]

    parts = []
    buf = ""
    for para in t.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= max_chars:
            buf = (buf + "\n\n" + para).strip()
        else:
            if buf:
                parts.append(buf)
            if len(para) > max_chars:
                sent_buf = ""
                for seg in re.split(r"(?<=[。！？.!?;；])\s+", para):
                    seg = seg.strip()
                    if not seg:
                        continue
                    if len(sent_buf) + len(seg) + 1 <= max_chars:
                        sent_buf = (sent_buf + " " + seg).strip()
                    else:
                        if sent_buf:
                            parts.append(sent_buf)
                        sent_buf = seg
                if sent_buf:
                    parts.append(sent_buf)
                buf = ""
            else:
                buf = para
    if buf:
        parts.append(buf)
    return parts

def translate_text(text, is_caption=False, retries=3):
    raw = (text or "")
    if len(raw.strip()) < 2:
        return raw

    ck = _cache_key(raw, is_caption)
    cached = _TRANSLATION_CACHE.get(ck)
    if cached is not None:
        return cached

    sys_prompt = """你是一个专业的物理学术翻译。
【规则】
1. 保持学术严谨性，尽量用规范学术中文。
2. 不要改变量名、单位、符号（例如 nm, μm, dB, SNR, GHz, μJ 等）。
3. 数学表达式/希腊字母/上下标尽量原样保留；如必须用 LaTeX，请用 $...$ 或 $$...$$。
4. 直接输出译文，不要加任何前缀/解释。
"""
    if is_caption:
        sys_prompt += "5. 这是图注/表注：必须保留原编号格式（如 Fig. 1 / Figure 1 / Table 1 / 图1 / 表1）。\n"

    parts = _split_text_for_translation(raw, max_chars=2500)
    out_parts = []

    for part in parts:
        last_err = None
        for attempt in range(retries):
            try:
                client = get_client()
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": part},
                    ],
                    stream=False
                )
                out_parts.append(resp.choices[0].message.content)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep((2 ** attempt) + random.random() * 0.5)

        if last_err is not None:
            if "translate_errors" not in st.session_state:
                st.session_state["translate_errors"] = []
            st.session_state["translate_errors"].append(str(last_err))
            out_parts.append(part)  # 回退原文

    final = "\n\n".join(out_parts).strip()
    _TRANSLATION_CACHE.set(ck, final)
    return final

def batch_translate_elements(elements, max_workers=4):
    tasks = []
    indices = []

    for i, el in enumerate(elements):
        if el.get('type') in ['text', 'caption']:
            txt = el.get('content', '')
            if len(str(txt).strip()) < 2:
                continue
            tasks.append((txt, el['type'] == 'caption'))
            indices.append(i)

    if not tasks:
        return elements

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))

    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]

    return elements

# ============================================================
# 4) 关键：真正的“图识别”（从 PDF image blocks 提取）
# ============================================================

def extract_image_blocks(page):
    """
    提取 page.get_text("dict") 里的图片块（type==1），避免用空白区域截屏猜图。
    """
    d = page.get_text("dict")
    img_blocks = []
    for b in d.get("blocks", []):
        if b.get("type") == 1 and "bbox" in b:
            x0, y0, x1, y1 = b["bbox"]
            # 过滤小图标/噪点
            if (x1 - x0) >= 80 and (y1 - y0) >= 60:
                img_blocks.append({
                    "bbox": fitz.Rect(x0, y0, x1, y1),
                    "used": False
                })
    img_blocks.sort(key=lambda it: it["bbox"].y0)_

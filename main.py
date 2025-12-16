import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image, ImageStat, ImageFilter
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
# 0) 配置
# ============================================================
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    API_KEY = "sk-xxxxxxxx"

BASE_URL = "https://api.deepseek.com"

_thread_local = threading.local()

def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _thread_local.client

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# ============================================================
# 1) CSS / MathJax
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
        .page-container {{ width: 100%; margin: 0 auto; }}

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
        .left-col-image img {{ width: 100%; height: auto; display: block; }}

        .right-col-text {{
            width: {text_width_pct}%;
            padding-left: 5px;
            text-align: justify;
            overflow-wrap: break-word;
        }}

        .MathJax {{ font-size: 100% !important; }}

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
# 2) 基础工具
# ============================================================
def image_to_base64(pil_image, fmt="JPEG", jpeg_quality=85):
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

def clean_latex(text):
    return (text or "").replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

def is_header_or_footer(rect, page_height):
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
    return hashlib.sha256((text + ("|cap" if is_caption else "|txt")).encode("utf-8")).hexdigest()

def _split_text_for_translation(text, max_chars=2500):
    t = (text or "").strip()
    if len(t) <= max_chars:
        return [t]

    parts, buf = [], ""
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
            st.session_state.setdefault("translate_errors", []).append(str(last_err))
            out_parts.append(part)  # 回退原文

    final = "\n\n".join(out_parts).strip()
    _TRANSLATION_CACHE.set(ck, final)
    return final

def batch_translate_elements(elements, max_workers=4):
    tasks, indices = [], []
    for i, el in enumerate(elements):
        if el.get("type") in ("text", "caption"):
            txt = el.get("content", "")
            if len(str(txt).strip()) < 2:
                continue
            tasks.append((txt, el["type"] == "caption"))
            indices.append(i)

    if not tasks:
        return elements

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(lambda p: translate_text(p[0], p[1]), tasks))

    for k, idx in enumerate(indices):
        elements[idx]["content"] = results[k]
    return elements

# ============================================================
# 4) 空白区裁图 + 公式/图混合智能判断
# ============================================================

# 你可以在侧边栏调的默认值（代码里先给个“工程默认”）
DEFAULT_MIN_GAP_HEIGHT = 120
DEFAULT_TOP_MARGIN = 60
DEFAULT_BOTTOM_MARGIN = 60
DEFAULT_SIDE_MARGIN = 40

def clip_rect_to_image(page, rect, zoom=2.2):
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png")))
    except Exception:
        return None

def extract_text_in_rect(page, rect):
    """提取区域内文本（用于判断是不是公式/纯文字）。"""
    try:
        blocks = page.get_text("blocks", clip=rect, sort=True)
        texts = []
        for b in blocks:
            if len(b) > 4 and str(b[4]).strip():
                texts.append(str(b[4]))
        return "\n".join(texts)
    except Exception:
        return ""

def is_formula_like_text(text):
    """偏保守：只有非常像公式才判 True。"""
    if not text:
        return False
    t = text.strip()
    if len(t) < 25:
        return False

    # 数学符号/结构字符
    math_symbols = r"[=+\-*/^_{}[\]<>∑∫√≈≠≤≥±λμσΩπ∞→←×·]"
    symbol_count = len(re.findall(math_symbols, t))

    # 单字母变量
    single_letters = re.findall(r"\b[a-zA-Z]\b", t)

    # 自然语言单词（>=3 字母）
    words = re.findall(r"\b[a-zA-Z]{3,}\b", t)

    symbol_ratio = symbol_count / max(1, len(t))
    single_letter_ratio = len(single_letters) / max(1, (len(words) + len(single_letters)))

    # 经验阈值：更偏向“别误杀”
    return (symbol_ratio > 0.12 and single_letter_ratio > 0.55 and len(words) < 8)

def image_visual_score(pil_img):
    """
    用图像特征判断“像不像图”：
    - 墨迹密度（非白像素占比）
    - 对比度（stddev）
    - 边缘强度（FIND_EDGES 的平均亮度）
    返回一个综合分数，越大越像“图形/曲线/结构”。
    """
    if pil_img is None:
        return 0.0

    # 降采样，速度更快
    img = pil_img.convert("L")
    w, h = img.size
    if w * h > 700_000:
        img = img.resize((max(200, w // 2), max(200, h // 2)))

    stat = ImageStat.Stat(img)
    mean = stat.mean[0]
    std = stat.stddev[0]

    # 非白像素占比（墨迹密度）
    # 阈值 245：接近白色算背景
    hist = img.histogram()
    total = sum(hist)
    whiteish = sum(hist[245:256])
    nonwhite_ratio = 1.0 - (whiteish / max(1, total))

    # 边缘强度
    edges = img.filter(ImageFilter.FIND_EDGES)
    estat = ImageStat.Stat(edges)
    edge_mean = estat.mean[0]

    # 综合评分（不追求绝对科学，追求稳）
    # std / edge_mean / nonwhite_ratio 都大，更像图（曲线/示意/照片）
    score = (std * 0.6) + (edge_mean * 0.8) + (nonwhite_ratio * 120.0) - (abs(mean - 245) * 0.05)
    return float(score)

def should_keep_cropped_region(page, rect):
    """
    最核心：决定“空白区”裁出来的区域要不要当图片插入。
    智能策略：
    1) 如果区域内几乎没有文本 -> 主要看图像特征，像图就保留
    2) 如果区域文本很像公式：
       - 但图像特征很强（混合：图 + 公式标注/坐标） -> 仍保留
       - 图像特征弱 -> 当成纯公式，丢弃
    """
    txt = extract_text_in_rect(page, rect)
    txt_stripped = (txt or "").strip()

    # 先渲染一张低倍率用于判断（速度快）
    img_probe = clip_rect_to_image(page, rect, zoom=1.3)
    vscore = image_visual_score(img_probe)

    # 1) 文本很少：主要看像不像图
    if len(txt_stripped) < 10:
        # vscore 阈值：偏保守，不要漏图
        return vscore >= 35.0

    # 2) 有文本：判断是否公式倾向
    formula_like = is_formula_like_text(txt_stripped)

    if not formula_like:
        # 不是公式：这种情况一般不该被当成“图空白区”，但如果落进来，仍用图像判断
        return vscore >= 40.0

    # 3) 公式倾向：做混合判定
    # 如果是“公式 + 图”混合（比如曲线图的公式标注/坐标），vscore 通常会明显更高
    # 这里阈值更高一些，避免大公式被当图
    return vscore >= 62.0

def build_gap_rects_from_text_blocks(page, min_gap_height, top_margin, bottom_margin, side_margin):
    """
    从文本块的 y 区间找“空白带”，输出候选裁剪 rect 列表。
    """
    page_rect = page.rect
    blocks = page.get_text("blocks", sort=True)

    # 只用“正文”文本块（过滤页眉页脚）
    text_rects = []
    for b in blocks:
        r = fitz.Rect(b[:4])
        if is_header_or_footer(r, page_rect.height):
            continue
        # b[4] 是文本内容，空的不要
        content = b[4] if len(b) > 4 else ""
        if str(content).strip():
            text_rects.append(r)

    # 没有文字：整页可能是图片/扫描页（可选：整页当图）
    if not text_rects:
        return []

    # 合并 y 区间（粗略合并：按 y 排序，重叠就合并）
    text_rects.sort(key=lambda r: r.y0)
    merged = []
    cur = fitz.Rect(text_rects[0])
    for r in text_rects[1:]:
        if r.y0 <= cur.y1 + 2:
            cur.y1 = max(cur.y1, r.y1)
            cur.x0 = min(cur.x0, r.x0)
            cur.x1 = max(cur.x1, r.x1)
        else:
            merged.append(cur)
            cur = fitz.Rect(r)
    merged.append(cur)

    # 找空白区：在 merged 区间之间的缝
    rects = []
    safe_top = max(top_margin, int(page_rect.height * 0.06))
    safe_bottom = min(page_rect.height - bottom_margin, int(page_rect.height * 0.94))
    x0 = side_margin
    x1 = page_rect.width - side_margin

    last_y = safe_top
    for r in merged:
        gap_top = last_y
        gap_bottom = min(r.y0, safe_bottom)
        if gap_bottom - gap_top >= min_gap_height:
            rects.append(fitz.Rect(x0, gap_top, x1, gap_bottom))
        last_y = max(last_y, r.y1)

    # 最后一段空白
    if safe_bottom - last_y >= min_gap_height:
        rects.append(fitz.Rect(x0, last_y, x1, safe_bottom))

    return rects

# ============================================================
# 5) parse_page：文本 + 空白裁图 + 图注（可选）
# ============================================================

def parse_page(page, min_gap_height, top_margin, bottom_margin, side_margin):
    """
    输出 elements：text / image / caption
    逻辑：
    - 先拿文本 blocks，按顺序拼段落
    - 同时找空白区，裁图并按 y 位置插入到元素流里
    - 图注单独识别（并翻译），但不强绑某张图（论文排版太复杂，强绑容易错）
    """
    page_rect = page.rect
    blocks = page.get_text("blocks", sort=True)

    # 1) 先收集“正文文本块”和“图注块”
    text_items = []
    caption_items = []
    for b in blocks:
        r = fitz.Rect(b[:4])
        if is_header_or_footer(r, page_rect.height):
            continue
        content = b[4] if len(b) > 4 else ""
        content = str(content)
        if not content.strip():
            continue
        if is_caption_node(content):
            caption_items.append((r, content))
        else:
            text_items.append((r, content))

    # 2) 找空白区候选裁图 rect（基于文本块）
    gap_rects = build_gap_rects_from_text_blocks(
        page,
        min_gap_height=min_gap_height,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        side_margin=side_margin,
    )

    # 3) 对每个 gap rect 做智能判断：保留图/过滤纯公式
    cropped_images = []
    for gr in gap_rects:
        # 先快速过滤太小区域
        if gr.height < min_gap_height:
            continue
        if should_keep_cropped_region(page, gr):
            img = clip_rect_to_image(page, gr, zoom=2.0)
            if img is not None and img.size[0] >= 120 and img.size[1] >= 80:
                cropped_images.append((gr, img))

    # 4) 构建元素流：按 y 排序插入
    # 策略：把 text 按 y 排序拼 buffer；遇到“在当前位置之前的图片”就先插图
    cropped_images.sort(key=lambda it: it[0].y0)
    caption_items.sort(key=lambda it: it[0].y0)
    text_items.sort(key=lambda it: it[0].y0)

    elements = []
    img_ptr = 0
    cap_ptr = 0

    buffer = ""
    current_y = 0.0

    def flush_buffer():
        nonlocal buffer
        if buffer.strip():
            elements.append({"type": "text", "content": buffer})
            buffer = ""

    # 把“文字与图注”当成一个按 y 的事件流
    events = []
    for r, t in text_items:
        events.append(("text", r, t))
    for r, t in caption_items:
        events.append(("cap", r, t))
    events.sort(key=lambda e: e[1].y0)

    for kind, r, t in events:
        # 先插入在当前事件之前的图片
        while img_ptr < len(cropped_images) and cropped_images[img_ptr][0].y0 <= r.y0 + 2:
            flush_buffer()
            elements.append({"type": "image", "content": cropped_images[img_ptr][1]})
            img_ptr += 1

        if kind == "cap":
            flush_buffer()
            elements.append({"type": "caption", "content": t})
        else:
            buffer += t + "\n\n"

        current_y = r.y1

    # 末尾收尾：剩余图片
    flush_buffer()
    while img_ptr < len(cropped_images):
        elements.append({"type": "image", "content": cropped_images[img_ptr][1]})
        img_ptr += 1

    # 5) 翻译文本/图注（图不翻）
    return batch_translate_elements(elements, max_workers=4)

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ============================================================
# 6) HTML 构建器
# ============================================================
def generate_html(doc, start, end, mode="pure", font_size=14, line_height=1.6, img_width=48,
                  min_gap_height=DEFAULT_MIN_GAP_HEIGHT, top_margin=DEFAULT_TOP_MARGIN,
                  bottom_margin=DEFAULT_BOTTOM_MARGIN, side_margin=DEFAULT_SIDE_MARGIN,
                  show_images_in_compare_right=False):
    css = get_css(font_size, line_height, img_width)
    html_body = '<div class="page-container">'

    for page_num in range(start, end + 1):
        page = doc[page_num - 1]
        marker_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{marker_class}"><div class="page-marker">- 第 {page_num} 页 -</div></div>'

        els = parse_page(
            page,
            min_gap_height=min_gap_height,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            side_margin=side_margin
        )

        if mode == "screenshot":
            left_b64 = image_to_base64(get_page_image(page), fmt="JPEG", jpeg_quality=85)
            html_body += f"""
            <div class="split-layout">
              <div class="left-col-image"><img src="{left_b64}" /></div>
              <div class="right-col-text">
            """
            for el in els:
                if el["type"] == "text":
                    paras = clean_latex(el["content"]).split("\n\n")
                    for p in paras:
                        p = (p or "").strip()
                        if p:
                            html_body += f"<p>{p.replace('**', '')}</p>"
                elif el["type"] == "caption":
                    html_body += f'<div class="caption">图注: {el["content"]}</div>'
                elif el["type"] == "image" and show_images_in_compare_right:
                    html_body += f'<img src="{image_to_base64(el["content"], fmt="JPEG", jpeg_quality=85)}" />'
            html_body += "</div></div>"

        else:
            html_body += '<div class="pure-mode-container">'
            for el in els:
                if el["type"] == "text":
                    paras = clean_latex(el["content"]).split("\n\n")
                    for p in paras:
                        p = (p or "").strip()
                        if p:
                            html_body += f"<p>{p.replace('**', '')}</p>"
                elif el["type"] == "image":
                    html_body += f'<img src="{image_to_base64(el["content"], fmt="JPEG", jpeg_quality=85)}" />'
                elif el["type"] == "caption":
                    html_body += f'<div class="caption">{el["content"]}</div>'
            html_body += "</div>"

    html_body += "</div>"
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

# ============================================================
# 7) PDF 导出
# ============================================================
def get_chrome_path():
    if shutil.which("chromium"):
        return shutil.which("chromium")
    if shutil.which("chromium-browser"):
        return shutil.which("chromium-browser")

    mac_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for p in mac_paths:
        if os.path.exists(p):
            return p

    win_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p
    return None

def html_to_pdf_with_chrome(html_content, output_pdf_path):
    chrome_bin = get_chrome_path()
    if not chrome_bin:
        return False, "❌ 未找到 Chrome/Chromium（导出 PDF 需要浏览器核心）"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    cmd = [
        chrome_bin, "--headless", "--disable-gpu",
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux":
        cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")
        return False, f"Chrome 导出失败：{err[:800]}"
    except Exception as e:
        return False, str(e)

# ============================================================
# 8) UI
# ============================================================
st.title("🔬 光学室学术论文翻译专用版 (V42 空白裁图 + 混合区智能判断)")

with st.sidebar:
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #dcdcdc;">
        <h4 style="margin:0; color:#333;">👤 专属定制</h4>
        <p style="margin:5px 0 0 0; font-size:14px; color:#555;">
        <strong>制作人：</strong> 白水<br>
        <strong>微信：</strong> <code style="background:white;">guo21615</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("上传 PDF", type="pdf")

    st.markdown("---")
    with st.expander("🎨 排版设置", expanded=True):
        ui_font_size = st.slider("字体大小 (px)", 10, 18, 14)
        ui_line_height = st.slider("行间距", 1.2, 2.0, 1.6, 0.1)
        ui_img_width = st.slider("左图占比 (%)", 30, 70, 48)

    st.markdown("---")
    with st.expander("🖼️ 识图参数（空白裁图法）", expanded=False):
        min_gap_height = st.slider("最小空白高度（越大越不容易误裁公式）", 60, 300, DEFAULT_MIN_GAP_HEIGHT, 10)
        side_margin = st.slider("左右留白（px）", 0, 120, DEFAULT_SIDE_MARGIN, 5)
        top_margin = st.slider("页眉避开（px）", 0, 160, DEFAULT_TOP_MARGIN, 5)
        bottom_margin = st.slider("页脚避开（px）", 0, 160, DEFAULT_BOTTOM_MARGIN, 5)

    st.markdown("---")
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    export_style = "中英对照版 (左图右文)"
    if app_mode == "🖨️ 导出 PDF":
        export_style = st.radio("导出风格：", ["纯净译文版", "中英对照版 (左图右文)"], index=1)

    st.markdown("---")
    show_images_in_compare_right = st.checkbox("对照版：右侧也插入裁剪图（可选）", value=False)

    if st.session_state.get("translate_errors"):
        with st.expander(f"⚠️ 翻译错误日志（{len(st.session_state['translate_errors'])}）", expanded=False):
            for i, e in enumerate(st.session_state["translate_errors"][-30:], 1):
                st.write(f"{i}. {e}")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if app_mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state["run_preview"] = True

        if st.session_state.get("run_preview"):
            with st.spinner("🚀 渲染预览中..."):
                preview_html = generate_html(
                    doc, page_num, page_num,
                    mode="screenshot",
                    font_size=ui_font_size,
                    line_height=ui_line_height,
                    img_width=ui_img_width,
                    min_gap_height=min_gap_height,
                    top_margin=top_margin,
                    bottom_margin=bottom_margin,
                    side_margin=side_margin,
                    show_images_in_compare_right=show_images_in_compare_right
                )
                components.html(preview_html, height=860, scrolling=True)
        else:
            st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1:
            start = st.number_input("起始页", 1, len(doc), 1)
        with c2:
            end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))

        style_code = "screenshot" if "对照" in export_style else "pure"

        if st.button("🚀 生成 PDF", type="primary"):
            st.session_state["translate_errors"] = []
            bar = st.progress(0)
            status = st.empty()

            status.text("正在翻译 + 构建 HTML（空白裁图 + 混合区判断）...")
            full_html = generate_html(
                doc, start, end,
                mode=style_code,
                font_size=ui_font_size,
                line_height=ui_line_height,
                img_width=ui_img_width,
                min_gap_height=min_gap_height,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                side_margin=side_margin,
                show_images_in_compare_right=show_images_in_compare_right
            )
            bar.progress(70)

            status.text("正在生成 PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    if st.session_state.get("translate_errors"):
                        status.warning(f"✅ 完成，但有 {len(st.session_state['translate_errors'])} 条翻译错误（已回退原文）。")
                    else:
                        status.success("✅ 完成！")

                    fname = "Translation_V42.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载文件", f, fname)
                else:
                    st.error(f"失败: {msg}")

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
    img_blocks.sort(key=lambda it: it["bbox"].y0)
    return img_blocks

def clip_rect_to_image(page, rect, zoom=2.5):
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    except Exception:
        return None

def find_best_image_for_caption(img_blocks, cap_rect, page_rect):
    """
    给定图注矩形 cap_rect，找一个最合理的图片块：
    - 优先找 cap 上方、距离最近、横向重叠较多的图
    - 上方找不到再找下方
    """
    def x_overlap(a, b):
        inter = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
        return inter / max(1, min(a.width, b.width))

    best = None
    best_score = -1

    # 上方优先
    for it in img_blocks:
        if it["used"]:
            continue
        r = it["bbox"]
        if r.y1 <= cap_rect.y0 + 5:
            dist = cap_rect.y0 - r.y1
            if dist > page_rect.height * 0.35:
                continue
            overlap = x_overlap(r, cap_rect)
            score = overlap * 2.0 - dist / 800.0
            if score > best_score:
                best_score, best = score, it

    # 下方兜底
    if best is None:
        for it in img_blocks:
            if it["used"]:
                continue
            r = it["bbox"]
            if r.y0 >= cap_rect.y1 - 5:
                dist = r.y0 - cap_rect.y1
                if dist > page_rect.height * 0.35:
                    continue
                overlap = x_overlap(r, cap_rect)
                score = overlap * 2.0 - dist / 800.0
                if score > best_score:
                    best_score, best = score, it

    return best

# ============================================================
# 5) parse_page：抽图 + 图注配对 + 文本翻译
# ============================================================

def parse_page(page):
    raw_elements = []
    blocks = page.get_text("blocks", sort=True)
    page_h = page.rect.height

    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page_h)]

    # 先提取本页真正的图片块
    img_blocks = extract_image_blocks(page)

    text_buffer = ""

    for b in valid_blocks:
        b_rect = fitz.Rect(b[:4])
        content = b[4] if len(b) > 4 else ""

        if is_caption_node(content):
            if text_buffer.strip():
                raw_elements.append({'type': 'text', 'content': text_buffer})
                text_buffer = ""

            best = find_best_image_for_caption(img_blocks, b_rect, page.rect)
            if best is not None:
                best["used"] = True
                img = clip_rect_to_image(page, best["bbox"], zoom=2.5)
                if img:
                    raw_elements.append({'type': 'image', 'content': img})

            raw_elements.append({'type': 'caption', 'content': content})
        else:
            text_buffer += str(content) + "\n\n"

    if text_buffer.strip():
        raw_elements.append({'type': 'text', 'content': text_buffer})

    return batch_translate_elements(raw_elements, max_workers=4)

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

# ============================================================
# 6) HTML 构建器
# ============================================================

def generate_html(doc, start, end, mode="pure", filename="Document", font_size=14, line_height=1.6, img_width=50):
    dynamic_css = get_css(font_size, line_height, img_width)
    html_body = f'<div class="page-container">'

    for page_num in range(start, end + 1):
        page = doc[page_num - 1]
        marker_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{marker_class}"><div class="page-marker">- 第 {page_num} 页 -</div></div>'

        page_els = parse_page(page)

        if mode == "screenshot":
            # 对照版：左边整页截图最稳，右边展示译文/图注
            img_b64 = image_to_base64(get_page_image(page), fmt="JPEG", jpeg_quality=85)

            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image"><img src="{img_b64}" /></div>
                <div class="right-col-text">
            """
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        p = (p or "").strip()
                        if p:
                            html_body += f"<p>{p.replace('**', '')}</p>"
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">图注: {el["content"]}</div>'
            html_body += "</div></div>"

        else:
            # 纯净译文版：会把抽出来的图插入正文（按图注配对）
            html_body += '<div class="pure-mode-container">'
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        p = (p or "").strip()
                        if p:
                            html_body += f"<p>{p.replace('**', '')}</p>"
                elif el['type'] == 'image':
                    html_body += f'<img src="{image_to_base64(el["content"], fmt="JPEG", jpeg_quality=85)}" />'
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">{el["content"]}</div>'
            html_body += '</div>'

    html_body += "</div>"
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{dynamic_css}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

# ============================================================
# 7) PDF 导出引擎
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
        "--virtual-time-budget=7000",
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
# 8) 界面逻辑
# ============================================================

st.title("🔬 光学室学术论文翻译专用版 (V40 图识别增强版)")

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
    with st.expander("🎨 排版设置 (防溢出)", expanded=True):
        ui_font_size = st.slider("字体大小 (px)", 10, 18, 14)
        ui_line_height = st.slider("行间距", 1.2, 2.0, 1.6, 0.1)
        ui_img_width = st.slider("左图占比 (%)", 30, 70, 48)

    st.markdown("---")
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    if app_mode == "🖨️ 导出 PDF":
        export_style = st.radio("导出风格：", ["纯净译文版", "中英对照版 (左图右文)"], index=1)

    st.markdown("---")
    if st.session_state.get("translate_errors"):
        with st.expander(f"⚠️ 翻译错误日志（{len(st.session_state['translate_errors'])}）", expanded=False):
            for i, e in enumerate(st.session_state["translate_errors"][-20:], 1):
                st.write(f"{i}. {e}")

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if app_mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True

        if st.session_state.get('run_preview'):
            with st.spinner("🚀 渲染预览中..."):
                preview_html = generate_html(
                    doc, page_num, page_num, mode="screenshot",
                    font_size=ui_font_size,
                    line_height=ui_line_height,
                    img_width=ui_img_width
                )
                components.html(preview_html, height=850, scrolling=True)
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

            status.text("正在翻译 + 构建 HTML（带缓存/重试/图识别）...")
            full_html = generate_html(
                doc, start, end, mode=style_code, filename=uploaded_file.name,
                font_size=ui_font_size,
                line_height=ui_line_height,
                img_width=ui_img_width
            )

            bar.progress(70)

            status.text("正在生成 PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    if st.session_state.get("translate_errors"):
                        status.warning(f"✅ 完成，但有 {len(st.session_state['translate_errors'])} 条翻译错误（已尽量用原文回退）。")
                    else:
                        status.success("✅ 完成！")

                    fname = "Translation_V40.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载文件", f, fname)
                else:
                    st.error(f"失败: {msg}")

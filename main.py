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

# ==========================================
# 0. 配置部分
# ==========================================
st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

with st.sidebar:
    st.markdown("### 🔑 API 设置")
    user_api_input = st.text_input("DeepSeek API Key (可选)", type="password")

API_KEY = None
if user_api_input:
    API_KEY = user_api_input
else:
    try:
        API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    except:
        API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 

BASE_URL = "https://api.deepseek.com"
if API_KEY and API_KEY.startswith("sk-"):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
else:
    client = None

# ==========================================
# 1. CSS & Layout
# ==========================================
def get_css(font_size, line_height, img_width_pct):
    text_width_pct = 100 - img_width_pct - 2
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');
        @page {{ size: A4 landscape; margin: 10mm; }}
        body {{ font-family: "Noto Serif SC", serif; font-size: {font_size}px; line-height: {line_height}; margin: 0; padding: 0; }}
        .page-container {{ width: 100%; margin: 0 auto; }}
        .split-layout {{ display: flex; flex-direction: row; gap: 20px; margin-bottom: 30px; align-items: flex-start; border-bottom: 1px dashed #ccc; padding-bottom: 30px; page-break-inside: avoid; }}
        .left-col-image {{ width: {img_width_pct}%; flex-shrink: 0; border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden; }}
        .left-col-image img {{ width: 100%; height: auto; display: block; }}
        .right-col-text {{ width: {text_width_pct}%; padding-left: 5px; text-align: justify; overflow-wrap: break-word; }}
        .caption {{ font-size: {font_size - 2}px; color: #555; text-align: center; font-weight: bold; margin-top: 5px; font-family: sans-serif; }}
        .page-marker {{ text-align: center; font-size: 12px; color: #aaa; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        .page-break {{ page-break-before: always; }}
    </style>
    """

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# ==========================================
# 2. 核心逻辑 (智能识图 V5.0)
# ==========================================

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def is_header_or_footer(rect, page_height):
    return rect.y1 < 40 or rect.y0 > page_height - 40

def is_caption_node(text):
    text = text.strip()
    # 匹配 Fig 1, Figure. 1, Fig.1, Figure 1-2 等
    pattern = r'^(Fig|Figure)(\.|,|\s).{0,5}\d+'
    return re.match(pattern, text, re.IGNORECASE) is not None

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

def translate_text(text, is_caption=False):
    if not client: return text 
    if len(text.strip()) < 2: return text
    sys_prompt = "你是一个物理学术翻译。公式用$$包裹。直接输出译文。"
    if is_caption: sys_prompt += " (这是图注，保留编号)"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except:
        return text

def batch_translate_elements(elements):
    tasks = []
    indices = []
    for i, el in enumerate(elements):
        if el['type'] in ['text', 'caption']:
            tasks.append((el['content'], el['type'] == 'caption'))
            indices.append(i)
    if not tasks: return elements
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))
    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]
    return elements

def capture_image_area(page, top, bottom):
    # 如果计算出的区域小于 10px，说明有问题，可能需要兜底逻辑
    height = bottom - top
    
    # 兜底：如果高度太小（比如重叠了），强制向上抓取 250px (经验值)
    # 这对双栏排版中紧贴顶部的图片非常有效
    if height < 20: 
        top = max(50, bottom - 300) 
    
    rect = fitz.Rect(40, top, page.rect.width - 40, bottom)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        if pix.height < 10 or pix.width < 10: return None
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    except:
        return None

# 🔥【核心修正函数】寻找当前列的“天花板” 🔥
def find_image_top(caption_rect, all_blocks, page_header_height=60):
    """
    不依赖上一段文字，而是查找：
    在所有block中，位于caption正上方，且在水平方向上有重叠（同列）的最低那个block的底部。
    """
    caption_mid_x = (caption_rect.x0 + caption_rect.x1) / 2
    
    # 默认天花板是页面顶部（避开页眉）
    ceiling = page_header_height 
    
    for b in all_blocks:
        b_rect = fitz.Rect(b[:4])
        
        # 1. 必须在 caption 的上方
        if b_rect.y1 <= caption_rect.y0:
            # 2. 必须在水平方向上有交集（判断是否同列）
            # 简单判断：block的中间点是否在 caption 的左右边界内，或者反过来
            b_mid_x = (b_rect.x0 + b_rect.x1) / 2
            
            # 宽松的同列判定 (只要x轴有重叠)
            is_same_column = not (b_rect.x1 < caption_rect.x0 or b_rect.x0 > caption_rect.x1)
            
            if is_same_column:
                # 3. 如果这个 block 比当前记录的 ceiling 更靠下，它就是新的天花板
                if b_rect.y1 > ceiling:
                    ceiling = b_rect.y1
                    
    return ceiling

def parse_page(page):
    raw_elements = []
    # 获取所有 block，保留给 find_image_top 用
    blocks = page.get_text("blocks", sort=True)
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        text_content = b[4]

        # === 遇到图注 ===
        if is_caption_node(text_content):
            # 🌟 调用智能算法：寻找这幅图的真实顶部
            image_top = find_image_top(b_rect, valid_blocks)
            image_bottom = b_rect.y0 # 图注的顶边就是图的底边
            
            # 截图
            img = capture_image_area(page, image_top, image_bottom)
            
            if img:
                raw_elements.append({'type': 'image', 'content': img})
            
            raw_elements.append({'type': 'caption', 'content': text_content})
            
        else:
            # 普通文本
            raw_elements.append({'type': 'text', 'content': text_content})

    return batch_translate_elements(raw_elements)

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

# ==========================================
# 3. HTML & PDF
# ==========================================
def generate_html(doc, start, end, mode="pure", filename="Document", font_size=14, line_height=1.6, img_width=50):
    dynamic_css = get_css(font_size, line_height, img_width)
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        html_body += f'<div class="page-break"><div class="page-marker">- 第 {page_num} 页 -</div></div>'
        
        if mode == "screenshot":
            page_els = parse_page(page) 
            img_b64 = image_to_base64(get_page_image(page))
            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image"><img src="{img_b64}" /></div>
                <div class="right-col-text">
            """
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">{el["content"]}</div>'
            html_body += "</div></div>"
        else:
            page_els = parse_page(page)
            html_body += '<div class="pure-mode-container">'
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'image':
                    html_body += f'<img src="{image_to_base64(el["content"])}" />'
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">{el["content"]}</div>'
            html_body += '</div>'
    html_body += "</div>"
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{dynamic_css}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

def get_chrome_path():
    if shutil.which("chromium"): return shutil.which("chromium")
    if shutil.which("chromium-browser"): return shutil.which("chromium-browser")
    win_paths = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]
    for p in win_paths: 
        if os.path.exists(p): return p
    return None

def html_to_pdf_with_chrome(html_content, output_pdf_path):
    chrome_bin = get_chrome_path()
    if not chrome_bin: return False, "❌ 未找到浏览器核心"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name
    cmd = [chrome_bin, "--headless", "--disable-gpu", f"--print-to-pdf={output_pdf_path}", "--no-pdf-header-footer", f"file://{tmp_html_path}"]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# ==========================================
# 4. UI 入口
# ==========================================
st.title("🔬 光学室学术论文翻译专用版 (V60 同列回溯修复版)")

uploaded_file = st.sidebar.file_uploader("上传 PDF", type="pdf")
with st.sidebar.expander("🎨 排版设置"):
    ui_font_size = st.slider("字体大小", 10, 18, 14)
    ui_line_height = st.slider("行间距", 1.2, 2.0, 1.6)
    ui_img_width = st.slider("左图占比 (%)", 30, 70, 48)

app_mode = st.sidebar.radio("模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if app_mode == "👁️ 实时预览":
        page_num = st.sidebar.number_input("页码", 1, len(doc), 1)
        if st.sidebar.button("🔄 翻译此页"):
            preview_html = generate_html(doc, page_num, page_num, mode="screenshot", font_size=ui_font_size, line_height=ui_line_height, img_width=ui_img_width)
            components.html(preview_html, height=800, scrolling=True)
    else:
        if st.sidebar.button("🚀 生成 PDF"):
            full_html = generate_html(doc, 1, 3, mode="screenshot" if "对照" in "对照" else "pure", filename="doc", font_size=ui_font_size, line_height=ui_line_height, img_width=ui_img_width)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp.name)
                if ok: 
                    with open(tmp.name, "rb") as f: st.download_button("📥 下载 PDF", f, "Translated.pdf")
                else: st.error(msg)

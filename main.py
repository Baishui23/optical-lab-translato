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

# --- 0. 配置部分 ---
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    API_KEY = "sk-xxxxxxxx" # 本地测试用

BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.set_page_config(page_title="PDF学术翻译", page_icon="📄", layout="wide")

# --- 1. CSS 样式 (纯净版：修复字体乱码) ---
COMMON_CSS = """
<style>
    /* 1. 引入网络字体作为备份 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    body {
        /* 2. 关键：指定字体栈。优先使用 Linux 服务器上的 Noto CJK 或 文泉驿微米黑 */
        font-family: "Noto Serif SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimSun", "Arial", serif;
        font-size: 16px;
        line-height: 1.6;
        color: #000;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    /* 页面容器 - 去除花哨边框，只保留基本的学术版式 */
    .page-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 40px;
        background-color: #fff;
        text-align: justify;
    }

    /* 段落与标题 */
    p { margin-bottom: 1em; text-indent: 2em; }
    h1, h2, h3 { font-family: "Noto Serif SC", "SimHei", sans-serif; color: #111; margin-top: 1.5em; }

    /* 图片与图注 */
    img { max-width: 95%; display: block; margin: 20px auto; }
    .caption { 
        font-size: 14px; 
        color: #444; 
        text-align: center; 
        font-weight: bold; 
        margin-top: 5px; 
        margin-bottom: 25px; 
        font-family: sans-serif;
    }

    /* 分页控制 (打印时不显示分割线) */
    .page-break { 
        page-break-before: always; 
        border-top: 1px dashed #eee; 
        margin-top: 30px; 
        padding-top: 10px; 
        text-align: center; 
        color: #ccc; 
        font-size: 12px; 
    }
    .page-break.first-page { page-break-before: avoid; border: none; display: none; }
    
    @media print { 
        .page-break { border: none; color: transparent; margin: 0; height: 0; } 
        body { -webkit-print-color-adjust: exact; }
    }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑 (保持不变) ---
def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def is_header_or_footer(rect, page_height):
    if rect.y1 < 50: return True
    if rect.y0 > page_height - 50: return True
    return False

def is_caption_node(text):
    text = text.strip()
    return text.startswith("Fig.") or (text.startswith("Figure") and re.match(r'^Figure\s?\d+[.:]', text))

def translate_text(text, is_caption=False):
    if len(text.strip()) < 2: return text
    sys_prompt = """你是一个专业的物理学术翻译。请将文本翻译成流畅的学术中文。
    【规则】
    1. 保持学术严谨性。
    2. 公式必须用 $...$ 或 $$...$$ 包裹。
    3. 直接输出译文，不要加任何前缀或解释。
    """
    if is_caption: sys_prompt += " (这是图注，请保留 Figure 编号)"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except: return text

def capture_image_between_blocks(page, prev_bottom, current_top):
    if current_top - prev_bottom < 40: return None
    rect = fitz.Rect(50, prev_bottom + 5, page.rect.width - 50, current_top - 5)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img if img.size[1] >= 20 else None
    except: return None

def parse_page(page):
    elements = []
    blocks = page.get_text("blocks", sort=True)
    last_bottom = 0
    text_buffer = ""
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        b_top = b_rect.y0
        if i == 0 and last_bottom == 0: last_bottom = b_top

        if is_caption_node(b[4]):
            if text_buffer.strip():
                elements.append({'type': 'text', 'content': translate_text(text_buffer)})
                text_buffer = ""
            img = capture_image_between_blocks(page, last_bottom, b_top)
            if img: elements.append({'type': 'image', 'content': img})
            elements.append({'type': 'caption', 'content': translate_text(b[4], True)})
        else:
            text_buffer += b[4] + "\n\n"
        last_bottom = b_rect.y1
        
    if text_buffer.strip():
        elements.append({'type': 'text', 'content': translate_text(text_buffer)})
    return elements

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (纯净版：移除所有封面和Metadata) ---
def generate_full_html(all_pages_data, filename="Document"):
    # 直接开始，不加 Header
    html_body = f'<div class="page-container">'
    
    for idx, page_els in enumerate(all_pages_data):
        # 页面标记仅用于调试，打印时会隐藏
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- {idx+1} -</div>'
        
        for el in page_els:
            if el['type'] == 'text':
                paras = clean_latex(el['content']).split('\n\n')
                for p in paras:
                    if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
            elif el['type'] == 'image':
                html_body += f'<img src="{image_to_base64(el["content"])}" />'
            elif el['type'] == 'caption':
                html_body += f'<div class="caption">{el["content"]}</div>'
                
    html_body += "</div>"
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{COMMON_CSS}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

# --- 4. PDF 引擎 (兼容版) ---
def get_chrome_path():
    if shutil.which("chromium"): return shutil.which("chromium")
    if shutil.which("chromium-browser"): return shutil.which("chromium-browser")
    mac_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for p in mac_paths: 
        if os.path.exists(p): return p
    win_paths = [r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for p in win_paths: 
        if os.path.exists(p): return p
    return None

def html_to_pdf_with_chrome(html_content, output_pdf_path):
    chrome_bin = get_chrome_path()
    if not chrome_bin:
        return False, "❌ 未找到浏览器核心，请检查 packages.txt 是否配置"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    cmd = [
        chrome_bin, "--headless", "--disable-gpu", 
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer", # 确保无页眉页脚
        "--virtual-time-budget=8000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 (侧边栏也简化了) ---
st.title("📄 学术论文翻译 (PDF)")

with st.sidebar:
    st.markdown("### 🛠️ 功能面板")
    uploaded_file = st.file_uploader("上传文件", type="pdf")
    st.markdown("---")
    mode = st.radio("选择模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**原文**")
            pix = doc[page_num-1].get_pixmap(matrix=fitz.Matrix(2,2))
            st.image(pix.tobytes("png"), use_container_width=True)
        with c2:
            st.markdown("**译文**")
            if st.session_state.get('run_preview'):
                with st.spinner("正在翻译..."):
                    els = parse_page(doc[page_num-1])
                    preview_html = generate_full_html([els])
                    components.html(preview_html, height=800, scrolling=True)

    else:
        st.info("批量导出模式")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        if st.button("开始生成 PDF", type="primary"):
            data = []
            bar = st.progress(0)
            status = st.empty()
            
            for i, p in enumerate(range(start, end + 1)):
                status.text(f"正在处理第 {p} 页...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在合成文档...")
            full_html = generate_full_html(data, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    status.success("完成！")
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 点击下载 PDF", f, "Translated_Paper.pdf")
                else:
                    st.error(f"生成失败: {msg}")

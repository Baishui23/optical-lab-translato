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

# --- 0. 配置部分 ---
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    API_KEY = "sk-xxxxxxxx" 

BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (V36: 强制横向 + 安全边距) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    /* === 核心修复：打印页面设置 === */
    @page {
        size: A4 landscape; /* 强制横向 A4 */
        margin: 15mm;       /* 关键：留出 1.5cm 的安全边距，防止切边 */
    }

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 14px; /* 横向排版，字体可以稍微精细一点 */
        line-height: 1.6;
        color: #111;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    .page-container {
        width: 100%;
        /* max-width: 1200px;  <-- 删掉这个限制，让它自适应横向纸张 */
        margin: 0 auto;
        padding: 0; /* padding 交给 @page 管理 */
    }

    /* === 左右对照布局 === */
    .split-layout {
        display: flex;
        flex-direction: row;
        gap: 30px; /* 增加间距，因为横向空间大 */
        margin-bottom: 30px;
        align-items: flex-start;
        border-bottom: 1px dashed #ccc;
        padding-bottom: 30px;
        page-break-inside: avoid; /* 尽量不要把一组对照切断 */
    }

    .left-col-image {
        width: 48%; /* 稍微留点余地，不要占满 50% */
        flex-shrink: 0;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        border-radius: 4px;
        overflow: hidden; /* 防止图片溢出框 */
    }
    
    .left-col-image img {
        width: 100%;
        height: auto;
        display: block;
    }

    .right-col-text {
        width: 52%; /* 文字部分稍微宽一点点 */
        padding-left: 10px;
        text-align: justify;
    }

    /* === 纯净模式样式 === */
    .pure-mode-container {
        max-width: 900px; /* 纯净模式还是回到中间比较好看 */
        margin: 0 auto;
    }
    .pure-mode-container p { margin-bottom: 1em; text-indent: 2em; }
    .pure-mode-container img { max-width: 80%; display: block; margin: 20px auto; }

    .caption { font-size: 13px; color: #555; text-align: center; font-weight: bold; margin-bottom: 20px; font-family: sans-serif;}

    .page-marker {
        text-align: center; font-size: 12px; color: #aaa; 
        margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;
    }
    .page-break { page-break-before: always; }
    .page-break.first-page { page-break-before: avoid; }
    
    @media print { 
        /* 打印时隐藏不必要的元素 */
        .page-break { height: 0; margin: 0; }
    }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑 (保持 V35 的并发极速版) ---

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def is_header_or_footer(rect, page_height):
    return rect.y1 < 50 or rect.y0 > page_height - 50

def is_caption_node(text):
    text = text.strip()
    return text.startswith("Fig.") or (text.startswith("Figure") and re.match(r'^Figure\s?\d+[.:]', text))

def translate_text(text, is_caption=False):
    if len(text.strip()) < 2: return text
    sys_prompt = """你是一个专业的物理学术翻译。
    【规则】
    1. 保持学术严谨性。
    2. 公式必须用 $...$ 或 $$...$$ 包裹。
    3. 直接输出译文，不要加前缀。
    """
    if is_caption: sys_prompt += " (这是图注，保留编号)"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except: return text

# 并发翻译
def batch_translate_elements(elements):
    tasks = []
    indices = []
    for i, el in enumerate(elements):
        if el['type'] in ['text', 'caption']:
            tasks.append((el['content'], el['type'] == 'caption'))
            indices.append(i)
    
    if not tasks: return elements

    # 5线程并发
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))
    
    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]
    return elements

def capture_image_between_blocks(page, prev_bottom, current_top):
    if current_top - prev_bottom < 40: return None
    rect = fitz.Rect(50, prev_bottom + 5, page.rect.width - 50, current_top - 5)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img if img.size[1] >= 20 else None
    except: return None

def parse_page(page):
    raw_elements = []
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
                raw_elements.append({'type': 'text', 'content': text_buffer})
                text_buffer = ""
            img = capture_image_between_blocks(page, last_bottom, b_top)
            if img: raw_elements.append({'type': 'image', 'content': img})
            raw_elements.append({'type': 'caption', 'content': b[4]})
        else:
            text_buffer += b[4] + "\n\n"
        last_bottom = b_rect.y1
        
    if text_buffer.strip():
        raw_elements.append({'type': 'text', 'content': text_buffer})
        
    return batch_translate_elements(raw_elements)

def get_page_image(page):
    # 稍微降低一点点分辨率以提高速度，matrix=2 足够清晰了
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 ---
def generate_html(doc, start, end, mode="pure", filename="Document"):
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        
        # 分页标记
        marker_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{marker_class}"><div class="page-marker">- 第 {page_num} 页 -</div></div>'
        
        if mode == "screenshot":
            # === 左图右文 (截图模式) ===
            # 这里不需要 translate 整个 page 的 text，只需要右边的 text
            # 为了简单复用，我们还是调 parse_page，虽然它会提取一些小图，但我们在截图模式下不显示小图
            page_els = parse_page(page) 
            img_b64 = image_to_base64(get_page_image(page))
            
            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image">
                    <img src="{img_b64}" />
                </div>
                <div class="right-col-text">
            """
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">图注: {el["content"]}</div>'
            html_body += "</div></div>"
            
        else:
            # === 纯净模式 ===
            # 纯净模式下，我们希望它是纵向的，所以这里用 CSS override 一下 @page 可能会比较复杂
            # 简单起见，我们在 html_body 里包一层
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
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{COMMON_CSS}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

# --- 4. PDF 引擎 ---
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
    if not chrome_bin: return False, "❌ 未找到浏览器核心"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    cmd = [
        chrome_bin, "--headless", "--disable-gpu", 
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer", 
        "--virtual-time-budget=5000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 ---
st.title("🔬 光学室学术论文翻译专用版 (防切边修复版)")

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
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    if app_mode == "🖨️ 导出 PDF":
        st.markdown("##### 📄 导出格式")
        export_style = st.radio("选择风格：", ["纯净译文版 (V27经典)", "中英对照版 (左图右文)"], index=1) # 默认选中对照

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
             with st.spinner("🚀 极速解析中..."):
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot")
                components.html(preview_html, height=800, scrolling=True)
        else:
             st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        style_code = "screenshot" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成防切边 PDF", type="primary"):
            bar = st.progress(0)
            status = st.empty()
            
            status.text("正在并发翻译...")
            full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name)
            
            status.text("正在生成 PDF (已强制横向)...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    status.success("✅ 完成！")
                    fname = "Translation_Landscape.pdf" if style_code == "screenshot" else "Translation_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载文件", f, fname)
                else:
                    st.error(f"失败: {msg}")

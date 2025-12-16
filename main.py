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

# --- 1. CSS 生成器 (V37: 动态注入排版参数) ---
def get_css(font_size, line_height, img_width_pct):
    text_width_pct = 100 - img_width_pct - 2 # 预留2%间隙
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

        @page {{
            size: A4 landscape;
            margin: 15mm; 
        }}

        body {{
            font-family: "Noto Serif SC", "SimSun", serif;
            font-size: {font_size}px;  /* 动态字号 */
            line-height: {line_height}; /* 动态行高 */
            color: #111;
            margin: 0;
            padding: 0;
            background-color: white;
        }}

        .page-container {{
            width: 100%;
            margin: 0 auto;
        }}

        /* === 左右对照布局 (动态宽度) === */
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
            width: {img_width_pct}%;  /* 动态图片列宽 */
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
            width: {text_width_pct}%; /* 动态文字列宽 */
            padding-left: 5px;
            text-align: justify;
            overflow-wrap: break-word; /* 防止长单词撑破 */
        }}
        
        /* 针对公式的微调 */
        .MathJax {{
            font-size: 100% !important; /* 跟随正文字号 */
        }}

        /* === 纯净模式 === */
        .pure-mode-container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .pure-mode-container p {{ margin-bottom: 1em; text-indent: 2em; }}
        .pure-mode-container img {{ max-width: 80%; display: block; margin: 20px auto; }}

        .caption {{ 
            font-size: {font_size - 2}px; /* 图注比正文小一点 */
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

# --- 2. 核心逻辑 (保持并发极速版) ---

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
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (接收排版参数) ---
def generate_html(doc, start, end, mode="pure", filename="Document", font_size=14, line_height=1.6, img_width=50):
    
    # 动态生成 CSS
    dynamic_css = get_css(font_size, line_height, img_width)
    
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        marker_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{marker_class}"><div class="page-marker">- 第 {page_num} 页 -</div></div>'
        
        if mode == "screenshot":
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
st.title("🔬 光学室学术论文翻译专用版 (V37 排版大师)")

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
    # === 新增排版控制区 ===
    with st.expander("🎨 排版设置 (防溢出)", expanded=True):
        ui_font_size = st.slider("字体大小 (px)", 10, 18, 14, help="字太多时调小一点，推荐 12-14")
        ui_line_height = st.slider("行间距 (Line Height)", 1.2, 2.0, 1.6, 0.1, help="越小越紧凑")
        ui_img_width = st.slider("左图占比 (%)", 30, 70, 48, help="字多的时候，把图片占比调小，给文字腾地方")

    st.markdown("---")
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    if app_mode == "🖨️ 导出 PDF":
        export_style = st.radio("导出风格：", ["纯净译文版", "中英对照版 (左图右文)"], index=1)

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
                # 传入排版参数
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot", 
                                             font_size=ui_font_size, 
                                             line_height=ui_line_height,
                                             img_width=ui_img_width)
                components.html(preview_html, height=800, scrolling=True)
        else:
             st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        style_code = "screenshot" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成 PDF", type="primary"):
            bar = st.progress(0)
            status = st.empty()
            
            status.text("正在并发翻译...")
            # 传入排版参数
            full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name,
                                      font_size=ui_font_size,
                                      line_height=ui_line_height,
                                      img_width=ui_img_width)
            
            status.text("正在生成 PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    status.success("✅ 完成！")
                    fname = "Translation_Custom.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载文件", f, fname)
                else:
                    st.error(f"失败: {msg}")

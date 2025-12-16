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

# --- 0. 基础配置 ---
st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

BASE_URL = "https://api.deepseek.com"

# --- 1. CSS 样式 (V27 经典布局) ---
def get_css(font_size, line_height, img_width_pct):
    text_width_pct = 100 - img_width_pct - 2
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

        @page {{
            size: A4 landscape;
            margin: 10mm; 
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
            page-break-after: always; /* 每一页强制换页 */
        }}
        
        .page-marker {{
            text-align: center; font-size: 12px; color: #aaa; 
            margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;
        }}

        /* 核心布局：左图右文 */
        .split-layout {{
            display: flex;
            flex-direction: row;
            gap: 20px;
            align-items: flex-start;
            height: 100%;
        }}

        /* 左侧：整页截图 */
        .left-col-image {{
            width: {img_width_pct}%;
            flex-shrink: 0;
            border: 1px solid #ddd;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }}
        
        .left-col-image img {{ 
            width: 100%; 
            height: auto; 
            display: block; 
        }}

        /* 右侧：纯译文 */
        .right-col-text {{
            width: {text_width_pct}%;
            padding: 10px;
            text-align: justify;
            overflow-wrap: break-word;
        }}
        
        .right-col-text p {{
            margin-bottom: 1.2em;
            text-indent: 2em;
        }}

        .MathJax {{ font-size: 100% !important; }}
    </style>
    """

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心处理逻辑 ---

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG") 
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def translate_text(client, text):
    if len(text.strip()) < 5: return text 
    # V27 经典 Prompt
    sys_prompt = "你是一个物理学翻译专家。直接翻译以下学术文本，保持专业术语准确。公式保留原样（使用$$或$包裹）。不要解释，直接输出译文。"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[翻译错误: {str(e)}]"

def clean_latex(text):
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    return text

def process_page_v27(page, client):
    # 1. 左侧：整页截图
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_b64 = image_to_base64(img)

    # 2. 右侧：提取文本
    page_height = page.rect.height
    blocks = page.get_text("blocks", sort=True)
    valid_text_blocks = []
    
    for b in blocks:
        y0, y1, text = b[1], b[3], b[4]
        if y0 < 50 or y1 > page_height - 50: continue
        if len(text.strip()) < 3: continue
        valid_text_blocks.append(text)

    # 3. 逐个翻译 (单线程，稳！)
    translated_paragraphs = []
    if client and valid_text_blocks:
        progress_text = st.empty()
        for i, text in enumerate(valid_text_blocks):
            progress_text.text(f"正在翻译第 {i+1}/{len(valid_text_blocks)} 段...")
            translated_paragraphs.append(translate_text(client, text))
        progress_text.empty()
    else:
        translated_paragraphs = valid_text_blocks 

    return img_b64, translated_paragraphs

def generate_html_document(doc, start_page, end_page, client, font_size, line_height, img_width):
    css = get_css(font_size, line_height, img_width)
    html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}{MATHJAX_SCRIPT}</head><body>"
    
    progress_bar = st.progress(0)
    total = end_page - start_page + 1
    
    for idx, page_num in enumerate(range(start_page, end_page + 1)):
        page = doc[page_num - 1]
        img_b64, paragraphs = process_page_v27(page, client)
        
        html_content += f"""
        <div class="page-container">
            <div class="page-marker">- 第 {page_num} 页 -</div>
            <div class="split-layout">
                <div class="left-col-image"><img src="{img_b64}" /></div>
                <div class="right-col-text">
        """
        for p in paragraphs:
            clean_p = clean_latex(p).replace('\n', ' ')
            if clean_p.strip(): html_content += f"<p>{clean_p}</p>"
                
        html_content += "</div></div></div>"
        progress_bar.progress((idx + 1) / total)
    
    html_content += "</body></html>"
    return html_content

# --- 4. PDF 导出 ---
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

def html_to_pdf(html_str, output_path):
    chrome = get_chrome_path()
    if not chrome: return False, "未找到 Chrome 浏览器"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        f.write(html_str)
        tmp_html = f.name
        
    cmd = [
        chrome, "--headless", "--disable-gpu", 
        f"--print-to-pdf={output_path}",
        "--no-pdf-header-footer",
        f"file://{tmp_html}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")
    
    try:
        subprocess.run(cmd, check=True, timeout=60)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 ---
st.title("🔬 光学室学术论文翻译 (V27 上午原版)")
st.markdown("**这是你觉得最稳的版本：侧边栏填 Key，单线程慢速翻译，绝不丢图。**")

with st.sidebar:
    api_key = st.text_input("输入 DeepSeek API Key", type="password")
    client = OpenAI(api_key=api_key, base_url=BASE_URL) if api_key else None
    
    uploaded_file = st.file_uploader("上传 PDF", type="pdf")
    
    st.markdown("---")
    font_size = st.slider("字体大小", 10, 20, 13)
    line_height = st.slider("行高", 1.0, 2.0, 1.5)
    img_width = st.slider("左侧原图占比 (%)", 20, 80, 50)

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    st.success(f"PDF 加载成功，共 {len(doc)} 页")
    
    col1, col2 = st.columns(2)
    with col1: start = st.number_input("起始页", 1, len(doc), 1)
    with col2: end = st.number_input("结束页", 1, len(doc), min(len(doc), 5))
    
    if st.button("🚀 开始翻译并生成 PDF"):
        if not client:
            st.error("请先输入 API Key！")
        else:
            with st.status("正在处理 (单线程模式，请耐心等待)...", expanded=True) as status:
                html_result = generate_html_document(doc, start, end, client, font_size, line_height, img_width)
                
                st.write("正在生成 PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_tmp:
                    success, msg = html_to_pdf(html_result, pdf_tmp.name)
                    
                if success:
                    status.update(label="✅ 完成！", state="complete", expanded=False)
                    with open(pdf_tmp.name, "rb") as f:
                        st.download_button("📥 下载 PDF", f, "Translation_V27_Original.pdf")
                else:
                    status.update(label="❌ 失败", state="error")
                    st.error(f"PDF 生成失败: {msg}")

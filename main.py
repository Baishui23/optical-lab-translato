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
    API_KEY = "sk-xxxxxxxx" # 本地测试请填入真实Key

BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (V33: 左右分栏，左图右文) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 14px;
        color: #111;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    /* 页面容器：为了容纳双栏，我们设置得宽一点 */
    .page-container {
        width: 100%;
        max-width: 1200px; 
        margin: 0 auto;
        padding: 20px;
    }

    /* === V33 核心布局：所见即所得 === */
    .split-layout {
        display: flex;
        flex-direction: row;
        gap: 20px; /* 左右间距 */
        margin-bottom: 30px;
        align-items: flex-start; /* 顶部对齐 */
        border-bottom: 1px dashed #ccc; /* 页与页之间的分割线 */
        padding-bottom: 30px;
    }

    /* 左栏：PDF原图截图 */
    .left-col-image {
        width: 50%;
        flex-shrink: 0; /* 防止被挤压 */
        border: 1px solid #eee;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    
    .left-col-image img {
        width: 100%;
        display: block;
    }

    /* 右栏：中文译文 */
    .right-col-text {
        width: 50%;
        padding-left: 10px;
        text-align: justify;
        line-height: 1.6;
    }

    /* 纯净模式 */
    .pure-text p { margin-bottom: 1em; text-indent: 2em; }

    /* 分页标记 */
    .page-marker {
        text-align: center;
        font-size: 12px;
        color: #999;
        margin-bottom: 10px;
        font-weight: bold;
    }

    /* 打印控制 */
    @media print { 
        .page-container { width: 100%; max-width: none; padding: 0; }
        .split-layout { page-break-inside: avoid; } /* 尽量不要把一页切成两半 */
    }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑 ---

def clean_pdf_text(text):
    # 简单的清洗，用于发给AI
    text = text.replace('-\n', '')
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text, is_caption=False):
    if len(text) < 5: return text # 太短不翻
    sys_prompt = "你是一个物理学术翻译。将英文翻译成中文。保持公式LaTeX格式 $...$。直接输出译文。"
    if is_caption: sys_prompt += " (这是图注)"
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except: return text

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def is_header_or_footer(rect, page_height):
    if rect.y1 < 50: return True
    if rect.y0 > page_height - 50: return True
    return False

def parse_page_content_only(page):
    """
    只提取文字内容用于翻译，不关心排版，因为排版看左边的图就行了。
    """
    blocks = page.get_text("blocks", sort=True)
    text_buffer = ""
    translations = []
    
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for b in valid_blocks:
        text_buffer += b[4] + "\n"
    
    # 简单按段落处理翻译
    paras = text_buffer.split('\n\n')
    for p in paras:
        cleaned = clean_pdf_text(p)
        if len(cleaned) > 10:
            translations.append(translate_text(cleaned))
            
    return translations

def get_page_image(page):
    """
    获取页面高清截图
    """
    # matrix=2 意味着放大2倍，保证PDF里看清楚
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (V33: 截图对照布局) ---
def generate_html(doc, start_page, end_page, mode="screenshot_compare", filename="Doc"):
    html_body = f'<div class="page-container">'
    
    # 循环处理每一页
    for page_num in range(start_page, end_page + 1):
        page = doc[page_num-1]
        
        # 1. 顶部页码标记
        html_body += f'<div class="page-marker">- 第 {page_num} 页 -</div>'
        
        if mode == "screenshot_compare":
            # === 左图右文模式 ===
            
            # 左边：获取页面截图
            page_img = get_page_image(page)
            img_b64 = image_to_base64(page_img)
            
            # 右边：获取翻译文本
            trans_paras = parse_page_content_only(page)
            
            # 构建 HTML 结构
            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image">
                    <img src="{img_b64}" />
                </div>
                <div class="right-col-text">
            """
            # 填充右侧译文
            for p in trans_paras:
                p_latex = clean_latex(p)
                html_body += f"<p>{p_latex}</p>"
                
            html_body += """
                </div>
            </div>
            """
            
        else:
            # === 纯译文模式 (旧逻辑) ===
            trans_paras = parse_page_content_only(page)
            for p in trans_paras:
                 p_latex = clean_latex(p)
                 html_body += f"<p>{p_latex}</p>"
            html_body += "<hr>"

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
    if not chrome_bin:
        return False, "❌ 未找到浏览器核心"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    cmd = [
        chrome_bin, "--headless", "--disable-gpu", 
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer", 
        # 设置宽页面，以适应双栏 (A4横向近似宽度)
        "--print-to-pdf-no-header",
        "--virtual-time-budget=10000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 ---
st.title("🔬 光学室学术论文翻译专用版")

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
        export_style = st.radio(
            "排版风格：",
            ["左右对照 (左图右文)", "纯净译文 (仅中文)"], 
            index=0
        )

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if app_mode == "👁️ 实时预览":
        # 实时预览本身就是 左图右文，所以这里直接复用逻辑
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        if st.session_state.get('run_preview'):
             with st.spinner("生成预览中..."):
                # 复用 generate_html 生成单页预览
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot_compare")
                components.html(preview_html, height=800, scrolling=True)
        else:
             st.info("👈 点击“翻译此页”查看效果")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        style_code = "screenshot_compare" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成 PDF ({export_style})", type="primary"):
            bar = st.progress(0)
            status = st.empty()
            status.text("正在截取页面并翻译...")
            
            # 这里不需要按页循环翻译了，因为 generate_html 内部会循环
            # 我们只是为了显示进度条，稍微假装一下
            bar.progress(50)
            
            full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name)
            
            bar.progress(80)
            status.text("正在调用浏览器打印...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    status.success("✅ 完成！")
                    fname = "Translation_Visual_Compare.pdf" if style_code == "screenshot_compare" else "Translation_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载所见即所得 PDF", f, fname)
                else:
                    st.error(f"失败: {msg}")

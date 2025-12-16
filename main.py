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

# V27 核心配置：Wide 布局以适应可能的双栏需求
st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (V27 核心样式 + 截图对照布局) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    body {
        /* V27 经典字体栈 */
        font-family: "Noto Serif SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimSun", serif;
        font-size: 16px;
        line-height: 1.6;
        color: #000;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    /* 页面容器 */
    .page-container {
        width: 100%;
        max-width: 1200px; /* 稍微宽一点以适应双栏 */
        margin: 0 auto;
        padding: 40px;
        background-color: #fff;
    }

    /* === 模式 1: V27 纯净模式 === */
    .pure-mode-container {
        max-width: 800px; /* 纯文本模式限制宽度，模拟A4 */
        margin: 0 auto;
        text-align: justify;
    }
    .pure-mode-container p { margin-bottom: 1em; text-indent: 2em; }
    .pure-mode-container img { max-width: 95%; display: block; margin: 20px auto; }

    /* === 模式 2: 左图右文对照模式 === */
    .split-layout {
        display: flex;
        flex-direction: row;
        gap: 25px;
        margin-bottom: 40px;
        border-bottom: 1px dashed #ccc;
        padding-bottom: 20px;
    }
    .left-col-image {
        width: 50%;
        flex-shrink: 0;
        border: 1px solid #eee;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .left-col-image img { width: 100%; display: block; }
    
    .right-col-text {
        width: 50%;
        padding-left: 10px;
        text-align: justify;
    }
    .right-col-text p { margin-bottom: 1em; text-indent: 0; } /* 对照模式不缩进，显得整齐 */

    /* 通用样式 */
    .caption { 
        font-size: 14px; color: #444; text-align: center; 
        font-weight: bold; margin-bottom: 25px; font-family: sans-serif;
    }

    /* 分页控制 */
    .page-break { 
        page-break-before: always; border-top: 1px dashed #eee; 
        margin-top: 30px; padding-top: 10px; text-align: center; 
        color: #ccc; font-size: 12px; 
    }
    .page-break.first-page { page-break-before: avoid; display: none; }
    
    @media print { 
        .page-break { border: none; color: transparent; margin: 0; height: 0; } 
    }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑 (V27 原版逻辑) ---
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
    # V27 的经典 Prompt，强调公式和严谨
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

# V27 的经典解析函数
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

# 新增：获取全页截图 (用于对照模式)
def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

# --- 3. HTML 构建器 (混合 V27 和 截图模式) ---
def generate_html(doc, start, end, mode="pure", filename="Document"):
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        
        # 解析页面内容 (使用 V27 逻辑)
        page_els = parse_page(page)
        
        # 分页标记
        page_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{page_class}">- {page_num} -</div>'
        
        if mode == "screenshot":
            # === 模式2: 截图对照 (V33 理念) ===
            # 左边：整页原图
            img_b64 = image_to_base64(get_page_image(page))
            
            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image">
                    <img src="{img_b64}" />
                </div>
                <div class="right-col-text">
            """
            
            # 右边：V27 解析出的纯文本 (忽略提取的小图，因为左边大图里都有)
            for el in page_els:
                if el['type'] == 'text':
                    paras = clean_latex(el['content']).split('\n\n')
                    for p in paras:
                        if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">图注: {el["content"]}</div>'
            
            html_body += "</div></div>"
            
        else:
            # === 模式1: 纯净 V27 模式 ===
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

# --- 4. PDF 引擎 (保持不变) ---
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
        "--virtual-time-budget=10000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 (V27 风格 + 截图模式选项) ---
st.title("🔬 光学室学术论文翻译专用版")

with st.sidebar:
    # V27 经典署名
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
        # 这里集成了 V27 的纯净版 和 V33 的截图对照版
        export_style = st.radio(
            "选择风格：",
            ["纯净译文版 (V27经典)", "中英对照版 (左图右文)"], 
            index=0
        )

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if app_mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        # 预览界面保持 V33 的左图右文逻辑，因为这样最直观
        if st.session_state.get('run_preview'):
             with st.spinner("V27 内核正在解析..."):
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot")
                components.html(preview_html, height=800, scrolling=True)
        else:
             st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        # 逻辑判断
        style_code = "screenshot" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成 PDF ({export_style})", type="primary"):
            bar = st.progress(0)
            status = st.empty()
            
            # 使用 generate_html 内部循环处理
            status.text("正在使用 V27 内核解析并渲染...")
            bar.progress(50)
            
            full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name)
            
            bar.progress(80)
            status.text("正在调用浏览器生成 PDF...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    bar.progress(100)
                    status.success("✅ 完成！")
                    fname = "Translation_Visual.pdf" if style_code == "screenshot" else "Translation_V27_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载文件", f, fname)
                else:
                    st.error(f"失败: {msg}")import streamlit as st
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

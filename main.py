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
import datetime
import shutil
import platform
import streamlit.components.v1 as components

# --- 0. 配置部分 (自动兼容本地和云端) ---
try:
    # 优先尝试从 Streamlit Cloud 的 Secrets 读取 Key
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    # ⚠️ 如果你在本地运行，请在这里填入你的 Key
    API_KEY = "sk-0652dfc17b3544acb48bccb2f5f225a8"

BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (保留 V24 的全部美化) ---
COMMON_CSS = """
<style>
    /* 引入学术字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;600;900&family=Times+New+Roman&display=swap');

    body {
        font-family: "Times New Roman", "Noto Serif SC", "SimSun", serif;
        font-size: 16px;
        line-height: 1.8;
        color: #222;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    /* 页面容器 */
    .page-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 40px 60px;
        background-color: #fff;
        text-align: justify; 
    }

    /* --- 封面样式 --- */
    .cover-page-container {
        text-align: center;
        margin-bottom: 60px;
        border-bottom: 3px double #333;
        padding-bottom: 30px;
    }
    
    .lab-title {
        font-family: "Noto Serif SC", serif;
        font-weight: 900;
        font-size: 28px;
        color: #1a1a1a;
        margin-bottom: 10px;
        letter-spacing: 2px;
    }

    .doc-title {
        font-size: 20px;
        color: #444;
        margin-top: 20px;
        margin-bottom: 30px;
        font-weight: bold;
    }

    .meta-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        display: inline-block;
        border: 1px solid #eee;
        font-size: 14px;
        color: #555;
        line-height: 1.6;
        min-width: 300px;
        text-align: left;
    }

    /* --- 正文样式 --- */
    p { margin-bottom: 1.2em; text-indent: 2em; }
    img { max-width: 90%; display: block; margin: 20px auto; border: 1px solid #eee; border-radius: 2px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .caption { font-size: 14px; color: #555; text-align: center; font-weight: 600; margin-top: -10px; margin-bottom: 25px; font-family: sans-serif; }

    /* 分页控制 */
    .page-break { page-break-before: always; border-top: 1px dashed #ddd; margin-top: 30px; padding-top: 30px; text-align: center; color: #999; font-size: 12px; }
    .page-break.first-page { page-break-before: avoid; border-top: none; margin-top: 0; padding-top: 0; display: none; }
    @media print { .page-break { border-top: none; color: transparent; margin: 0; padding: 0; height: 0; } }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] }, svg: { fontCache: 'global' } };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑函数 (V24 逻辑) ---

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
    【关键排版规则】
    1. **绝对禁止**使用 \[ \] 或 \( \) 来包裹公式。
    2. **必须**使用 $ ... $ 包裹行内公式。
    3. **必须**使用 $$ ... $$ 包裹独立公式。
    4. 保持段落结构，不要随意合并段落。
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

# --- 3. HTML 构建器 (包含 V24 的封面逻辑) ---
def generate_full_html(all_pages_data, filename="Document"):
    # 生成时间
    now_str = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 封面 HTML (光学室专用版)
    header_html = f"""
    <div class="page-container">
        <div class="cover-page-container">
            <div class="lab-title">🔬 光学室学术论文翻译专用版</div>
            <div class="doc-title">{filename}</div>
            <div class="meta-box">
                <div><strong>翻译制作：</strong> 白水</div>
                <div><strong>微信号：</strong> guo21615</div>
                <div><strong>生成日期：</strong> {now_str}</div>
                <div><strong>引擎支持：</strong> DeepSeek V3</div>
            </div>
        </div>
    """
    
    html_body = header_html
    
    for idx, page_els in enumerate(all_pages_data):
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- 第 {idx+1} 页 -</div>'
        
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

# --- 4. 智能全平台 PDF 引擎 (兼容 Cloud/Mac/Win) ---
def get_chrome_path():
    """智能查找 Chrome/Chromium 路径"""
    # 1. 检测 Streamlit Cloud (Linux) 的 Chromium
    if shutil.which("chromium"): return shutil.which("chromium")
    if shutil.which("chromium-browser"): return shutil.which("chromium-browser")
    
    # 2. 检测 Mac
    mac_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    ]
    for p in mac_paths:
        if os.path.exists(p): return p
        
    # 3. 检测 Windows
    win_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    ]
    for p in win_paths:
        if os.path.exists(p): return p
        
    return None

def html_to_pdf_with_chrome(html_content, output_pdf_path):
    chrome_bin = get_chrome_path()
    
    if not chrome_bin:
        # 针对 Streamlit Cloud 的特殊提示
        if platform.system() == "Linux":
            return False, "❌ 未找到 Chromium。请检查 packages.txt 是否包含 'chromium'。"
        return False, "❌ 未找到 Chrome 或 Edge 浏览器。"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    # 构建命令
    cmd = [
        chrome_bin,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000", 
        f"file://{tmp_html_path}"
    ]
    
    # Linux/Cloud 环境必须加 --no-sandbox
    if platform.system() == "Linux":
        cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 (V24 的侧边栏) ---
st.title("🔬 光学室学术论文翻译专用版")

with st.sidebar:
    # V24 的个人名片
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
    mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出专用 PDF"])

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("选择页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("原文")
            pix = doc[page_num-1].get_pixmap(matrix=fitz.Matrix(2,2))
            st.image(pix.tobytes("png"), use_container_width=True)
            
        with c2:
            st.subheader("译文预览")
            if st.session_state.get('run_preview'):
                with st.spinner("光学室 AI 引擎正在解析..."):
                    els = parse_page(doc[page_num-1])
                    # 生成单页预览 HTML (带完整 CSS)
                    preview_html = generate_full_html([els], filename=f"Page {page_num}")
                    # 使用 components 渲染
                    components.html(preview_html, height=800, scrolling=True)
            else:
                st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        if st.button("🚀 生成专用 PDF", type="primary"):
            data = []
            bar = st.progress(0)
            status = st.empty()
            
            for i, p in enumerate(range(start, end + 1)):
                status.text(f"正在处理第 {p} 页...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在生成光学室专用报告...")
            full_html = generate_full_html(data, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                
                if ok:
                    status.success("✅ 报告生成完毕！")
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载翻译报告", f, "Optical_Lab_Report.pdf")
                else:
                    st.error(f"失败: {msg}")
                    st.download_button("📥 下载 HTML (Chrome调用失败时备用)", full_html, "report.html", "text/html")

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

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="centered") # 改回 centered

# --- 1. CSS 样式 (V32: 纯净单栏，专注于公式渲染) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 16px; /* 字体稍微加大，阅读更舒服 */
        line-height: 1.8; /* 行间距拉大，防止公式打架 */
        color: #111;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    .page-container {
        max-width: 800px; /* 限制阅读宽度，模拟A4纸质感 */
        margin: 0 auto;
        padding: 40px;
        background-color: #fff;
    }

    /* === 纯净译文样式 === */
    p {
        margin-bottom: 1.5em;
        text-align: justify;
        text-justify: inter-ideograph;
    }

    /* === 公式优化 === */
    /* 使得行内公式左右有点间隙 */
    mjx-container[jax="CHTML"][display="true"] {
        margin: 1em 0 !important;
    }

    /* 图片样式 */
    img { 
        max-width: 100%; 
        display: block; 
        margin: 20px auto; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* 加点阴影更好看 */
    }
    
    .caption { 
        font-size: 14px; 
        color: #555; 
        text-align: center; 
        font-weight: bold; 
        margin-top: -10px;
        margin-bottom: 30px; 
        font-family: sans-serif;
    }

    /* 分页符 */
    .page-break { 
        page-break-before: always; 
        border-top: 1px solid #eee; 
        margin-top: 30px; 
        padding-top: 20px; 
        text-align: center; 
        color: #bbb; 
        font-size: 12px; 
    }
    .page-break.first-page { page-break-before: avoid; display: none; }
    
    @media print { 
        .page-container { max-width: 100%; padding: 0; }
        .page-break { border: none; color: transparent; margin: 0; height: 0; } 
    }
</style>
"""

MATHJAX_SCRIPT = """
<script>
MathJax = { 
    tex: { 
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
    }, 
    svg: { fontCache: 'global' } 
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# --- 2. 核心逻辑 (V32: 专注于公式修复与文本清洗) ---

def clean_pdf_text(text):
    """
    清洗 PDF 文本，拼接断行，为 AI 提供连贯的输入。
    """
    # 1. 拼接连字符换行: "experi-\nment" -> "experiment"
    text = text.replace('-\n', '')
    # 2. 拼接普通换行: "This is\na test" -> "This is a test"
    text = text.replace('\n', ' ')
    # 3. 去除多余空格
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text, is_caption=False):
    if len(text) < 2: return text
    
    # --- V32 Prompt: 强调公式修复 ---
    sys_prompt = """你是一个专业的物理学术翻译助手。
    【任务】
    1. 将英文学术文本翻译成流畅、准确的中文。
    2. **高度重视数学公式**：PDF提取的公式可能支离破碎（如字符间有空格），请根据物理上下文修复它们，并使用标准 LaTeX 格式（行内用 $...$，独立公式用 $$...$$）。
    3. 保持学术用语的严谨性。
    4. 不要输出“好的”、“以下是翻译”等废话，直接输出译文。
    """
    if is_caption: sys_prompt += " (注意：这是一段图注，请保留 Figure 编号)"
    
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

def is_caption_node(text):
    text = text.strip()
    return text.startswith("Fig.") or (text.startswith("Figure") and re.match(r'^Figure\s?\d+[.:]', text))

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
    text_buffer = "" # 用于积攒文本段落
    
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        b_top = b_rect.y0
        if i == 0 and last_bottom == 0: last_bottom = b_top

        raw_text = b[4]
        
        # 遇到图注，说明上面的部分（图片+之前的文本）可以结算了
        if is_caption_node(raw_text):
            # 1. 先结算 buffer 里的正文
            if text_buffer.strip():
                cleaned = clean_pdf_text(text_buffer)
                trans = translate_text(cleaned, False)
                elements.append({'type': 'text', 'content': trans})
                text_buffer = ""
            
            # 2. 抓图片
            img = capture_image_between_blocks(page, last_bottom, b_top)
            if img: elements.append({'type': 'image', 'content': img})
            
            # 3. 处理图注
            cleaned_cap = clean_pdf_text(raw_text)
            trans_cap = translate_text(cleaned_cap, True)
            elements.append({'type': 'caption', 'content': trans_cap})
            
        else:
            # 普通文本，加入 buffer，攒在一起翻译以利用上下文修复公式
            text_buffer += raw_text + "\n"
        
        last_bottom = b_rect.y1
        
    # 页面结束，结算剩余文本
    if text_buffer.strip():
        cleaned = clean_pdf_text(text_buffer)
        trans = translate_text(cleaned, False)
        elements.append({'type': 'text', 'content': trans})

    return elements

def clean_latex(text):
    # 简单的 LaTeX 兼容性处理
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (V32: 纯文本流) ---
def generate_html(all_pages_data, filename="Doc"):
    html_body = f'<div class="page-container">'
    
    for idx, page_els in enumerate(all_pages_data):
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- 第 {idx+1} 页 -</div>'
        
        for el in page_els:
            if el['type'] == 'image':
                html_body += f'<img src="{image_to_base64(el["content"])}" />'
            
            elif el['type'] == 'caption':
                html_body += f'<div class="caption">{el["content"]}</div>'
            
            elif el['type'] == 'text':
                # 将翻译结果按段落分割，包裹 p 标签
                paras = clean_latex(el['content']).split('\n\n')
                for p in paras:
                    if p.strip():
                        # 去掉markdown的加粗，让排版更干净
                        clean_p = p.replace('**', '') 
                        html_body += f"<p>{clean_p}</p>"

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
st.title("🔬 光学室学术论文翻译专用版 (公式修复版)")

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

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if app_mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("原文")
            pix = doc[page_num-1].get_pixmap(matrix=fitz.Matrix(2,2))
            st.image(pix.tobytes("png"), use_container_width=True)
        with c2:
            st.subheader("纯净译文")
            if st.session_state.get('run_preview'):
                with st.spinner("AI 正在修复公式并翻译..."):
                    els = parse_page(doc[page_num-1])
                    preview_html = generate_html([els])
                    components.html(preview_html, height=800, scrolling=True)

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        if st.button("🚀 生成纯净译文 PDF", type="primary"):
            data = []
            bar = st.progress(0)
            status = st.empty()
            
            for i, p in enumerate(range(start, end + 1)):
                status.text(f"正在处理第 {p} 页 (公式重构中)...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在渲染文档...")
            full_html = generate_html(data, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    status.success("✅ 完成！")
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载纯净版 PDF", f, "Translation_Pure_Math.pdf")
                else:
                    st.error(f"失败: {msg}")

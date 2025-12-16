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
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # 请在此处填入你的 Key

BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (V28 原版样式) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Times+New+Roman&display=swap');

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 15px; 
        line-height: 1.6;
        color: #000;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    .page-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 40px;
        background-color: #fff;
    }

    /* === 纯净模式样式 === */
    .pure-text p { margin-bottom: 1em; text-indent: 2em; text-align: justify; }

    /* === 对照模式样式 (V28核心) === */
    .bilingual-row {
        display: flex;
        flex-direction: row;
        margin-bottom: 1.5em;
        border-bottom: 1px solid #f0f0f0; 
        padding-bottom: 1em;
    }
    
    .col-eng {
        flex: 1;
        padding-right: 20px;
        color: #555; 
        font-family: "Times New Roman", serif;
        text-align: justify;
        font-size: 14px;
        border-right: 2px solid #eee; 
    }
    
    .col-chn {
        flex: 1;
        padding-left: 20px;
        color: #000; 
        text-align: justify;
    }

    img { max-width: 95%; display: block; margin: 20px auto; }
    
    .caption { 
        font-size: 13px; color: #444; text-align: center; 
        font-weight: bold; margin-bottom: 25px; font-family: sans-serif;
    }

    .page-break { 
        page-break-before: always; border-top: 1px dashed #ccc; 
        margin-top: 30px; padding-top: 10px; text-align: center; 
        color: #999; font-size: 12px; 
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

# --- 2. 核心逻辑 ---

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
    3. 直接输出译文，不要加任何前缀。
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
    # V28 的原始阈值是 40
    if current_top - prev_bottom < 40: return None
    rect = fitz.Rect(50, prev_bottom + 5, page.rect.width - 50, current_top - 5)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img if img.size[1] >= 20 else None
    except: return None

# V28 核心：同时保存原文和译文
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
            # 翻译积压的文本
            if text_buffer.strip():
                trans = translate_text(text_buffer)
                elements.append({'type': 'text', 'original': text_buffer, 'translation': trans})
                text_buffer = ""
            
            # 抓图
            img = capture_image_between_blocks(page, last_bottom, b_top)
            if img: elements.append({'type': 'image', 'content': img})
            
            # 翻译图注
            caption_trans = translate_text(b[4], True)
            elements.append({'type': 'caption', 'original': b[4], 'translation': caption_trans})
        else:
            text_buffer += b[4] + "\n\n"
        last_bottom = b_rect.y1
        
    if text_buffer.strip():
        trans = translate_text(text_buffer)
        elements.append({'type': 'text', 'original': text_buffer, 'translation': trans})
    return elements

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 ---
def generate_html(all_pages_data, mode="pure", filename="Doc"):
    html_body = f'<div class="page-container">'
    
    for idx, page_els in enumerate(all_pages_data):
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- {idx+1} -</div>'
        
        for el in page_els:
            if el['type'] == 'image':
                html_body += f'<img src="{image_to_base64(el["content"])}" />'
            
            elif el['type'] == 'caption':
                if mode == "bilingual":
                    html_body += f"""
                    <div class="caption">
                        <span style="color:#666; font-size:0.9em;">[原文] {el['original']}</span><br>
                        <span>{el['translation']}</span>
                    </div>
                    """
                else:
                    html_body += f'<div class="caption">{el["translation"]}</div>'
            
            elif el['type'] == 'text':
                if mode == "bilingual":
                    # V28 特有的左右分栏
                    orig = clean_latex(el['original']).replace('\n', '<br>')
                    trans = clean_latex(el['translation']).replace('\n\n', '</p><p>')
                    
                    html_body += f"""
                    <div class="bilingual-row">
                        <div class="col-eng">{orig}</div>
                        <div class="col-chn"><p>{trans}</p></div>
                    </div>
                    """
                else:
                    paras = clean_latex(el['translation']).split('\n\n')
                    html_body += '<div class="pure-text">'
                    for p in paras:
                        if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
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
        "--virtual-time-budget=8000",
        f"file://{tmp_html_path}"
    ]
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# --- 5. 界面逻辑 ---
st.title("🔬 光学室学术论文翻译专用版 (V28 双模式)")

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
            "选择排版风格：",
            ["纯净译文版 (仅中文)", "中英对照版 (左英右中)"],
            index=1
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
        
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("原文")
            pix = doc[page_num-1].get_pixmap(matrix=fitz.Matrix(2,2))
            st.image(pix.tobytes("png"), use_container_width=True)
        with c2:
            st.subheader("译文预览")
            if st.session_state.get('run_preview'):
                with st.spinner("AI 解析中..."):
                    els = parse_page(doc[page_num-1])
                    preview_html = generate_html([els], mode="pure")
                    components.html(preview_html, height=800, scrolling=True)

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        style_code = "bilingual" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成 PDF ({export_style})", type="primary"):
            data = []
            bar = st.progress(0)
            status = st.empty()
            
            for i, p in enumerate(range(start, end + 1)):
                status.text(f"正在处理第 {p} 页...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在排版...")
            full_html = generate_html(data, mode=style_code, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    status.success("✅ 完成！")
                    fname = "Translation_Bilingual.pdf" if style_code == "bilingual" else "Translation_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载 PDF 文件", f, fname)
                else:
                    st.error(f"失败: {msg}")

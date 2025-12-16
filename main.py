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
from itertools import cycle

# ==========================================
# 👇 【关键配置】请在这里填入你的所有 Key 👇
#    (多填几个，程序会自动轮询，速度更快！)
# ==========================================
USER_KEYS = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
]
# ==========================================

BASE_URL = "https://api.deepseek.com"

# --- 初始化 Key 轮询器 ---
VALID_KEYS = [k.strip() for k in USER_KEYS if k.strip().startswith("sk-")]
# 如果代码里没填，尝试从 secrets 读取
if not VALID_KEYS:
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            VALID_KEYS = [st.secrets["DEEPSEEK_API_KEY"]]
    except:
        pass

KEY_CYCLE = cycle(VALID_KEYS) if VALID_KEYS else None

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式生成器 (支持 纯净版 & 对照版) ---
def get_css(mode="pure", font_size=16, line_height=1.6, img_width_pct=50):
    
    # 通用基础样式
    base_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');
    body {
        font-family: "Noto Serif SC", "Noto Sans CJK SC", "SimSun", serif;
        color: #000; margin: 0; padding: 0; background-color: white;
    }
    .caption { font-size: 14px; color: #444; text-align: center; font-weight: bold; margin-bottom: 15px; font-family: sans-serif; }
    .page-break { page-break-before: always; margin-top: 30px; border-top: 1px dashed #eee; padding-top: 10px; text-align: center; color: #ccc; font-size: 12px; }
    .page-break.first-page { page-break-before: avoid; display: none; }
    @media print { .page-break { border: none; color: transparent; margin: 0; height: 0; } }
    """

    if mode == "pure":
        # === 纯净版样式 (V27原版) ===
        return f"""
        <style>
            {base_css}
            @page {{ size: A4 portrait; margin: 20mm; }}
            body {{ font-size: {font_size}px; line-height: {line_height}; }}
            .page-container {{ max-width: 800px; margin: 0 auto; padding: 20px; text-align: justify; }}
            p {{ margin-bottom: 1em; text-indent: 2em; }}
            img {{ max-width: 95%; display: block; margin: 20px auto; }}
        </style>
        """
    else:
        # === 对照版样式 (左图右文) ===
        text_width_pct = 100 - img_width_pct - 2
        return f"""
        <style>
            {base_css}
            @page {{ size: A4 landscape; margin: 10mm; }} /* 强制横向 */
            body {{ font-size: {font_size - 2}px; line-height: {line_height}; }} /* 对照版字稍微小点 */
            .page-container {{ width: 100%; margin: 0 auto; }}
            
            .split-layout {{
                display: flex; flex-direction: row; gap: 20px;
                align-items: flex-start; margin-bottom: 30px;
                border-bottom: 1px dashed #ccc; padding-bottom: 30px;
                page-break-inside: avoid;
            }}
            .left-col-image {{
                width: {img_width_pct}%; flex-shrink: 0;
                border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            }}
            .left-col-image img {{ width: 100%; height: auto; display: block; }}
            .right-col-text {{
                width: {text_width_pct}%; padding-left: 10px;
                text-align: justify; overflow-wrap: break-word;
            }}
            .right-col-text p {{ margin-bottom: 1em; text-indent: 2em; }}
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

# === 获取 Client (轮询逻辑) ===
def get_next_client():
    if not KEY_CYCLE: return None
    return OpenAI(api_key=next(KEY_CYCLE), base_url=BASE_URL)

# === V27 核心翻译 Prompt ===
def translate_text(text, is_caption=False):
    if len(text.strip()) < 2: return text
    
    client = get_next_client()
    if not client: return "[Key未配置]"

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
    except Exception as e:
        print(f"Error: {e}")
        return text

def capture_image_between_blocks(page, prev_bottom, current_top):
    if current_top - prev_bottom < 40: return None
    rect = fitz.Rect(50, prev_bottom + 5, page.rect.width - 50, current_top - 5)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img if img.size[1] >= 20 else None
    except: return None

# === 多核并发翻译器 ===
def batch_translate_elements(elements):
    tasks = []
    indices = []
    
    # 1. 收集所有需要翻译的文本
    for i, el in enumerate(elements):
        if el['type'] in ['text', 'caption']:
            tasks.append((el['content'], el['type'] == 'caption'))
            indices.append(i)
    
    if not tasks: return elements

    # 2. 线程池并发 (线程数根据 Key 数量自动调整)
    max_workers = 8 if len(VALID_KEYS) >= 3 else 4
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))
    
    # 3. 填回结果
    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]
        
    return elements

def parse_page(page):
    raw_elements = []
    blocks = page.get_text("blocks", sort=True)
    last_bottom = 50 # 修复顶部图 bug
    text_buffer = ""
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        b_top = b_rect.y0
        
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
        
    # === 关键：这里调用多核翻译 ===
    return batch_translate_elements(raw_elements)

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (双模式) ---
def generate_html(doc, start, end, mode="pure", filename="Document", font_size=16, line_height=1.6, img_width=50):
    
    css = get_css(mode, font_size, line_height, img_width)
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        
        # 解析页面 (包含并发翻译)
        page_els = parse_page(page)
        
        page_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{page_class}">- {page_num} -</div>'
        
        if mode == "screenshot":
            # === 对照模式：左图右文 ===
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
            # === 纯净模式：上下结构 (V27原版) ===
            html_body += '<div class="pure-content">'
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
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}{MATHJAX_SCRIPT}</head><body>{html_body}</body></html>"

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
st.title("🔬 光学室学术论文翻译专用版 (V44)")

with st.sidebar:
    # 👇👇👇 保留了你的署名！ 👇👇👇
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #dcdcdc;">
        <h4 style="margin:0; color:#333;">👤 专属定制</h4>
        <p style="margin:5px 0 0 0; font-size:14px; color:#555;">
        <strong>制作人：</strong> 白水<br>
        <strong>微信：</strong> <code style="background:white;">guo21615</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not VALID_KEYS:
        st.warning("⚠️ 请在代码顶部 `USER_KEYS` 填入你的 API Key！")
    else:
        st.success(f"✅ 已加载 {len(VALID_KEYS)} 个 API Key，多核加速中！")
    
    uploaded_file = st.file_uploader("上传 PDF", type="pdf")
    st.markdown("---")
    
    # 模式选择
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    
    if app_mode == "🖨️ 导出 PDF":
        export_style = st.radio("导出风格：", ["纯净译文版 (竖向)", "中英对照版 (横向·左图右文)"], index=1)
        
        # 仅在对照模式下显示的排版选项
        if "对照" in export_style:
            with st.expander("🎨 对照版排版设置", expanded=True):
                ui_font_size = st.slider("字体大小", 10, 18, 14)
                ui_line_height = st.slider("行间距", 1.2, 2.0, 1.5)
                ui_img_width = st.slider("左图占比 (%)", 30, 70, 50)
        else:
            ui_font_size = 16
            ui_line_height = 1.6
            ui_img_width = 0

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if app_mode == "👁️ 实时预览":
        with st.sidebar:
            st.markdown("---")
            page_num = st.number_input("页码", 1, len(doc), 1)
            if st.button("🔄 翻译此页", type="primary"):
                st.session_state['run_preview'] = True
        
        # 预览默认使用对照模式，因为最直观
        if st.session_state.get('run_preview'):
             with st.spinner("🚀 多核加速渲染中..."):
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot", 
                                             font_size=14, line_height=1.5, img_width=50)
                components.html(preview_html, height=800, scrolling=True)
        else:
             st.info("👈 点击“翻译此页”")

    else:
        st.subheader("📄 批量导出")
        c1, c2 = st.columns(2)
        with c1: start = st.number_input("起始页", 1, len(doc), 1)
        with c2: end = st.number_input("结束页", 1, len(doc), min(3, len(doc)))
        
        # 确定模式字符串
        style_code = "screenshot" if "对照" in export_style else "pure"
        
        if st.button(f"🚀 生成 PDF", type="primary"):
            if not VALID_KEYS:
                 st.error("没有 Key 无法开工！")
            else:
                bar = st.progress(0)
                status = st.empty()
                
                status.text("正在多核并发翻译...")
                # 调用生成器
                full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name,
                                          font_size=ui_font_size, line_height=ui_line_height, img_width=ui_img_width)
                
                status.text("正在调用浏览器打印...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                    if ok:
                        bar.progress(100)
                        status.success("✅ 完成！")
                        fname = "Translation_V44.pdf"
                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button("📥 下载文件", f, fname)
                    else:
                        st.error(f"失败: {msg}")

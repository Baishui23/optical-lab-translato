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

# ==========================================
# 0. 配置部分 (API Key设置)
# ==========================================
st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# 优先从侧边栏获取，其次从 Secrets 获取，最后从代码默认值获取
with st.sidebar:
    st.markdown("### 🔑 API 设置")
    user_api_input = st.text_input("DeepSeek API Key (可选)", type="password", help="如果不填，将尝试使用配置文件或默认Key")

# 初始化 API_KEY
API_KEY = None

if user_api_input:
    API_KEY = user_api_input
else:
    try:
        API_KEY = st.secrets["DEEPSEEK_API_KEY"]
    except:
        # 👇 【如果本地运行且没有配置 secrets，请在这里填入你的 Key】
        API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 

BASE_URL = "https://api.deepseek.com"

# 初始化 Client
if API_KEY and API_KEY.startswith("sk-"):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
else:
    client = None
    # 仅在未提供有效 Key 时警告，不阻断 UI 渲染
    # st.sidebar.warning("⚠️ 未检测到有效 API Key，翻译功能将不可用")

# ==========================================
# 1. CSS 生成器 (动态排版)
# ==========================================
def get_css(font_size, line_height, img_width_pct):
    text_width_pct = 100 - img_width_pct - 2
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

        @page {{
            size: A4 landscape;
            margin: 15mm; 
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
        }}

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
            width: {img_width_pct}%;
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
            width: {text_width_pct}%;
            padding-left: 5px;
            text-align: justify;
            overflow-wrap: break-word;
        }}
        
        .MathJax {{ font-size: 100% !important; }}

        .pure-mode-container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .pure-mode-container p {{ margin-bottom: 1em; text-indent: 2em; }}
        .pure-mode-container img {{ max-width: 80%; display: block; margin: 20px auto; }}

        .caption {{ 
            font-size: {font_size - 2}px;
            color: #555; 
            text-align: center; 
            font-weight: bold; 
            margin-bottom: 15px; 
            font-family: sans-serif;
            margin-top: 5px;
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

# ==========================================
# 2. 核心逻辑 (图像识别与翻译)
# ==========================================

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def is_header_or_footer(rect, page_height):
    # 判定页眉页脚：页面最上方 50px 和最下方 50px
    return rect.y1 < 50 or rect.y0 > page_height - 50

# --- 🔥 关键修改：正则增强，支持 Fig, Figure ---
def is_caption_node(text):
    text = text.strip()
    # 匹配: Fig. 1, Figure 2, Fig 1, Figure. 1 等常见格式
    pattern = r'^(Fig|Figure)(\.|,|\s)\s?\d+'
    return re.match(pattern, text, re.IGNORECASE) is not None

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

def translate_text(text, is_caption=False):
    if not client: return text # 如果没有API key，直接返回原文
    if len(text.strip()) < 2: return text
    
    sys_prompt = """你是一个专业的物理学术翻译。
    【规则】
    1. 保持学术严谨性。
    2. 公式必须用 $...$ 或 $$...$$ 包裹。
    3. 直接输出译文，不要加前缀。
    """
    if is_caption: sys_prompt += " (这是图注，保留编号，例如 '图1: ...')"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def batch_translate_elements(elements):
    tasks = []
    indices = []
    for i, el in enumerate(elements):
        if el['type'] in ['text', 'caption']:
            tasks.append((el['content'], el['type'] == 'caption'))
            indices.append(i)
    
    if not tasks: return elements

    # 5线程并发翻译
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))
    
    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]
    return elements

# --- 🔥 关键修改：区域截图函数 ---
def capture_image_area(page, top, bottom):
    # 高度太小（小于15px）忽略，可能是误判
    if bottom - top < 15: return None
    
    # 定义截图区域：左右留白 40，上下就是传入的坐标
    rect = fitz.Rect(40, top, page.rect.width - 40, bottom)
    
    try:
        # matrix=3 保证清晰度
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        if pix.height < 10 or pix.width < 10: return None
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    except:
        return None

# --- 🔥 关键修改：锚点扫描逻辑 ---
def parse_page(page):
    raw_elements = []
    blocks = page.get_text("blocks", sort=True)
    
    # 初始高度避开页眉 (设为 60)
    last_bottom = 60 
    
    # 过滤掉页眉页脚
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        current_top = b_rect.y0
        current_bottom = b_rect.y1
        text_content = b[4]

        # === 核心逻辑：遇到图注，回头抓图 ===
        if is_caption_node(text_content):
            # 💡 发现图注！抓取 [上一段结尾] 到 [图注开头] 之间的区域
            img = capture_image_area(page, last_bottom, current_top)
            
            if img:
                raw_elements.append({'type': 'image', 'content': img})
            
            # 添加图注本身
            raw_elements.append({'type': 'caption', 'content': text_content})
            
            # 更新 last_bottom 为图注的底部
            last_bottom = current_bottom
            
        else:
            # === 兜底逻辑：防止无图注的巨型图片漏掉 ===
            # 如果当前文字和上一段文字中间空隙极大 (>250px)，可能中间有个没图注的图
            if current_top - last_bottom > 250:
                img = capture_image_area(page, last_bottom, current_top)
                if img: raw_elements.append({'type': 'image', 'content': img})

            # 添加普通文本
            if text_content.strip():
                raw_elements.append({'type': 'text', 'content': text_content})
            
            last_bottom = current_bottom # 更新底边位置

    return batch_translate_elements(raw_elements)

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

# ==========================================
# 3. HTML 构建器
# ==========================================
def generate_html(doc, start, end, mode="pure", filename="Document", font_size=14, line_height=1.6, img_width=50):
    
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
                    html_body += f'<div class="caption">{el["content"]}</div>'
                # 注意：对照模式下，右侧不重复显示通过 parse_page 抓取的小图，只显示整页截图
            
            html_body += "</div></div>"
            
        else:
            # 纯净模式
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

# ==========================================
# 4. PDF 引擎 (Chrome Headless)
# ==========================================
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
        # 🔥 修复：增加 stderr=subprocess.DEVNULL 屏蔽 DBus/OOM 报错
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "Success"
    except Exception as e:
        return False, str(e)

# ==========================================
# 5. 界面逻辑
# ==========================================
st.title("🔬 光学室学术论文翻译专用版 (V50 终极修复版)")

with st.sidebar:
    st.markdown("---")
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
    with st.expander("🎨 排版设置 (防溢出)", expanded=True):
        ui_font_size = st.slider("字体大小 (px)", 10, 18, 14)
        ui_line_height = st.slider("行间距", 1.2, 2.0, 1.6, 0.1)
        ui_img_width = st.slider("左图占比 (%)", 30, 70, 48)

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
            with st.spinner("🚀 智能识别图片与文本中..."):
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
            if not client:
                st.error("❌ 请先配置 API Key 才能导出！")
            else:
                bar = st.progress(0)
                status = st.empty()
                
                status.text("正在并发翻译...")
                full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name,
                                          font_size=ui_font_size,
                                          line_height=ui_line_height,
                                          img_width=ui_img_width)
                
                status.text("正在调用 Chrome 生成 PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                    if ok:
                        bar.progress(100)
                        status.success("✅ 完成！")
                        fname = "Translation_Result.pdf"
                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button("📥 下载 PDF", f, fname)
                    else:
                        st.error(f"生成失败: {msg}")

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

# --- 1. CSS 样式 (V31: 左侧强制左对齐，保留原始换行) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Times+New+Roman&display=swap');

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 15px; 
        line-height: 1.5;
        color: #111;
        margin: 0;
        padding: 0;
        background-color: white;
    }

    .page-container {
        max-width: 95%; 
        margin: 0 auto;
        padding: 30px;
        background-color: #fff;
    }

    /* === 双栏对照表格 === */
    .bilingual-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        table-layout: fixed; 
    }
    
    .bilingual-row {
        vertical-align: top;
        border-bottom: 1px dashed #e0e0e0; 
    }

    /* 左侧：原文列 (V31核心修改：保留原始排版) */
    .col-eng {
        width: 48%;
        padding: 10px 15px 10px 0;
        color: #333; 
        font-family: "Times New Roman", serif;
        /* 关键：左对齐，不要两端对齐，否则原始断行会很难看 */
        text-align: left; 
        font-size: 14px;
        line-height: 1.4; /* 稍微紧凑一点，还原PDF质感 */
        border-right: 2px solid #f0f0f0; 
        word-wrap: break-word;
        white-space: pre-wrap; /* 核心：保留所有换行符和空格！ */
    }
    
    /* 右侧：译文列 */
    .col-chn {
        width: 48%;
        padding: 10px 0 10px 15px;
        color: #000; 
        font-family: "Noto Serif SC", serif;
        text-align: justify;
        font-size: 15px;
        line-height: 1.6;
        word-wrap: break-word;
    }

    img { max-width: 90%; display: block; margin: 15px auto; }
    
    .caption { 
        font-size: 13px; color: #555; text-align: center; 
        font-weight: bold; margin-bottom: 25px; font-family: sans-serif;
        background: #f9f9f9; padding: 5px; border-radius: 4px;
    }

    .page-break { 
        page-break-before: always; border-top: 2px solid #eee; 
        margin-top: 20px; padding-top: 10px; text-align: center; 
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

# --- 2. 核心逻辑 (V31: 原文不动，只处理译文) ---

def clean_for_ai(text):
    """
    只为AI清洗文本，方便翻译。
    绝对不影响原文显示！
    """
    text = text.replace('-\n', '') # 拼接连字符
    text = text.replace('\n', ' ') # 拼接换行
    return re.sub(r'\s+', ' ', text).strip()

def translate_text(text, is_caption=False):
    # 先清洗一下给AI看，不然AI会被断行搞晕
    cleaned_text = clean_for_ai(text)
    if len(cleaned_text) < 2: return text
    
    sys_prompt = """你是一个物理学术翻译。
    【指令】
    1. 直接翻译给定的文本。
    2. 保持公式格式 $...$ 不变。
    3. 不要输出任何闲聊。
    """
    if is_caption: sys_prompt += " (这是图注)"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": cleaned_text}],
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
    # 获取原始文本块，不做任何 flag 处理，保证拿到最 raw 的数据
    blocks = page.get_text("blocks", sort=True) 
    last_bottom = 0
    
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    # 简单的逻辑：一个Block就是一个元素，不合并，不拆分，保持PDF原样
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        b_top = b_rect.y0
        if i == 0 and last_bottom == 0: last_bottom = b_top
        
        raw_text = b[4] # 这是 PDF 里最原始的字符串，包含 \n
        
        # 1. 尝试抓取图片
        img = capture_image_between_blocks(page, last_bottom, b_top)
        if img: elements.append({'type': 'image', 'content': img})

        # 2. 处理文本
        if is_caption_node(raw_text):
            # 图注
            trans = translate_text(raw_text, True)
            elements.append({'type': 'caption', 'original': raw_text, 'translation': trans})
        else:
            # 正文
            # 只有当文本不是纯页码数字时才翻译
            if len(clean_for_ai(raw_text)) > 5:
                trans = translate_text(raw_text, False)
                # 重点：这里存入的 original 是 raw_text (带换行符的)
                elements.append({'type': 'text_pair', 'original': raw_text, 'translation': trans})
            
        last_bottom = b_rect.y1
                
    return elements

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (V31: 左侧直接显示 Raw Text) ---
def generate_html(all_pages_data, mode="pure", filename="Doc"):
    html_body = f'<div class="page-container">'
    
    for idx, page_els in enumerate(all_pages_data):
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- {idx+1} -</div>'
        
        if mode == "bilingual":
            html_body += '<table class="bilingual-table">'
        
        for el in page_els:
            if el['type'] == 'image':
                if mode == "bilingual": html_body += '</table>'
                html_body += f'<img src="{image_to_base64(el["content"])}" />'
                if mode == "bilingual": html_body += '<table class="bilingual-table">'
            
            elif el['type'] == 'caption':
                if mode == "bilingual": html_body += '</table>'
                html_body += f"""
                <div class="caption">
                    <div>[原文] {el['original']}</div>
                    <div style="margin-top:4px; color:#000;">[译文] {el['translation']}</div>
                </div>
                """
                if mode == "bilingual": html_body += '<table class="bilingual-table">'
                
            elif el['type'] == 'text_pair':
                if mode == "bilingual":
                    # --- V31 核心：左侧不处理换行符 ---
                    # original 直接就是 PDF 里的样子，CSS 的 white-space: pre-wrap 会渲染出换行
                    op = el['original'] 
                    tp = clean_latex(el['translation'])
                    
                    html_body += f"""
                    <tr class="bilingual-row">
                        <td class="col-eng">{op}</td>
                        <td class="col-chn">{tp}</td>
                    </tr>
                    """
                else:
                    tp = clean_latex(el['translation'])
                    html_body += f'<div class="pure-text"><p>{tp}</p></div>'

        if mode == "bilingual":
            html_body += '</table>'

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
            ["纯净译文版 (仅中文)", "中英对照版 (Raw模式)"], 
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
                    preview_html = generate_html([els], mode="bilingual") 
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
                status.text(f"正在处理第 {p} 页 (Raw模式)...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在生成文档...")
            full_html = generate_html(data, mode=style_code, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    status.success("✅ 完成！")
                    fname = "Translation_Raw.pdf" if style_code == "bilingual" else "Translation_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载 Raw 对照版 PDF", f, fname)
                else:
                    st.error(f"失败: {msg}")

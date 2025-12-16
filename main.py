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

# --- 1. CSS 样式 (V30: 表格布局 + 原文美化) ---
COMMON_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&family=Times+New+Roman&display=swap');

    body {
        font-family: "Noto Serif SC", "SimSun", serif;
        font-size: 15px; 
        line-height: 1.6;
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

    /* === 双栏对照表格 (核心) === */
    .bilingual-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
        table-layout: fixed; /* 强制等宽，防止挤压 */
    }
    
    .bilingual-row {
        vertical-align: top;
        border-bottom: 1px dashed #e0e0e0; /* 每段之间加虚线，清晰 */
    }
    
    .bilingual-row:last-child {
        border-bottom: none;
    }

    /* 左侧：原文列 */
    .col-eng {
        width: 48%;
        padding: 12px 15px 12px 0;
        color: #333; 
        font-family: "Times New Roman", serif;
        text-align: justify; /* 两端对齐，解决“排版烂” */
        font-size: 15px;
        line-height: 1.5;
        border-right: 2px solid #f0f0f0; 
        word-wrap: break-word;
        hyphens: auto; /* 英文自动断词 */
    }
    
    /* 右侧：译文列 */
    .col-chn {
        width: 48%;
        padding: 12px 0 12px 15px;
        color: #000; 
        font-family: "Noto Serif SC", serif;
        text-align: justify;
        font-size: 15px;
        line-height: 1.6;
        word-wrap: break-word;
    }

    /* 纯净模式 */
    .pure-text p { margin-bottom: 1em; text-indent: 2em; text-align: justify; }

    /* 图片 */
    img { max-width: 90%; display: block; margin: 15px auto; }
    
    .caption { 
        font-size: 13px; color: #555; text-align: center; 
        font-weight: bold; margin-bottom: 25px; font-family: sans-serif;
        background: #f9f9f9; padding: 5px; border-radius: 4px;
    }

    /* 分页 */
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

# --- 2. 核心逻辑 (V30: 积木式对齐 + 强力清洗) ---

def clean_pdf_text(text):
    """
    V30关键函数：清洗PDF的烂排版
    1. 去除行尾连字符 (pro-\ngram -> program)
    2. 去除硬换行，变成流畅段落
    """
    # 替换连字符换行: "word-\nnext" -> "wordnext"
    text = text.replace('-\n', '')
    # 替换普通换行: "word\nnext" -> "word next"
    text = text.replace('\n', ' ')
    # 去除多余空格
    return re.sub(r'\s+', ' ', text).strip()

def translate_batch(text_list, is_caption=False):
    """
    批量翻译列表，保持一一对应
    """
    if not text_list: return []
    
    # 构造带分隔符的 Prompt，强迫模型保持结构
    separator = " ||| "
    combined_text = separator.join(text_list)
    
    sys_prompt = """你是一个物理学术翻译。
    【指令】
    1. 翻译给定的文本片段。
    2. 输入中有 ' ||| ' 分隔符，输出中必须保留该分隔符，严格一一对应。
    3. 保持公式格式 $...$ 不变。
    4. 不要合并段落，不要自由发挥。
    """
    if is_caption: sys_prompt += " (这是图注)"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": combined_text}],
            stream=False
        )
        result = response.choices[0].message.content
        # 按分隔符拆回列表
        trans_list = result.split("|||")
        
        # 兜底：如果拆分数量不对，强制补齐或截断
        if len(trans_list) != len(text_list):
            # 如果AI没听话，就回退到逐个翻译（稍微慢点但稳）
            return [translate_single(t) for t in text_list]
            
        return [t.strip() for t in trans_list]
    except:
        return text_list # 失败返回原文

def translate_single(text):
    # 备用单条翻译
    try:
        res = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "翻译为学术中文，保留LaTeX公式"}, {"role": "user", "content": text}],
            stream=False
        )
        return res.choices[0].message.content
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
    blocks = page.get_text("blocks", sort=True) # 按位置排序
    last_bottom = 0
    
    # 临时收集器
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    # 1. 预处理：将所有Block分类 (图片/图注/正文)
    text_buffer_list = [] # 待翻译的纯文本块
    
    for i, b in enumerate(valid_blocks):
        b_rect = fitz.Rect(b[:4])
        b_top = b_rect.y0
        if i == 0 and last_bottom == 0: last_bottom = b_top
        
        raw_text = b[4]
        
        # 检查是否是图注
        if is_caption_node(raw_text):
            # 先处理之前积攒的文本
            if text_buffer_list:
                # 批量翻译之前攒的积木
                cleaned_texts = [clean_pdf_text(t) for t in text_buffer_list]
                trans_texts = translate_batch(cleaned_texts)
                # 存入elements
                for src, trans in zip(cleaned_texts, trans_texts):
                    if src.strip():
                        elements.append({'type': 'text_pair', 'original': src, 'translation': trans})
                text_buffer_list = []

            # 抓取图注上方的图片
            img = capture_image_between_blocks(page, last_bottom, b_top)
            if img: elements.append({'type': 'image', 'content': img})
            
            # 处理图注本身
            clean_cap = clean_pdf_text(raw_text)
            trans_cap = translate_single(clean_cap)
            elements.append({'type': 'caption', 'original': clean_cap, 'translation': trans_cap})
            
        else:
            # 普通文本，先清洗，如果太短（可能是页码噪音）就丢弃
            cleaned = clean_pdf_text(raw_text)
            if len(cleaned) > 5: # 忽略太碎的噪点
                text_buffer_list.append(raw_text) # 暂存，稍后批量翻译
            
        last_bottom = b_rect.y1
        
    # 2. 处理页面剩余的文本
    if text_buffer_list:
        cleaned_texts = [clean_pdf_text(t) for t in text_buffer_list]
        trans_texts = translate_batch(cleaned_texts)
        for src, trans in zip(cleaned_texts, trans_texts):
            if src.strip():
                elements.append({'type': 'text_pair', 'original': src, 'translation': trans})
                
    return elements

def clean_latex(text):
    return text.replace(r'\[', '$$').replace(r'\]', '$$').replace(r'\(', '$').replace(r'\)', '$')

# --- 3. HTML 构建器 (V30: 严格表格行生成) ---
def generate_html(all_pages_data, mode="pure", filename="Doc"):
    html_body = f'<div class="page-container">'
    
    for idx, page_els in enumerate(all_pages_data):
        page_class = "page-break first-page" if idx == 0 else "page-break"
        html_body += f'<div class="{page_class}">- {idx+1} -</div>'
        
        # 如果是对照模式，开启大表格
        if mode == "bilingual":
            html_body += '<table class="bilingual-table">'
        
        for el in page_els:
            if el['type'] == 'image':
                # 图片暂时打断表格（如果表格已开启，先闭合，放图，再开）
                if mode == "bilingual": html_body += '</table>'
                html_body += f'<img src="{image_to_base64(el["content"])}" />'
                if mode == "bilingual": html_body += '<table class="bilingual-table">'
            
            elif el['type'] == 'caption':
                if mode == "bilingual": html_body += '</table>' # 打断表格
                html_body += f"""
                <div class="caption">
                    <div>[原文] {el['original']}</div>
                    <div style="margin-top:4px; color:#000;">[译文] {el['translation']}</div>
                </div>
                """
                if mode == "bilingual": html_body += '<table class="bilingual-table">'
                
            elif el['type'] == 'text_pair':
                if mode == "bilingual":
                    # --- V30: 完美的表格行 ---
                    op = el['original']
                    tp = clean_latex(el['translation'])
                    html_body += f"""
                    <tr class="bilingual-row">
                        <td class="col-eng">{op}</td>
                        <td class="col-chn">{tp}</td>
                    </tr>
                    """
                else:
                    # 纯净模式
                    tp = clean_latex(el['translation'])
                    html_body += f'<div class="pure-text"><p>{tp}</p></div>'

        if mode == "bilingual":
            html_body += '</table>' # 闭合本页表格

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
            ["纯净译文版 (仅中文)", "中英对照版 (严格对齐)"], 
            index=1 # 默认选中对照版
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
                with st.spinner("AI 正在积木式解析..."):
                    els = parse_page(doc[page_num-1])
                    preview_html = generate_html([els], mode="bilingual") # 预览也直接看对照效果
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
                status.text(f"正在处理第 {p} 页 (精准对齐中)...")
                data.append(parse_page(doc[p-1]))
                bar.progress((i+1) / (end-start+1))
            
            status.text("正在渲染文档...")
            full_html = generate_html(data, mode=style_code, filename=uploaded_file.name)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                if ok:
                    status.success("✅ 完成！")
                    fname = "Translation_Aligned.pdf" if style_code == "bilingual" else "Translation_Pure.pdf"
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("📥 下载完美对齐版 PDF", f, fname)
                else:
                    st.error(f"失败: {msg}")

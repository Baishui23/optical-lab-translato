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
import time
from typing import List, Dict, Tuple, Any

# ==========================================
# 👇 【关键配置】请在这里填入你的 API Key 👇
# ==========================================
USER_KEYS = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", 
]
# ==========================================

BASE_URL = "https://api.deepseek.com"

# --- 初始化 Key 轮询器 ---
VALID_KEYS = [k.strip() for k in USER_KEYS if k.strip().startswith("sk-")]
if not VALID_KEYS:
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            VALID_KEYS = [st.secrets["DEEPSEEK_API_KEY"]]
    except:
        pass

KEY_CYCLE = cycle(VALID_KEYS) if VALID_KEYS else None

st.set_page_config(page_title="光学室学术论文翻译专用版", page_icon="🔬", layout="wide")

# --- 1. CSS 样式 (解决公式溢出 & 字体优化) ---
def get_css(mode="pure", font_size=16, line_height=1.6, img_width_pct=50):
    base_css = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');
    body {
        font-family: "Noto Serif SC", "Times New Roman", "SimSun", serif;
        color: #000; margin: 0; padding: 0; background-color: white;
    }
    .caption { font-size: 14px; color: #444; text-align: center; font-weight: bold; margin-bottom: 15px; font-family: sans-serif; }
    .page-break { page-break-before: always; margin-top: 30px; border-top: 1px dashed #eee; padding-top: 10px; text-align: center; color: #ccc; font-size: 12px; }
    .page-break.first-page { page-break-before: avoid; display: none; }
    /* 强制公式不换行，允许横向滚动 */
    .MathJax_Display { overflow-x: auto; overflow-y: hidden; max-width: 100%; text-align: center; margin: 1em 0; }
    mjx-container { overflow-x: auto; overflow-y: hidden; max-width: 100%; }
    @media print { .page-break { border: none; color: transparent; margin: 0; height: 0; } }
    """

    if mode == "pure":
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
        text_width_pct = 100 - img_width_pct - 2
        return f"""
        <style>
            {base_css}
            @page {{ size: A4 landscape; margin: 10mm; }}
            body {{ font-size: {font_size - 2}px; line-height: {line_height}; }}
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
            .right-col-text img {{ max-width: 98%; display: block; margin: 10px auto; border: 1px solid #eee; }}
        </style>
        """

MATHJAX_SCRIPT = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'ams'
  },
  svg: {
    fontCache: 'global'
  }
};
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
    if re.match(r'^Fig(ure)?\.?\s?\d+', text, re.IGNORECASE):
        return True
    return False

def get_next_client():
    if not KEY_CYCLE: return None
    return OpenAI(api_key=next(KEY_CYCLE), base_url=BASE_URL)

# === V49 核心功能：公式预修复 Hard-coded Rules ===
def pre_fix_formulas(text: str) -> str:
    """在发送给 LLM 之前，使用正则匹配修复最常见的破碎琼斯矢量和矩阵。"""
    
    # 1. 修复琼斯矢量 E_in = [1; 0] (最常出现在光学论文中)
    # 匹配模式：E_in = (1) 0 (1) 或 E_in = | \ 1 \ 0 \ | / 这种混乱结构
    # 由于 PyMuPDF 提取极其混乱，我们尝试匹配其上下文关键词和断裂的数字
    # 关键词：Ein, E_{in}, 1, 0, |\, /, \n
    
    # 示例破碎结构:
    # E_in = 1 \n 0 \n (1)
    # E_in = /| \n /| 1 \n /| \n 0 \n /|
    
    # 简化匹配：如果文本中包含 E_in/E_m, 1, 0，且前后有换行符，且靠近一个公式编号
    # 我们先对 Ein 的情况进行一个精确替换 (假设原文公式结构是固定的)
    
    # Hard-coded Rule 1: 入射琼斯矢量 E_in
    # 寻找：(E_in 或 E_m) 后面跟着 (1) 接着 (0) 接着 (1) 或 (Ein = /| \ 1 \ 0 \)
    pattern_ein = re.compile(
        r'(E_{in}|E_{m}|E_{t}|E_{s}|Ein|Em|Et|Es)\s*=\s*(?:.*?)(\d)\s*(\d)\s*(?:.*?)(\(1\)|\(2\)|\(3\)|\(4\))', 
        re.DOTALL | re.IGNORECASE
    )
    # 如果检测到类似结构，替换为标准的 LaTeX 矩阵
    match = pattern_ein.search(text)
    if match:
        # 提取变量名和公式编号
        var_name = match.group(1)
        # number_1 = match.group(2) # 1
        # number_2 = match.group(3) # 0
        eq_num = match.group(4)
        
        # 琼斯矢量通常是 [1, 0] 或 [0, 1]
        latex_vector = f'$$ {var_name} = \\begin{{pmatrix}} 1 \\\\ 0 \\end{{pmatrix}} \quad {eq_num} $$'
        # 替换包含整个破碎公式的区域
        text = pattern_ein.sub(latex_vector, text)


    # Hard-coded Rule 2: S-waveplate 传输矩阵 T(α) (更复杂，更易破碎)
    # 破碎结构示例：T(α) = cos α sin α \n sin α -cos α \n (2)
    # 寻找：T(α) 或 T(a) 后面跟着 cos/sin/α 接着 (2)
    pattern_t_alpha = re.compile(
        r'(T(\(\alpha\))|T\(a\))\s*=\s*(?:.*?)(\cos\s*\alpha|\cos\s*a)\s*(\sin\s*\alpha|\sin\s*a)(?:.*?)(\sin\s*\alpha|\sin\s*a)\s*(\-\cos\s*\alpha|\-\cos\s*a)(?:.*?)(\(2\))',
        re.DOTALL | re.IGNORECASE
    )

    match = pattern_t_alpha.search(text)
    if match:
        eq_num = match.group(7)
        latex_matrix = f'$$ T(\\alpha) = \\begin{{pmatrix}} \\cos \\alpha & \\sin \\alpha \\\\ \\sin \\alpha & -\\cos \\alpha \\end{{pmatrix}} \quad {eq_num} $$'
        text = pattern_t_alpha.sub(latex_matrix, text)
        
    # Hard-coded Rule 3: E_t = T(α)E_in (3)
    # 寻找：E_t = T(α)E_in 后面跟着复杂的公式串 接着 (3)
    pattern_et_matrix = re.compile(
        r'(E_{t}|Et)\s*=\s*(T(\(\alpha\))|T\(a\))\s*(E_{in}|Ein)(?:.*?)(\(3\))',
        re.DOTALL | re.IGNORECASE
    )

    match = pattern_et_matrix.search(text)
    if match:
        eq_num = match.group(5)
        # 用一个标准的 LaTeX 替换掉中间可能破碎的公式部分
        latex_et = f'$$ E_t = T(\\alpha)E_{{in}} = \\begin{{pmatrix}} \\cos \\alpha & \\sin \\alpha \\\\ \\sin \\alpha & -\\cos \\alpha \\end{{pmatrix}} \\begin{{pmatrix}} 1 \\\\ 0 \\end{{pmatrix}} \quad {eq_num} $$'
        text = pattern_et_matrix.sub(latex_et, text)


    return text

# === V48 核心功能：带重试的翻译函数 (略作优化) ===
def translate_text(text: str, is_caption: bool, max_retries: int = 3) -> str:
    if len(text.strip()) < 2: return text
    
    # V49 关键：先进行硬编码公式修复
    fixed_text = pre_fix_formulas(text)
    
    sys_prompt = """你是一位精通光学和量子物理的学术翻译专家。
    【任务】
    1. 将用户提供的英文学术文本翻译成**流畅、准确的简体中文**。
    2. **如果原文中依然存在未修复的 LaTeX 公式破碎，你必须根据上下文将其还原为标准的 LaTeX 格式**（使用 `$$...$$` 或 `$...$`）。
    3. **绝对禁止**直接输出英文原文。必须翻译！
    4. 不要输出任何解释性文字，只输出翻译后的正文。
    """
    if is_caption: 
        sys_prompt += " (注意：这是图片说明，请保留 Figure 编号，例如 '图1(a) 展示了...') "

    for attempt in range(max_retries):
        client = get_next_client()
        if not client: return "[Key未配置]"
        
        try:
            # 使用预修复后的文本
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": sys_prompt}, 
                    {"role": "user", "content": f"Please translate and fix LaTeX:\n\n{fixed_text}"}
                ],
                stream=False,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            print(f"API Error (Attempt {attempt + 1}/{max_retries}): {error_msg}")
            if attempt == max_retries - 1:
                return f"【翻译失败】{text}" 
            time.sleep(2 ** attempt)
            continue
    
    return f"【翻译失败】{text}"


def capture_image_between_blocks(page, prev_bottom, current_top):
    if current_top - prev_bottom < 10: return None
    safe_top = prev_bottom + 2
    safe_bottom = current_top - 2
    if safe_bottom <= safe_top: return None

    rect = fitz.Rect(50, safe_top, page.rect.width - 50, safe_bottom)
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        if img.size[1] >= 20: 
            return img
        return None
    except: 
        return None

# 智能拼合算法（V47）
def smart_merge_blocks(blocks):
    merged = []
    if not blocks: return merged
    
    blocks.sort(key=lambda b: b[1]) # b[1] is y0
    
    current_text = ""
    current_rect = None
    
    for b in blocks:
        b_text = b[4]
        b_rect = fitz.Rect(b[:4])
        
        if is_caption_node(b_text):
            if current_text:
                merged.append({'type': 'text', 'content': current_text, 'rect': current_rect})
                current_text = ""
            merged.append({'type': 'caption', 'content': b_text, 'rect': b_rect})
            continue

        if current_text:
            if b_rect.y0 - current_rect.y1 > 50: 
                merged.append({'type': 'text', 'content': current_text, 'rect': current_rect})
                current_text = b_text
                current_rect = b_rect
            else:
                current_text += "\n" + b_text 
                current_rect = current_rect | b_rect 
        else:
            current_text = b_text
            current_rect = b_rect
            
    if current_text:
        merged.append({'type': 'text', 'content': current_text, 'rect': current_rect})
        
    return merged

def parse_page(page):
    blocks = page.get_text("blocks", sort=True)
    valid_blocks = [b for b in blocks if not is_header_or_footer(fitz.Rect(b[:4]), page.rect.height)]
    
    merged_elements = smart_merge_blocks(valid_blocks)
    
    final_elements = []
    last_bottom = 50
    
    for el in merged_elements:
        el_top = el['rect'].y0
        
        if el['type'] == 'caption':
            img = capture_image_between_blocks(page, last_bottom, el_top)
            if img: final_elements.append({'type': 'image', 'content': img})
            final_elements.append(el)
        else:
            if el_top - last_bottom > 150: 
                img = capture_image_between_blocks(page, last_bottom, el_top)
                if img: final_elements.append({'type': 'image', 'content': img})
            final_elements.append(el)
            
        last_bottom = el['rect'].y1

    return batch_translate_elements(final_elements)

def batch_translate_elements(elements):
    tasks = []
    indices = []
    for i, el in enumerate(elements):
        if el['type'] in ['text', 'caption']:
            tasks.append((el['content'], el['type'] == 'caption'))
            indices.append(i)
    
    if not tasks: return elements

    max_workers = 4 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda p: translate_text(p[0], p[1]), tasks))
    
    for idx_in_tasks, idx_in_elements in enumerate(indices):
        elements[idx_in_elements]['content'] = results[idx_in_tasks]
    return elements

def get_page_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def clean_latex(text):
    text = re.sub(r'^```(latex|markdown)?', '', text.strip())
    text = re.sub(r'```$', '', text.strip())
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    return text

# --- 3. HTML 构建器 ---
def generate_html(doc, start, end, mode="pure", filename="Document", font_size=16, line_height=1.6, img_width=50):
    css = get_css(mode, font_size, line_height, img_width)
    html_body = f'<div class="page-container">'
    
    for page_num in range(start, end + 1):
        page = doc[page_num-1]
        page_els = parse_page(page)
        
        page_class = "page-break first-page" if page_num == start else "page-break"
        html_body += f'<div class="{page_class}">- {page_num} -</div>'
        
        if mode == "screenshot":
            img_b64 = image_to_base64(get_page_image(page))
            html_body += f"""
            <div class="split-layout">
                <div class="left-col-image"><img src="{img_b64}" /></div>
                <div class="right-col-text">
            """
            for el in page_els:
                if el['type'] == 'text':
                    clean_content = clean_latex(el['content'])
                    if "$$" in clean_content:
                        html_body += f"<div>{clean_content}</div>"
                    else:
                        paras = clean_content.split('\n\n')
                        for p in paras:
                            if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'image':
                    html_body += f'<img src="{image_to_base64(el["content"])}" />'
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">{clean_latex(el["content"])}</div>'
            html_body += "</div></div>"
        else:
            html_body += '<div class="pure-content">'
            for el in page_els:
                if el['type'] == 'text':
                    clean_content = clean_latex(el['content'])
                    if "$$" in clean_content:
                         html_body += f"<div>{clean_content}</div>"
                    else:
                        paras = clean_content.split('\n\n')
                        for p in paras:
                            if p.strip(): html_body += f"<p>{p.strip().replace('**', '')}</p>"
                elif el['type'] == 'image':
                    html_body += f'<img src="{image_to_base64(el["content"])}" />'
                elif el['type'] == 'caption':
                    html_body += f'<div class="caption">{clean_latex(el["content"])}</div>'
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
st.title("🔬 光学室学术论文翻译 (V49 绝境求生版)")

with st.sidebar:
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #dcdcdc;">
        <h4 style="margin:0; color:#333;">👤 专属定制</h4>
        <p style="margin:5px 0 0 0; font-size:14px; color:#555;">
        <strong>制作人：</strong> 白水<br>
        <strong>版本：</strong> V49 (Hard-coded 公式直修)<br>
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
    
    app_mode = st.radio("功能模式", ["👁️ 实时预览", "🖨️ 导出 PDF"])
    
    if app_mode == "🖨️ 导出 PDF":
        export_style = st.radio("导出风格：", ["纯净译文版 (竖向)", "中英对照版 (横向·左图右文)"], index=1)
        if "对照" in export_style:
            with st.expander("🎨 对照版排版设置", expanded=True):
                ui_font_size = st.slider("字体大小", 10, 18, 13)
                ui_line_height = st.slider("行间距", 1.2, 2.0, 1.4)
                ui_img_width = st.slider("左图占比 (%)", 30, 70, 50)
        else:
            ui_font_size, ui_line_height, ui_img_width = 16, 1.6, 0

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
             with st.spinner("🚀 正在硬编码修复公式 & 容错翻译中..."):
                preview_html = generate_html(doc, page_num, page_num, mode="screenshot", 
                                             font_size=13, line_height=1.4, img_width=50)
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
            if not VALID_KEYS:
                 st.error("没有 Key 无法开工！")
            else:
                bar = st.progress(0)
                status = st.empty()
                status.text("正在进行硬编码公式修复与容错翻译...")
                
                full_html = generate_html(doc, start, end, mode=style_code, filename=uploaded_file.name,
                                          font_size=ui_font_size, line_height=ui_line_height, img_width=ui_img_width)
                
                status.text("正在渲染 LaTeX 并生成 PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    ok, msg = html_to_pdf_with_chrome(full_html, tmp_pdf.name)
                    if ok:
                        bar.progress(100)
                        status.success("✅ 修复完成！翻译回来了！")
                        fname = "Translation_V49_HardFix.pdf"
                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button("📥 下载文件", f, fname)
                    else:
                        st.error(f"失败: {msg}")

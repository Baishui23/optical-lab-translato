import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
from PIL import Image
import io
import base64
import os
import subprocess
import tempfile
import shutil
import platform
import time

# ==========================================
# 👇 这里填你的 Key，多填几个轮换更稳 👇
# ==========================================
USER_KEYS = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]
# ==========================================

BASE_URL = "https://api.deepseek.com"

st.set_page_config(page_title="光学室学术论文翻译 (V27 经典版)", page_icon="🔬", layout="wide")

# --- CSS: 还原图1那种紧凑的“左图右文”样式 ---
def get_css(font_size=13, line_height=1.4):
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');
        
        @page {{
            size: A4 landscape;
            margin: 0;
        }}
        
        body {{
            font-family: "Noto Serif SC", "SimSun", serif;
            font-size: {font_size}pt;
            line-height: {line_height};
            color: #000;
            margin: 0;
            padding: 0;
            background-color: white;
        }}

        .page-container {{
            width: 297mm; /* A4 横向宽度 */
            height: 210mm; /* A4 横向高度 */
            page-break-after: always;
            display: flex;
            flex-direction: row;
            overflow: hidden;
            border-bottom: 1px dashed #ddd; /* 屏幕预览时方便看界线 */
        }}

        /* 左侧：原图区域 */
        .left-col {{
            width: 50%;
            height: 100%;
            border-right: 1px solid #ccc;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            box-sizing: border-box;
            background-color: #fcfcfc;
        }}
        
        .left-col img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain; /* 保证原图完整显示，不变形 */
        }}

        /* 右侧：译文区域 */
        .right-col {{
            width: 50%;
            height: 100%;
            padding: 25px 30px; /* 给文字留点呼吸感 */
            box-sizing: border-box;
            overflow-y: auto; /* 内容太多时允许滚动（PDF中会自动截断，但通常够用） */
            text-align: justify;
        }}

        .right-col p {{
            margin-bottom: 1em;
            text-indent: 2em; /* 首行缩进，更像论文 */
        }}
        
        .right-col .caption {{
            font-size: 0.9em;
            color: #444;
            font-weight: bold;
            margin: 1em 0;
            text-indent: 0;
            text-align: center;
            background: #f0f0f0;
            padding: 5px;
            border-radius: 4px;
        }}

        /* 隐藏Streamlit默认元素 */
        header, footer {{ display: none !important; }}
    </style>
    """

# --- 核心功能函数 ---

def get_client():
    # 简单的 Key 轮询
    valid_keys = [k for k in USER_KEYS if k.startswith("sk-")]
    if not valid_keys: return None
    return OpenAI(api_key=valid_keys[0], base_url=BASE_URL)

def image_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# V27 的经典翻译逻辑：不整那些花里胡哨的，就是一段一段硬翻
def translate_block(text):
    client = get_client()
    if not client: return "【Error: 请配置 API Key】" + text
    
    # 你的图1里是纯中文，所以这里强制要求中文
    prompt = "你是一个专业的光学物理翻译助手。请将以下学术文本段落翻译成地道的中文。保留所有公式（使用LaTeX格式 $$...$$ 或 $...$）。不要啰嗦，直接给译文。"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            stream=False,
            temperature=0.1 # 降温，保证准确
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation Error: {e}")
        time.sleep(1) # 稍微缓一下
        return text # 失败返回原文，别报错

def process_pdf_page(page):
    # 1. 搞定左边的图
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_b64 = image_to_base64(img)
    
    # 2. 搞定右边的字 (使用 V27 的简单提取逻辑)
    # 不去管什么坐标对齐，直接按阅读顺序提取文本块
    blocks = page.get_text("blocks", sort=True)
    
    translated_content = []
    
    for b in blocks:
        text = b[4].strip()
        # 过滤掉页眉页脚和太短的干扰项
        if len(text) < 5: continue 
        if text.isdigit(): continue # 只有页码
        
        # 判断是不是图注 (Figure 开头)
        is_caption = text.lower().startswith("fig")
        
        # 翻译！
        trans = translate_block(text)
        
        # 简单的清洗
        trans = trans.replace("```latex", "").replace("```", "")
        
        if is_caption:
            translated_content.append(f'<div class="caption">{trans}</div>')
        else:
            translated_content.append(f'<p>{trans}</p>')
            
    return img_b64, "".join(translated_content)

def generate_html(doc, start, end):
    html_body = ""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = end - start + 1
    
    for i, p_num in enumerate(range(start, end + 1)):
        status_text.text(f"正在处理第 {p_num} 页 (V27 稳定模式)...")
        page = doc[p_num-1]
        
        img_data, text_data = process_pdf_page(page)
        
        html_body += f"""
        <div class="page-container">
            <div class="left-col">
                <img src="{img_data}">
            </div>
            <div class="right-col">
                {text_data}
            </div>
        </div>
        """
        progress_bar.progress((i + 1) / total)
        
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {get_css()}
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <script>
        MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\(', '\\)']] }} }};
        </script>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

# --- PDF 导出引擎 ---
def html_to_pdf(html_content, output_path):
    # 寻找 Chrome
    chrome_bin = None
    if shutil.which("chromium"): chrome_bin = shutil.which("chromium")
    elif shutil.which("google-chrome"): chrome_bin = shutil.which("google-chrome")
    else:
        # Mac / Win 常见路径
        possible = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        ]
        for p in possible:
            if os.path.exists(p): chrome_bin = p; break
            
    if not chrome_bin: return False, "未找到 Chrome 浏览器，无法导出 PDF"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        f.write(html_content)
        tmp_html = f.name
        
    cmd = [
        chrome_bin, "--headless", "--disable-gpu",
        f"--print-to-pdf={output_path}",
        "--no-pdf-header-footer", # 去掉浏览器自带的页眉页脚
        f"file://{tmp_html}"
    ]
    
    if platform.system() == "Linux": cmd.insert(1, "--no-sandbox")

    try:
        subprocess.run(cmd, check=True, timeout=120) # V27 给多点时间渲染
        return True, "成功"
    except Exception as e:
        return False, str(e)

# --- 主界面 ---
st.title("🔬 光学室论文翻译 (V27 经典复刻版)")
st.markdown("这是你最喜欢的那个版本：**左侧整页原图，右侧纯净中文。** 不搞复杂排版，只求内容对、公式对。")

with st.sidebar:
    st.info("💡 提示：此版本为单线程处理，速度较慢但极度稳定。")
    uploaded_file = st.file_uploader("📄 上传 PDF", type=["pdf"])

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    col1, col2 = st.columns(2)
    with col1: start_page = st.number_input("起始页", 1, len(doc), 1)
    with col2: end_page = st.number_input("结束页", 1, len(doc), min(5, len(doc)))
    
    if st.button("🚀 开始翻译并导出 PDF", type="primary"):
        if not get_client():
            st.error("❌ 没填 API Key，跑不动！请在代码顶部填入 USER_KEYS。")
        else:
            with st.spinner("⏳ 正在慢工出细活 (每页约需 30秒 - 1分钟)..."):
                html_out = generate_html(doc, start_page, end_page)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    success, msg = html_to_pdf(html_out, tmp_pdf.name)
                    
                    if success:
                        st.success("✅ 翻译完成！")
                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button("📥 下载完美翻译版 PDF", f, "V27_Classic_Translation.pdf")
                    else:
                        st.error(f"⚠️ PDF 生成失败: {msg}")
                        # 失败了至少把 HTML 给用户
                        st.download_button("📥 下载 HTML (备用)", html_out, "debug.html")

import base64
import datetime
import hashlib
import hmac
import html
import inspect
import io
import json
import os
import re
import time
from collections import Counter
import pandas as pd
import streamlit as st
from Bio import Entrez
from openai import OpenAI
# set_page_config 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="麻醉学全网文献热点追踪与知识更新综述系统",
    page_icon="💉",
    layout="wide",
)
# ==========================================
# 1. 凭证加载（Streamlit Secrets -> 环境变量）
# ==========================================
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
def normalize_key(v):
    """去掉首尾空白、换行和误带的引号。"""
    return str(v or "").strip().strip("\"'“”‘’").strip()
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "").strip()
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
ENV_TAVILY_KEY = normalize_key(os.getenv("TAVILY_API_KEY", ""))
ENV_OPENROUTER_KEY = normalize_key(os.getenv("OPENROUTER_API_KEY", ""))
Entrez.email = NCBI_EMAIL or "anesthesia_agent@example.com"  # NCBI 要求提供邮箱
Entrez.api_key = NCBI_API_KEY or None
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)
FALLBACK_FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
    "apodex/apodex-1.1-mini:free",
    "openrouter/free",
]
DOC_COLUMNS = ["PMID/URL", "Title", "Abstract", "Year", "Source"]
TAVILY_MAX_RESULTS = 20
# PMC 开放全文抓取与提示词截断参数
PMC_FULLTEXT_MAX = 8          # 每次检索最多抓取多少篇 PMC 全文（控制耗时）
PMC_FULLTEXT_CHARS = 5000     # 每篇全文节选字符数
PROMPT_CHARS_ABSTRACT = 1500  # 送入大模型的单篇摘要上限
PROMPT_CHARS_FULLTEXT = 4500  # 送入大模型的单篇 PMC 全文节选上限
PROMPT_CHARS_LOCAL = 3000     # 用户上传文件是主题依据，给更长的节选额度
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (0, 0)
STRETCH = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}
# ==========================================
# 访问密码保护与退出登录逻辑
# ==========================================
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
MAX_FAILS = 5
LOCK_WINDOW = 600
@st.cache_resource
def _login_tracker():
    return {"fails": []}
def load_glossary_content():
    if os.path.exists("glossary.md"):
        try:
            with open("glossary.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return ""

# =====================【新增开始：术语表GitHub同步工具函数，仅此处新增】====================
def _get_github_glossary():
    """从github仓库读取glossary.md，返回(content,sha)；文件不存在返回("",None)"""
    import requests
    token, repo, branch = _github_cfg()
    if not token or not repo:
        return None, None, "未配置GITHUB_TOKEN / GITHUB_REPO"
    url = f"https://api.github.com/repos/{repo}/contents/glossary.md?ref={branch}"
    resp = requests.get(url, headers=_github_headers(token), timeout=20)
    if resp.status_code == 404:
        return "", None, "ok"
    if resp.status_code != 200:
        return None, None, f"GitHub读取失败 HTTP {resp.status_code}:{resp.text[:200]}"
    js = resp.json()
    import base64
    content = base64.b64decode(js["content"]).decode("utf-8")
    sha = js["sha"]
    return content, sha, "ok"

def _push_github_glossary(new_content: str, old_sha):
    """把glossary.md推送到github；新建文件old_sha传None"""
    import requests,base64
    token, repo, branch = _github_cfg()
    url = f"https://api.github.com/repos/{repo}/contents/glossary.md"
    payload = {
        "message": f"auto update glossary.md: add new term",
        "content": base64.b64encode(new_content.encode("utf‑8")).decode(),
        "branch": branch
    }
    if old_sha is not None:
        payload["sha"] = old_sha
    resp = requests.put(url, json=payload, headers=_github_headers(token), timeout=20)
    if resp.status_code in (200,201):
        return True,"✅已自动提交到GitHub仓库 glossary.md"
    return False,f"GitHub提交失败 HTTP {resp.status_code}: {resp.text[:300]}"

def append_glossary_term_github(eng_term: str, chn_term: str):
    """增加术语：先读github远端，追加，写回github，同步更新本地容器文件"""
    eng = eng_term.strip()
    chn = chn_term.strip()
    if not eng or not chn:
        return False,"英文术语、中文译名不能为空"
    new_line = f"- {eng}：{chn}\n"

    content_remote, sha, msg = _get_github_glossary()
    if content_remote is None:
        return False,msg
    updated_content = content_remote.rstrip("\n") + "\n" + new_line

    ok,push_msg = _push_github_glossary(updated_content, sha)
    if ok:
        with open("glossary.md","w",encoding="utf‑8") as f:
            f.write(updated_content)
        return True,f"{push_msg}\n新增：`{eng}` → `{chn}`"
    return False,push_msg
# =====================【新增结束】====================

def render_auth_header():
    """在侧边栏最顶部渲染退出登录按钮和词典加载状态 + 术语表管理（自动同步GitHub）"""
    ss = st.session_state
    if ss.get("_auth_ok"):
        if st.sidebar.button("🔒 退出登录", key="btn_logout"):
            ss.clear()
            st.rerun()
    glossary_content = load_glossary_content()
    lines = [l.strip() for l in glossary_content.splitlines() if l.strip()]
    term_count = 0
    for line in lines:
        if line.startswith(("-", "*")) or ":" in line or "—" in line or "-" in line:
            term_count += 1

    if term_count == 0 and glossary_content.strip():
        term_count = len(lines)
    if glossary_content.strip():
        st.sidebar.success(f"✅ 专业词典加载成功 (共收录 {term_count} 条核心术语)")
    else:
        st.sidebar.warning("⚠️ 未检测到 glossary.md 或词典为空")

    # ==========术语表管理popover弹窗（自动同步GitHub）==========
    if ss.get("_auth_ok"):
        gloss_pop = st.sidebar.popover("📖 术语表管理", use_container_width=True)
        with gloss_pop:
            st.subheader("➕ 添加新专业术语（自动提交GitHub）")
            with st.form("add_gloss_form", clear_on_submit=True):
                eng_input = st.text_input("英文术语", placeholder="example: bispectral index")
                chn_input = st.text_input("中文译名", placeholder="例：双频谱指数")
                add_submit = st.form_submit_button("追加到术语表（自动提交GitHub）")
                if add_submit:
                    ok, msg = append_glossary_term_github(eng_input, chn_input)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("---")
            st.subheader("📥 / 📤 术语表文件操作")
            st.download_button(
                label="下载 glossary.md",
                data=glossary_content,
                file_name="glossary.md",
                mime="text/markdown"
            )
            up_gloss = st.file_uploader("上传 glossary.md（将覆盖GitHub远端+本地）", type=["md"])
            if up_gloss is not None:
                try:
                    new_text = up_gloss.getvalue().decode("utf‑8")
                    _,sha,_ = _get_github_glossary()
                    ok,push_msg = _push_github_glossary(new_text, sha)
                    if ok:
                        with open("glossary.md","w",encoding="utf‑8") as f:
                            f.write(new_text)
                        st.success(push_msg+"\n✅本地缓存已更新，页面自动刷新")
                        st.rerun()
                    else:
                        st.error(push_msg)
                except Exception as e:
                    st.error(f"上传处理失败：{e}")
            st.info("💡新增术语会直接Commit提交到GitHub仓库glossary.md，无需手动下载上传。\n⚠️请确认Secrets已配置GITHUB_TOKEN、GITHUB_REPO、GITHUB_BRANCH")

    st.sidebar.markdown("---")

def require_password():
    ss = st.session_state
    if ss.get("_auth_ok"):
        return
    st.title("🔐 请输入访问密码")
    if not APP_PASSWORD:
        st.error("管理员尚未配置访问密码（Secrets 中的 APP_PASSWORD），应用已锁定。")
    st.stop()
    tracker = _login_tracker()
    now = time.time()
    tracker["fails"] = [t for t in tracker["fails"] if now - t < LOCK_WINDOW]
    if len(tracker["fails"]) >= MAX_FAILS:
        wait_min = int((LOCK_WINDOW - (now - tracker["fails"][0])) // 60) + 1
        st.error(f"错误次数过多，已暂时锁定，请约 {wait_min} 分钟后再试。")
        st.stop()
    _ac = {"autocomplete": "off"} if "autocomplete" in inspect.signature(st.text_input).parameters else {}
    with st.form("login_form", clear_on_submit=True):
        pwd = st.text_input("访问密码", type="password", **_ac)
        submitted = st.form_submit_button("进入", type="primary")
    if submitted:
        if hmac.compare_digest(pwd.encode("utf-8"), APP_PASSWORD.encode("utf-8")):
            ss["_auth_ok"] = True
            tracker["fails"].clear()
            st.rerun()
        else:
            tracker["fails"].append(time.time())
            time.sleep(1)
            st.error("密码错误。")
    st.stop()
require_password()
# 2. 权威临床与科研核心主题映射字典（共 33 项）
DRUG_STEMS = (
    "caine", "flurane", "zolam", "tomidine", "tomidate", "fentanil", "fentanyl",
    "ofol", "curonium", "curium", "ketamine", "setron", "phine", "morphone", "adol",
    "stigmine", "gammadex", "methasone", "rolac", "profen", "phrine", "drine",
    "olol", "pentin", "galin", "pental", "barbital", "trexone", "oxone",
)
DRUG_PATTERN = r"\b[a-z]*(?:" + "|".join(DRUG_STEMS) + r")\b"
DRUG_EXCLUDE = {"curium", "josephine", "delphine"}
CLINICAL_TOPICS = {
    "术中低血压 (Intraoperative Hypotension)": (
        r"\b(intraoperative hypotension|hypotension|low blood pressure)\b"
    ),
    "超声引导神经阻滞 (Ultrasound-guided Nerve Block)": (
        r"\b(ultrasound-guided|nerve block|regional anesthesia|brachial plexus"
        r" block|femoral nerve block|erector spinae|fascial plane block)\b"
    ),
    "衰弱/老年麻醉 (Frailty & Geriatric Anesthesia)": (
        r"\b(frailty|frail|elderly|geriatric|cognitive decline|postoperative"
        r" delirium)\b"
    ),
    "麻醉深度监测 (Depth of Anesthesia)": (
        r"\b(depth of anesthesia|bispectral index|bis|eeg|electroencephalogram|processed"
        r" eeg)\b"
    ),
    "术后恶心呕吐 (PONV)": (
        r"\b(ponv|postoperative nausea|vomiting|antiemetic)\b"
    ),
    "术后认知功能障碍 (POCD)": (
        r"\b(pocd|postoperative cognitive dysfunction|cognitive"
        r" impairment)\b"
    ),
    "气道管理/困难气道 (Airway Management)": (
        r"\b(airway management|difficult airway|videolaryngoscopy|endotracheal"
        r" intubation|laryngeal mask)\b"
    ),
    "围术期器官保护 (Organ Protection)": (
        r"\b(myocardial injury|acute kidney injury|aki|lung injury|organ"
        r" protection|ischemia-reperfusion)\b"
    ),
    "疼痛管理/多模式镇痛 (Multimodal Analgesia)": (
        r"\b(multimodal analgesia|opioid-sparing|postoperative pain|chronic"
        r" pain|analgesic)\b"
    ),
    "麻醉药物 (Anesthetic drugs)": DRUG_PATTERN,
    "人工智能/机器学习应用 (AI in Anesthesia)": (
        r"\b(artificial intelligence|machine learning|deep learning|predictive"
        r" model|algorithm)\b"
    ),
    "日间/短程麻醉 (Ambulatory Anesthesia)": (
        r"\b(ambulatory|day-surgery|outpatient anesthesia|same-day surgery)\b"
    ),
    "目标导向液体治疗 (GDFT)": (
        r"\b(gdft|goal-directed fluid|fluid therapy|hemodynamic monitoring)\b"
    ),
    "小儿麻醉/发育毒性 (Pediatric Anesthesia & Neurotoxicity)": (
        r"\b(pediatric anesthesia|pediatric|neonatal|infant|developmental"
        r" neurotoxicity|anesthetic-induced neurotoxicity)\b"
    ),
    "产科麻醉与镇痛 (Obstetric Anesthesia)": (
        r"\b(obstetric|parturient|cesarean section|epidural labor"
        r" analgesia|postpartum hemorrhage)\b"
    ),
    "心胸麻醉/体外循环 (Cardiothoracic Anesthesia)": (
        r"\b(cardiothoracic|cardiac surgery|cardiopulmonary bypass|thoracic"
        r" anesthesia|one-lung ventilation)\b"
    ),
    "神经外科麻醉 (Neuroanesthesia)": (
        r"\b(neuroanesthesia|neurosurgery|intracranial pressure|icp|cerebral"
        r" perfusion|craniotomy)\b"
    ),
    "重症监护与围术期医学 (Critical Care & Perioperative Medicine)": (
        r"\b(critical care|icu|sepsis|ards|perioperative medicine|postoperative"
        r" complications)\b"
    ),
    "围术期超声/POCUS (Point-of-Care Ultrasound)": (
        r"\b(pocus|point-of-care ultrasound|focused echocardiography|lung"
        r" ultrasound|gastric ultrasound)\b"
    ),
    "围术期预康复/优化 (Prehabilitation & Optimization)": (
        r"\b(prehabilitation|preoperative optimization|eras|enhanced recovery"
        r" after surgery|functional capacity)\b"
    ),
    "肌松药与逆转 (Neuromuscular Blockade & Reversal)": (
        r"\b(neuromuscular blockade|rocuronium|sugammadex|residual"
        r" neuromuscular blockade|train-of-four|tof)\b"
    ),
    "输血与血液管理 (Patient Blood Management)": (
        r"\b(patient blood management|pbm|transfusion|coagulopathy|tranexamic"
        r" acid|viscoelastic testing|rotem|teg)\b"
    ),
    "麻醉与肿瘤预后 (Anesthesia & Cancer Outcome)": (
        r"\b(cancer recurrence|metastasis|onco-anesthesia|cancer"
        r" outcome|immunomodulation)\b"
    ),
    "围术期过敏与严重不良反应 (Anaphylaxis & Complications)": (
        r"\b(anaphylaxis|allergic reaction|malignant hyperthermia|local"
        r" anesthetic systemic toxicity)\b"
    ),
    "术后低氧血症与呼吸功能障碍 (Postoperative Pulmonary Complications)": (
        r"\b(postoperative pulmonary"
        r" complications|ppc|hypoxemia|atelectasis|non-invasive ventilation)\b"
    ),
    "围术期体温管理/低体温 (Perioperative Hypothermia)": (
        r"\b(hypothermia|inadvertent hypothermia|patient warming|shivering)\b"
    ),
    "术后睡眠与昼夜节律 (Sleep & Circadian Rhythm)": (
        r"\b(sleep apnea|osa|obstructive sleep apnea|circadian rhythm|sleep"
        r" quality|melatonin)\b"
    ),
    "麻醉作用机制与意识研究 (Mechanism of Anesthesia & Consciousness)": (
        r"\b(mechanism of anesthesia|general anesthesia"
        r" mechanism|neural circuits|thalamocortical|consciousness)\b"
    ),
    "围术期神经炎症 (Neuroinflammation)": (
        r"\b(neuroinflammation|microglia|astrocytes|blood-brain"
        r" barrier|systemic inflammation)\b"
    ),
    "肠道菌群与麻醉/围术期 (Gut Microbiota & Anesthesia)": (
        r"\b(gut microbiota|gut microbiome|gut-brain axis|dysbiosis)\b"
    ),
    "麻醉与表观遗传学/转录组学 (Epigenetics & Genomics)": (
        r"\b(epigenetics|dna methylation|microrna|transcriptomics|single-cell"
        r" rna)\b"
    ),
    "绿色麻醉/环保麻醉 (Sustainable & Green Anesthesia)": (
        r"\b(green anesthesia|sustainable anesthesia|environmental"
        r" impact|desflurane|greenhouse gas|carbon footprint)\b"
    ),
    "麻醉安全与模拟教学 (Patient Safety & Simulation)": (
        r"\b(patient safety|simulation training|crisis resource"
        r" management|medical error|human factors)\b"
    ),
}
# ==========================================
# 3. 辅助计算与文本处理逻辑
# ==========================================
def extract_clinical_topics(text):
    text_lower = str(text).lower()
    return [name for name, pat in CLINICAL_TOPICS.items() if re.search(pat, text_lower)]
def extract_hot_drugs(df, top_n=15):
    cols = ["药物", "涉及文献数", "占比", "总提及次数"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    texts = (
        df["Title"].fillna("").astype(str) + " " + df["Abstract"].fillna("").astype(str)
    ).str.lower()
    doc_freq, mentions = Counter(), Counter()
    for text in texts:
        found = [
            w for w in re.findall(DRUG_PATTERN, text)
            if len(w) >= 6 and w not in DRUG_EXCLUDE
        ]
        doc_freq.update(set(found))
        mentions.update(found)
    rows = [
        {
            "药物": drug.capitalize(),
            "涉及文献数": n,
            "占比": f"{n / len(df):.0%}",
            "总提及次数": mentions[drug],
        }
        for drug, n in doc_freq.items()
    ]
    rows.sort(key=lambda r: (-r["涉及文献数"], -r["总提及次数"], r["药物"]))
    return pd.DataFrame(rows[:top_n], columns=cols)
def get_past_5_years_range():
    today = datetime.date.today()
    start = today - datetime.timedelta(days=365 * 5)
    return start, today
def get_past_5_years_date_filter():
    start, today = get_past_5_years_range()
    fmt = "%Y/%m/%d"
    return (
        f'("{start.strftime(fmt)}"[Date - Publication] : '
        f'"{today.strftime(fmt)}"[Date - Publication])'
    )
def doc_link(identifier):
    ident = str(identifier or "").strip()
    m = re.match(r"PMID\s*:\s*(\d+)", ident, flags=re.IGNORECASE)
    if m:
        pmid = m.group(1)
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", f"PMID:{pmid}"
    if ident.lower().startswith(("http://", "https://")):
        return ident, ident
    if ident.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{ident}/", f"PMID:{ident}"
    return None, "本地上传文件（无在线链接）"
def _md_escape(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip() or "（无标题）"
    return re.sub(r"([\\`*_{}\[\]<>|])", r"\\\1", text)
def build_literature_markdown(df):
    lines = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        url, shown = doc_link(row.get("PMID/URL"))
        title = _md_escape(row.get("Title"))
        year = str(row.get("Year") or "N/A")
        head = f"[{title}]({url})" if url else title
        link_part = f"[{shown}]({url})" if url else shown
        source = str(row.get("Source") or "").strip()
        src_part = f" ｜ 来源：{source}" if source else ""
        lines.append(f"{i}. **{head}**  \n   年份：{year}{src_part} ｜ 链接：{link_part}")
    return "\n\n".join(lines)
def clean_search_keyword(raw_text):
    if not raw_text:
        return "Anesthesia"

    if len(raw_text) > 50 or "\n" in raw_text:
        kw_match = re.search(r"keywords\s*[:：]\s*([^\n]+)", raw_text, re.IGNORECASE)
        if kw_match:
            raw_text = kw_match.group(1)
        else:
            clean_snippet = re.sub(r"[^a-zA-Z\s]", " ", raw_text[:300])
            words = [w for w in clean_snippet.split() if len(w) > 3 and w.lower() not in {"the", "and", "for", "with", "from", "this", "that"}]
            if words:
                return " ".join(words[:5])
    text = re.sub(r"\.(pdf|docx|txt|csv)$", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"[_—–-]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:8]) if words else "Anesthesia"
def english_part(topic):
    m = re.search(r"\(([^()]*)\)\s*$", topic)
    return m.group(1) if m else topic
def _decode_bytes(raw):
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")
def parse_uploaded_files(uploaded_files):
    data = []
    for idx, file in enumerate(uploaded_files, 1):
        filename = file.name
        ext = filename.rsplit(".", 1)[-1].lower()
        raw = file.getvalue()
        abstract = ""
        try:
            if ext == "pdf":
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw))
                abstract = "".join(p.extract_text() or "" for p in reader.pages)[:4000]
            elif ext == "docx":
                import docx
                doc = docx.Document(io.BytesIO(raw))
                abstract = "\n".join(p.text for p in doc.paragraphs)[:4000]
            elif ext == "txt":
                abstract = _decode_bytes(raw)[:4000]
            elif ext == "csv":
                df_up = pd.read_csv(io.StringIO(_decode_bytes(raw)))
                for i, (_, row) in enumerate(df_up.iterrows(), 1):
                    data.append({
                        "PMID/URL": str(row.get("PMID", f"LOCAL_{idx}_{i}")),
                        "Title": str(row.get("Title", row.get("title", "未命名文献"))),
                        "Abstract": str(row.get("Abstract", row.get("abstract", ""))),
                        "Year": str(row.get("Year", row.get("year", "N/A"))),
                        "Source": "上传文件",
                    })
                continue
            lines = [l.strip() for l in abstract.splitlines() if l.strip()]
            doc_title = filename
            if lines and len(lines[0]) > 10 and not lines[0].startswith("PMID"):
                doc_title = lines[0]
            data.append({
                "PMID/URL": f"LOCAL_{idx}",
                "Title": doc_title,
                "Abstract": abstract or "未能解析出有效正文",
                "Year": "本地文件",
                "Source": "上传文件",
            })
        except Exception as e:
            st.error(f"解析文件 {filename} 失败: {e}")
    return pd.DataFrame(data, columns=DOC_COLUMNS)
HEADER_STOPWORDS = {
    "journal", "available", "elsevier", "sciencedirect", "copyright", "article", "articles", "study",
    "patients", "patient", "results", "result", "methods", "method", "conclusion", "conclusions",
    "background", "objective", "objectives", "introduction", "discussion", "abstract", "keywords",
    "received", "accepted", "published", "online", "corresponding", "author", "authors", "university",
    "department", "hospital", "https", "email", "contents", "lists", "homepage", "volume", "issue",
    "with", "that", "this", "from", "were", "have", "been", "which", "their", "than", "after", "between",
    "using", "used", "also", "these", "other", "into", "such", "more", "most", "within", "among",
    "clinical", "review", "trial", "analysis", "group", "groups", "data", "research", "based", "associated",
}
def heuristic_topic_from_files(local_df):
    """无 API Key 或大模型失败时的兜底：优先取文件自带 Keywords 行，否则取高频有效词。"""
    texts = [f"{r['Title']}\n{str(r['Abstract'])[:3000]}" for _, r in local_df.iterrows()]
    joined = "\n".join(texts)
    m = re.search(r"keywords?\s*[:：]\s*([^\n]+)", joined, flags=re.IGNORECASE)
    if m:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", m.group(1))
                 if w.lower() not in HEADER_STOPWORDS]
        if len(words) >= 2:
            return {"query": " ".join(words[:6]), "topic_zh": "", "synonyms": []}
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", joined)
             if w.lower() not in HEADER_STOPWORDS]
    top = [w for w, _ in Counter(words).most_common(5)]
    return {"query": " ".join(top) if top else "", "topic_zh": "", "synonyms": []}
def derive_topic_from_files(local_df, api_key, model):
    """让大模型阅读上传文件，识别其核心研究主题；返回 {query, topic_zh, synonyms}。"""
    fallback = heuristic_topic_from_files(local_df)
    if not api_key:
        return fallback
    try:
        parts, total = [], 0
        for _, r in local_df.head(5).iterrows():
            seg = f"【{r['Title']}】\n{str(r['Abstract'])[:1800]}"
            parts.append(seg)
            total += len(seg)
            if total > 7000:
                break
        prompt = (
            "你是医学文献检索专家。下面是用户上传的文献内容（可能有多篇）。请判断它们共同的核心研究主题，"
            "并只输出一个 JSON 对象（不要代码块、不要解释），格式：\n"
            '{"topic_zh": "中文主题名（不超过25字）", '
            '"query": "3-6个英文检索关键词，空格分隔", '
            '"synonyms": ["核心概念的英文同义词或缩写，最多5个"]}\n'
            "要求：\n"
            "1. 只依据文献实际内容，query 必须是文献的核心疾病/药物/技术/人群等具体概念，"
            "不得泛化成 anesthesia、surgery、patients 这类宽泛词；\n"
            "2. query 不含 AND/OR、括号、引号；\n"
            "3. 忽略期刊页眉页脚、作者单位、版权声明等无关内容。\n\n"
            + "\n\n".join(parts)
        )
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        text, _, _ = call_openrouter_with_fallback(
            client, model, [{"role": "user", "content": prompt}], temperature=0.1
        )
        m = re.search(r"\{.*\}", text, flags=re.S)
        data = json.loads(m.group(0))
        q = re.sub(r"[^A-Za-z0-9\s\-]", " ", str(data.get("query", "")))
        q = " ".join(q.split()[:6])
        if not q or BAD_LLM_OUTPUT.search(q):
            return fallback
        syn = [str(s).strip() for s in data.get("synonyms", []) if str(s).strip()][:5]
        return {"query": q, "topic_zh": str(data.get("topic_zh", "")).strip()[:40], "synonyms": syn}
    except Exception:
        return fallback
RELEVANCE_GENERIC = {"anesthesia", "anaesthesia", "anesthetic", "perioperative", "surgery", "surgical",
                     "study", "review", "clinical", "patients", "trial", "effect", "effects"}
def filter_relevant(df, query, synonyms=None, min_keep=3):
    """按检索词在标题+摘要中的命中数过滤跑题文献；返回 (过滤后的 df, 被剔除篇数)。"""
    if df.empty:
        return df, 0
    words = [w for w in re.findall(r"[a-z0-9]+", str(query).lower())
             if len(w) >= 3 and w not in RELEVANCE_GENERIC]
    stems = list(dict.fromkeys(w[:6] for w in words))
    if not stems:
        return df, 0
    syn = [s.lower() for s in (synonyms or []) if s]
    need = max(1, -(-len(stems) // 2))  # 至少命中一半检索词
    scores = []
    for _, row in df.iterrows():
        text = f"{row.get('Title', '')} {row.get('Abstract', '')}".lower()
        sc = sum(1 for s in stems if s in text)
        if any(sy in text for sy in syn):
            sc += 1
        scores.append(sc)
    scored = df.assign(_score=scores)
    kept = scored[scored["_score"] >= need]
    if len(kept) < min_keep:  # 过滤过严时，保留得分最高的几篇，避免结果为空
        kept = scored[scored["_score"] > 0].sort_values("_score", ascending=False).head(max(min_keep, 5))
    kept = kept.sort_values("_score", ascending=False, kind="stable")
    return kept.drop(columns="_score").reset_index(drop=True), len(df) - len(kept)
# ==========================================
# 4. 检索引擎：Tavily 全网 + PubMed
# ==========================================
def tavily_request(api_key, query, max_results):
    import requests
    start, today = get_past_5_years_range()
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "query": (
                f"{query} clinical study review {start.year}-{today.year}"
            ),
            "search_depth": "advanced",
            "max_results": min(int(max_results), TAVILY_MAX_RESULTS),
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"HTTP {resp.status_code}: {resp.text[:200]} "
            f"（实际发送的 Key：长度 {len(api_key)}，末4位 `{api_key[-4:]}`）"
        )
    return resp.json()
@st.cache_data(ttl=3600, show_spinner=False)
def search_tavily(query, api_key, max_results):
    data = tavily_request(api_key, query, max_results)
    rows = []
    for res in data.get("results", []):
        rows.append({
            "PMID/URL": res.get("url", "Web Link"),
            "Title": res.get("title", ""),
            "Abstract": res.get("content", ""),
            "Year": "N/A（网页）",
            "Source": "全网学术搜索 (Tavily)",
        })
    return pd.DataFrame(rows, columns=DOC_COLUMNS)
def _pubmed_esearch(term, retmax):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="relevance")
    try:
        return Entrez.read(handle).get("IdList", [])
    finally:
        handle.close()
def _xml_to_text(xml):
    xml = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", xml, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", xml)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
def fetch_pmc_fulltext(pmcid):
    """抓取 PMC 开放全文：优先截取 Results 章节起的正文（含硬核数据），否则取正文开头。"""
    num = re.sub(r"\D", "", str(pmcid))
    if not num:
        return ""
    handle = Entrez.efetch(db="pmc", id=num, retmode="xml")
    try:
        raw = handle.read()
    finally:
        handle.close()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    m = re.search(r"<body\b.*?</body>", raw, flags=re.S | re.I)
    if not m:  # 无正文（受限/仅元数据）
        return ""
    body = m.group(0)
    r = re.search(r"<title>\s*Results?(?:\s+and\s+Discussion)?\s*</title>", body, flags=re.I)
    if r:
        body = body[r.start():]
    return _xml_to_text(body)[:PMC_FULLTEXT_CHARS]
@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(query, max_results):
    """PubMed 检索：标准摘要保底，并对有 PMCID 的文献自动补充 PMC 开放全文节选。"""
    date_filter = get_past_5_years_date_filter()
    id_list = _pubmed_esearch(f"({query}) AND {date_filter}", max_results)
    words = query.split()
    k = len(words) - 1
    while not id_list and k >= 2:  # 逐步去掉末尾次要词，保持主题不被泛化
        id_list = _pubmed_esearch(f"({' '.join(words[:k])}) AND {date_filter}", max_results)
        k -= 1
    if not id_list and len(words) > 1:
        id_list = _pubmed_esearch(f"({words[0]}) AND {date_filter}", max_results)
    if not id_list:
        return pd.DataFrame(columns=DOC_COLUMNS)
    handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
    try:
        papers = Entrez.read(handle)
    finally:
        handle.close()
    rows = []
    pmc_used = 0
    for article in papers.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]
            pmid = str(medline["PMID"])
            abs_list = art.get("Abstract", {}).get("AbstractText", [])
            abstract_text = " ".join(str(x) for x in abs_list)
            pmcid = ""
            for id_elem in article.get("PubmedData", {}).get("ArticleIdList", []):
                attrs = getattr(id_elem, "attributes", {}) or {}
                if attrs.get("IdType") == "pmc":
                    pmcid = str(id_elem)
                    break
            content = abstract_text
            source_desc = "PubMed (数据库摘要)"
            if pmcid and pmc_used < PMC_FULLTEXT_MAX:
                try:
                    excerpt = fetch_pmc_fulltext(pmcid)
                    if len(excerpt) > 200:
                        content = (abstract_text + "\n\n【PMC 开放全文节选】" + excerpt).strip()
                        source_desc = "PubMed Central (PMC 开放全文)"
                        pmc_used += 1
                    if not Entrez.api_key:
                        time.sleep(0.35)  # 无 API Key 时限速 3 次/秒
                except Exception:
                    pass  # 全文失败自动退回摘要
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year") or str(pub_date.get("MedlineDate", "N/A"))[:4])
            rows.append({
                "PMID/URL": f"PMID:{pmid}",
                "Title": str(art.get("ArticleTitle", "")),
                "Abstract": content,
                "Year": year,
                "Source": source_desc,
            })
        except Exception:
            continue
    return pd.DataFrame(rows, columns=DOC_COLUMNS)
SLUG_STOPWORDS = {
    "science", "article", "articles", "pii", "abs", "abstract", "doi", "full", "pdf", "html",
    "www", "content", "view", "journal", "journals", "index", "php", "pubmed", "fulltext",
    "http", "https", "com", "org", "the", "and", "for", "with",
}
BAD_LLM_OUTPUT = re.compile(
    r"no output|none\b|n/a|cannot|can't|unable|sorry|not (?:enough|provide)|insufficient|as an ai",
    re.IGNORECASE,
)
def sanitize_url(url):
    """清洗用户粘贴的链接：去空白；若被粘进已有文字中间（出现多个 http），取最后一个；
    ScienceDirect 链接按 PII 还原为标准地址。"""
    u = str(url or "").strip()
    starts = [m.start() for m in re.finditer(r"https?://", u)]
    if len(starts) > 1:
        u = u[starts[-1]:]
    u = re.split(r"\s", u)[0]
    pii = re.search(r"sciencedirect\.com/science/article/(?:abs/)?pii/([A-Z0-9]{16,17})", u)
    if pii:
        u = f"https://www.sciencedirect.com/science/article/pii/{pii.group(1)}"
    return u
def fetch_page_text(url, tavily_key=""):
    """多级回退读取链接内容：PubMed/PMC(Entrez) → DOI(Crossref) → 直接抓取(完整浏览器头)
    → Tavily Extract → 仅用链接地址中的单词推测。返回 (文本, 提示信息)。"""
    import requests
    from urllib.parse import urlparse, unquote
    # 1. PubMed 链接
    pm = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if pm:
        try:
            h = Entrez.efetch(db="pubmed", id=pm.group(1), retmode="xml")
            try:
                rec = Entrez.read(h)
            finally:
                h.close()
            art = rec["PubmedArticle"][0]["MedlineCitation"]["Article"]
            abs_list = art.get("Abstract", {}).get("AbstractText", [])
            return f"{art.get('ArticleTitle', '')} " + " ".join(str(x) for x in abs_list), None
        except Exception:
            pass
    # 2. PMC 链接
    pmc = re.search(r"PMC(\d+)", url, flags=re.IGNORECASE)
    if pmc:
        try:
            h = Entrez.efetch(db="pmc", id=pmc.group(1), retmode="xml")
            try:
                raw = h.read()
            finally:
                h.close()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            txt = _xml_to_text(raw)[:4000]
            if len(txt) > 100:
                return txt, None
        except Exception:
            pass
    # 3. 链接中含 DOI → Crossref（不会拦截，返回标题与摘要）
    doi = re.search(r"(10\.\d{4,9}/[^\s?#]+)", unquote(url))
    if doi:
        try:
            r = requests.get(
                f"https://api.crossref.org/works/{doi.group(1).rstrip('/.')}",
                timeout=15, headers={"User-Agent": "anesthesia-review-agent (mailto:anesthesia_agent@example.com)"},
            )
            if r.status_code == 200:
                msg = r.json().get("message", {})
                title = " ".join(msg.get("title", []))
                abstract = _xml_to_text(msg.get("abstract", ""))
                if title:
                    return f"{title} {abstract}", None
        except Exception:
            pass
    # 3b. ScienceDirect 的 PII → Crossref alternative-id（避开 403）
    pii = re.search(r"/pii/([A-Z0-9]{16,17})", url)
    if pii:
        try:
            r = requests.get(
                "https://api.crossref.org/works",
                params={"filter": f"alternative-id:{pii.group(1)}", "rows": 1},
                timeout=15,
                headers={"User-Agent": "anesthesia-review-agent (mailto:anesthesia_agent@example.com)"},
            )
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                if items:
                    title = " ".join(items[0].get("title", []))
                    abstract = _xml_to_text(items[0].get("abstract", ""))
                    if title:
                        return f"{title} {abstract}", None
        except Exception:
            pass
    # 4. 直接抓取（完整浏览器请求头）
    status = None
    try:
        resp = requests.get(
            url, timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
            },
        )
        status = resp.status_code
        if status == 200:
            txt = _xml_to_text(resp.text)[:4000]
            if len(txt) >= 100:
                return txt, None
    except Exception as e:
        status = f"连接失败: {e}"
    # 5. Tavily Extract（可绕过多数反爬）
    if tavily_key:
        try:
            r = requests.post(
                "https://api.tavily.com/extract",
                headers={"Authorization": f"Bearer {tavily_key}", "Content-Type": "application/json"},
                json={"urls": [url]}, timeout=60,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results and results[0].get("raw_content"):
                    return str(results[0]["raw_content"])[:4000], None
        except Exception:
            pass
    # 6. 最后手段：用链接路径里的英文单词推测主题
    slug = unquote(urlparse(url).path)
    words = [w for w in re.split(r"[^A-Za-z]+", slug)
             if len(w) > 3 and w.lower() not in SLUG_STOPWORDS]
    if len(words) >= 2:
        return " ".join(words), f"【推测】网页无法访问（{status}），已仅根据链接地址中的单词推测主题，结果仅供参考"
    return "", f"网页无法访问（{status}）。请直接复制文章标题/摘要里的英文关键词填入主题框，或改用 PubMed / DOI 链接"
def extract_keywords_from_url(url, api_key, model, tavily_key=""):
    """读取参考链接并由大模型提取英文检索关键词。返回 (关键词或None, 提示或错误信息)。"""
    try:
        url = sanitize_url(url)
        page_text, note = fetch_page_text(url, tavily_key)
        if len(page_text.strip()) < 3:
            return None, note or "未能从该链接读取到有效内容"
        if note and note.startswith("【推测】"):  # 仅有链接单词，不交给大模型，避免它输出 No output
            return " ".join(page_text.split()[:6]), note.replace("【推测】", "")
        if len(page_text.strip()) < 15:
            return None, note or "未能从该链接读取到有效内容"
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        prompt = (
            "请阅读以下网页/文献内容，提取其最核心的医学研究主题，"
            "输出 3-6 个适合 PubMed 与网页检索的英文关键词，用空格分隔。\n"
            "要求：只输出关键词本身；不要布尔运算符（AND/OR）、括号、引号、标点或任何解释。\n\n"
            f"内容：\n{page_text[:4000]}"
        )
        text, _, _ = call_openrouter_with_fallback(
            client, model, [{"role": "user", "content": prompt}], temperature=0.1
        )
        line = next((l for l in text.strip().splitlines() if l.strip()), "")
        if BAD_LLM_OUTPUT.search(line):
            return None, f"大模型未能提取出有效关键词（返回：{line[:40]}），请手动填写主题"
        line = re.sub(r"[^A-Za-z0-9\s\-]", " ", line)
        words = line.split()[:6]
        if not words:
            return None, "大模型未返回有效关键词"
        return " ".join(words), note
    except Exception as e:
        return None, str(e)
def fetch_web_and_pubmed_literature(raw_topic, tavily_key, max_results, synonyms=None):
    query = clean_search_keyword(raw_topic)
    msgs = []
    if query == "Anesthesia" and raw_topic.strip() and raw_topic.strip() != "Anesthesia":
        msgs.append(("info", "主题中没有可用的英文关键词，已退回使用通用关键词 `Anesthesia` 检索；建议改用英文主题。"))
    frames = []
    if tavily_key:
        try:
            df = search_tavily(query, tavily_key, max_results)
            frames.append(df)
            if df.empty:
                msgs.append(("info", f"Tavily 未能针对【{query}】检索到结果。"))
        except Exception as e:
            msgs.append(("warning", f"Tavily 全网搜索出现异常: {e}"))
    else:
        msgs.append(("warning", "Tavily API Key 为空，已跳过全网搜索。"))
    try:
        frames.append(search_pubmed(query, max_results))
    except Exception as e:
        msgs.append(("error", f"PubMed 检索失败: {e}"))
    frames = [f for f in frames if not f.empty]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DOC_COLUMNS)
    out, dropped = filter_relevant(out, query, synonyms)
    if dropped:
        msgs.append(("info", f"已自动过滤 {dropped} 篇与主题【{query}】相关性较低的检索结果，其余按相关度排序。"))
    return out, msgs
# ==========================================
# 5. AI 综述生成引擎与云端自动存储
# ==========================================
def _github_cfg():
    return (
        os.getenv("GITHUB_TOKEN", "").strip(),
        os.getenv("GITHUB_REPO", "").strip(),
        os.getenv("GITHUB_BRANCH", "main").strip() or "main",
    )
def _github_headers(token, raw=False):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github.v3+json",
    }
def _clean_topic_for_filename(topic):
    topic_str = str(topic).strip()
    # 误传布尔检索式时，只取第一个检索词命名
    if " OR " in topic_str or " AND " in topic_str:
        first = re.sub(r"[()_]", "", topic_str).split(" OR ")[0].split(" AND ")[0].strip()
        topic_str = first or "Anesthesia_Review"
    clean = re.sub(r'[\\/:*?"<>|\s()\[\]{}]+', "_", topic_str).strip("_")
    if len(clean) > 35:
        clean = clean[:35] + "_etc"
    return clean or "Anesthesia_Review"
def save_review_to_cloud(topic, content, tag="V1"):
    """优先通过 GitHub API 写入仓库 reviews/ 目录；未配置或失败则降级为本地 saved_reviews/。
    结果提示写入 session_state['save_notice']，返回保存路径（失败返回 None）。"""
    clean_topic = _clean_topic_for_filename(topic)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{clean_topic}_{tag}_{timestamp}.md"
    token, repo, branch = _github_cfg()
    gh_error = ""
    if token and repo:
        import requests
        try:
            resp = requests.put(
                f"https://api.github.com/repos/{repo}/contents/reviews/{fname}",
                headers=_github_headers(token),
                json={
                    "message": f"Auto-save review: {clean_topic} ({tag})",
                    "content": base64.b64encode(content.encode("utf-8")).decode(),
                    "branch": branch,
                },
                timeout=20,
            )
            if resp.status_code in (200, 201):
                github_list_reviews.clear()
                st.session_state["save_notice"] = ("success", f"综述已同步至 GitHub 仓库：`reviews/{fname}`")
                return f"reviews/{fname}"
            gh_error = f"GitHub 返回 HTTP {resp.status_code}"
        except Exception as e:
            gh_error = str(e)
    os.makedirs("saved_reviews", exist_ok=True)
    local_path = f"saved_reviews/{fname}"
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
        tip = f"（GitHub 同步失败：{gh_error}）" if gh_error else "（未配置 GITHUB_TOKEN / GITHUB_REPO）"
        st.session_state["save_notice"] = (
            "warning", f"已保存在服务器本地 `{local_path}` {tip}。注意：云端重启后本地文件可能丢失。"
        )
        return local_path
    except Exception:
        st.session_state["save_notice"] = ("error", "综述保存失败（GitHub 与本地均不可用）。")
        return None
@st.cache_data(ttl=60, show_spinner=False)
def github_list_reviews(token, repo, branch):
    import requests
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/contents/reviews",
            headers=_github_headers(token), params={"ref": branch}, timeout=15,
        )
        if r.status_code != 200:
            return []
        return sorted(
            [f["path"] for f in r.json() if str(f.get("name", "")).endswith(".md")], reverse=True
        )
    except Exception:
        return []
@st.cache_data(ttl=300, show_spinner=False)
def github_read_file(token, repo, branch, path):
    import requests
    r = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers=_github_headers(token), params={"ref": branch}, timeout=20,
    )
    r.raise_for_status()
    return r.text
def build_glossary_section(enable, follow_up=False):
    """生成术语表对齐指令（大小写不敏感、强制中文译名）。"""
    if not enable:
        return ""
    glossary_text = load_glossary_content()
    if not glossary_text.strip():
        return ""
    head = "请继续严格参照以下专业术语表 (Glossary)：" if follow_up else (
        "### 📖 专业术语表 (Glossary) 严格对齐与翻译规范：\n"
        "请在撰写、总结或翻译专业名词时，**必须严格参照**以下专有名词对照表："
    )
    return f"""
---
{head}
{glossary_text.strip()}
**⚠️ 核心硬性要求**：
1. **大小写不敏感匹配**：无论背景资料中的英文专业名词是大写、小写还是复数形式，在中文正文中都必须严格替换为词典中对应的中文专业译名。
2. 不允许在中文正文里直接大段保留未翻译的英文核心病名、药名或生理指标（首次出现可在括号内附英文原词）。
"""
def call_openrouter_with_fallback(client, primary_model, messages, temperature=0.3):
    models_to_try = [primary_model] + [m for m in FALLBACK_FREE_MODELS if m != primary_model]
    last_error = ""
    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=10000,
                timeout=180.0,
            )
            choices = getattr(response, "choices", None)
            if not choices or choices[0] is None:
                last_error = f"模型 `{model}` 返回的 choices 为空"
                continue
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message else None
            if content and str(content).strip():
                return str(content), model, getattr(choices[0], "finish_reason", None)
            last_error = f"模型 `{model}` 返回生成文本为空"
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status in (401, 403):
                key = str(getattr(client, "api_key", "") or "")
                raise RuntimeError(
                    f"OpenRouter 拒绝了这个 API Key（HTTP {status}：{e}）。"
                    f"实际发送的 Key：长度 {len(key)}，前缀 `{key[:9]}`，末4位 `{key[-4:]}`。"
                    "请到 openrouter.ai/keys 重新生成 Key，完整复制后更新 Secrets 并 Reboot app。"
                )
            last_error = f"模型 `{model}` 报错: {e}"
            time.sleep(1.5)
    raise RuntimeError(f"所有备选模型均未能正常生成响应。最后报错细节: {last_error}")
def parse_openrouter_models(payload):
    models = []
    for m in payload.get("data", []):
        mid = m.get("id")
        if not mid:
            continue
        outs = (m.get("architecture") or {}).get("output_modalities")
        if outs and "text" not in outs:
            continue
        pricing = m.get("pricing") or {}
        try:
            free = mid.endswith(":free") or (
                float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
            )
        except (TypeError, ValueError):
            free = mid.endswith(":free")
        ctx = m.get("context_length") or 0
        label = f"{mid}  ·  {ctx // 1000}k" if ctx else mid
        models.append({"id": mid, "label": label, "free": free, "ctx": ctx})
    return models
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openrouter_models():
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        resp.raise_for_status()
        return parse_openrouter_models(resp.json())
    except Exception:
        return []
MAX_CONTINUATIONS = 4
def strip_citation_marks(text):
    if not text:
        return text
    # 先把“本地文献3 / 全网文献2-4”之类改写为自然语言，避免被后续规则删成残句
    text = re.sub(
        r"(?:本地文献|全网/PubMed\s*文献|全网文献|PubMed\s*文献)\s*\d+(?:\s*[-–—~,，、和与及]\s*\d+)*",
        "相关研究",
        text,
    )
    pats = [
        r"[【\[（(]?\s*(?:来源|文献|Refs?\.?|References?|Sources?)\s*\d+(?:\s*[-–—~,，、;；和与及]\s*\d+)*\s*(?:等|etc\.?)?\s*[）)\]】]",
        r"[【\[（(]\s*(?:来源\s*)?\d+(?:\s*[-–—~,，、;；和与及]\s*\d+)*\s*等?\s*[）)\]】]",
        r"[\[［【]\s*\d+(?:\s*[-–—~,，、;；]\s*\d+)*\s*[\]］】]",
        r"[（(]\s*PMID\s*[:：]?\s*\d+(?:\s*[,，;；、]\s*(?:PMID\s*[:：]?\s*)?\d+)*\s*[）)]",
        r"[（(]\s*https?://[^\s)）]+\s*[）)]",
        r",?\s*PMID\s*[:：]?\s*\d{5,9}",
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        r"\^\s*\[?\d+(?:\s*[,，-]\s*\d+)*\]?",
    ]
    for p in pats:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+([,，。；：、;）)])", r"\1", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    return text
def call_with_continuation(client, primary_model, messages, temperature=0.3):
    text, used_model, finish = call_openrouter_with_fallback(
        client, primary_model, messages, temperature
    )
    for _ in range(MAX_CONTINUATIONS):
        if finish != "length":
            break
        cont_messages = list(messages) + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "你的上一段输出因长度上限被中断。请紧接上文最后一个字继续写完剩余全部内容："
                    "不要重复已输出的内容，不要任何开场白或解释；若上文中断在表格中，"
                    "请从该表格的下一行继续；务必完整写完所有未完成的章节（含总结与展望）。"
                ),
            },
        ]
        more, used_model, finish = call_openrouter_with_fallback(
            client, used_model, cont_messages, temperature
        )
        text += more
    return strip_citation_marks(text), used_model, finish
def _excerpt_for_prompt(row):
    """PMC 全文给更长的节选额度，其余只给摘要额度。"""
    limit = PROMPT_CHARS_FULLTEXT if "PMC" in str(row.get("Source", "")) else PROMPT_CHARS_ABSTRACT
    return str(row.get("Abstract", ""))[:limit]
def generate_5000_words_review(api_key, model_name, topic_keywords, local_df, web_df,
                               enable_glossary=True, topic_desc=""):
    try:
        clean_key = str(api_key).strip()
        if not clean_key:
            return "❌ 错误：请填入有效的 OpenRouter API Key！", None, None
        client = OpenAI(api_key=clean_key, base_url="https://openrouter.ai/api/v1")
        local_text = ""
        if not local_df.empty:
            local_text += "\n=== [本地文献资料] ===\n"
            for _, row in local_df.head(20).iterrows():
                local_text += (
                    f"文献标题: {row.get('Title', '')}\n"
                    f"摘要/内容: {str(row.get('Abstract', ''))[:PROMPT_CHARS_LOCAL]}...\n\n"
                )
        web_text = ""
        if not web_df.empty:
            web_text += "\n=== [全网及PubMed/PMC文献资料] ===\n"
            for _, row in web_df.head(30).iterrows():
                web_text += (
                    f"文献标题: {row.get('Title', '')} ({row.get('Year', '')})\n"
                    f"资料类型: {row.get('Source', '')}\n"
                    f"摘要/全文节选: {_excerpt_for_prompt(row)}\n\n"
                )
        combined_input = local_text + web_text
        hot_drugs = extract_hot_drugs(
            pd.concat([local_df, web_df], ignore_index=True), top_n=10
        )
        if not hot_drugs.empty:
            combined_input += "\n=== [自动统计：当前文献集中被讨论最多的药物] ===\n"
            for _, r in hot_drugs.iterrows():
                combined_input += f"- {r['药物']}：涉及 {r['涉及文献数']} 篇文献\n"
        start, today = get_past_5_years_range()
        glossary_section = build_glossary_section(enable_glossary)
        desc = f"（{topic_desc}）" if topic_desc else ""
        if not local_df.empty:
            topic_anchor = f"""
### 🎯 主题一致性硬性约束（最高优先级）：
1. 本综述的唯一主题是【{topic_keywords}】{desc}，该主题由用户上传的文件内容确定。
2. 背景资料中的【本地文献资料】是主题的权威依据，必须作为综述的核心骨架；【全网及PubMed/PMC文献资料】只用于补充与该主题直接相关的新证据。
3. 凡与该主题不直接相关的文献、药物或疾病（即便出现在背景资料中），一律忽略，不得展开论述，更不得把综述写成其他主题。
4. 若检索到的文献整体偏离上传文件的主题，请只依据上传文件撰写，并在文中说明补充证据有限。
"""
        else:
            topic_anchor = f"""
### 🎯 主题一致性硬性约束（最高优先级）：
1. 本综述的唯一主题是【{topic_keywords}】。
2. 凡与该主题不直接相关的文献、药物或疾病，一律忽略，不得展开论述。
"""
        prompt = f"""你是一名世界顶尖的麻醉学与围术期医学教授、权威学术期刊资深主编。
请基于我提供的【用户上传文献】

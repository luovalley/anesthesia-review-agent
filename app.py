import base64
import datetime
import hmac
import html
import inspect
import io
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


def render_auth_header():
    """在侧边栏最顶部渲染退出登录按钮和词典加载状态"""
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
    return " ".join(words[:5]) if words else "Anesthesia"


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
                f"{query} anesthesia perioperative trial review "
                f"recent research {start.year}-{today.year}"
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

    if not id_list and len(query.split()) > 1:
        id_list = _pubmed_esearch(f"({query.split()[0]}) AND {date_filter}", max_results)
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
                timeout=15, headers={"User-Agent": "anesthesia-review-app (mailto:anesthesia_agent@example.com)"},
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
                headers={"User-Agent": "anesthesia-review-app (mailto:anesthesia_agent@example.com)"},
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


def fetch_web_and_pubmed_literature(raw_topic, tavily_key, max_results):
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
                    "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
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
        headers=_github_headers(token, raw=True), params={"ref": branch}, timeout=20,
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


def generate_5000_words_review(api_key, model_name, topic_keywords, local_df, web_df, enable_glossary=True):
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
                    f"摘要/内容: {str(row.get('Abstract', ''))[:PROMPT_CHARS_ABSTRACT]}...\n\n"
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

        prompt = f"""你是一名世界顶尖的麻醉学与围术期医学教授、权威学术期刊资深主编。
请基于我提供的【用户上传文献】以及【全网学术搜索与 PubMed/PMC 数据库近 5 年检索到的文献（部分含 PMC 开放全文）】，围绕主题 **【{topic_keywords}】**，撰写一篇高度专业、结构严谨的**《麻醉学与围术期医学前沿知识更新与重难点热点深度综述》**，全文约 5000 字。

背景资料如下：
{combined_input}
{glossary_section}
---

### 📝 论文撰写规范与结构要求（必须完全输出完以下所有章节，绝对不能中断）：

#### 一、 摘要与关键词 (Abstract & Keywords)
* **必须使用中文**撰写约 300 字的精炼摘要，涵盖研究背景、核心进展、核心争论及未来方向；关键词需提供中英文对照。

#### 二、 主题背景与近五年研究演进 (Introduction)
* 梳理该领域（【{topic_keywords}】）近 5 年（{start.year}-{today.year}年）的发展轨迹与演变逻辑。

#### 三、 近五年核心学术突破与重大研究进展 (Major Breakthroughs)
* **【硬核数据强制要求】**：必须深入引用背景资料中具体临床试验或前瞻性研究的**量化数据**，例如：
  1. **具体样本量**（如“一项纳入 420 例患者的随机对照试验显示……”）；
  2. **具体的生理/药理量化指标**（如均数±标准差、OR/RR/HR 及 95%CI、P 值）；
  3. **具体的给药方案与终点数据**。
  资料中确无具体数值时，请如实说明“原文未提供具体数值”，**严禁编造数据**，不能只写笼统结论。

#### 四、 当前热点领域的核心争论与未解决的临床困境 (Key Controversies)
* **重点章节**：深入剖析当前学术界存在的重大争论与分歧点（疗效争议、安全隐患、矛盾结论等），并直接对比不同研究在具体终点指标上的**数据冲突与对立结果**。

#### 五、 机制探讨与方法学瓶颈 (Mechanistic Insights)
* 讨论当前基础与临床研究面临的方法学瓶颈。

#### 六、 临床转化与围术期管理落地建议 (Clinical Implications)
* **必须包含一份完整 Markdown 表格**：
| 环节 | 建议措施 | 证据等级与推荐强度 |
（请填入具体的术前评估、麻醉诱导、血流动力学、肌松监测、镇痛策略等表格行）

#### 七、 总结与展望 (Conclusion)
* 总结全篇，提出未来 3-5 年最值得投入的科研切入点。

---

### ⚠️ 输出格式极其严格的约束（切记）：
1. **全文（包括第一章摘要与关键词）必须完全使用中文撰写**，专业名词可在括号中附英文原词。
2. 必须完整输出完所有的 7 个章节，**特别是“六、临床转化表格”和“七、总结与展望”必须完整撰写完，绝对不能中途截断**！
3. 请使用标准 Markdown 格式，语言专业严谨。
4. **绝对禁止在正文中出现任何形式的文献序号、标号或标记**：包括但不限于 `[1]`、`[1,2]`、`(来源2、3等)`、上标数字、或者任何形式的“文献1”、“本地文献”、“全网文献”、“PMID”等引用标识。正文表述文献时，必须完全采用无序号的自然语言学术叙述。
5. 论述必须以上方【背景资料】中的文献为依据，严禁编造文献、作者、数据或 PMID；资料不足之处请明确说明。
"""

        messages = [
            {
                "role": "system",
                "content": "你是一名精通麻醉学前沿研究的权威期刊主编。你的输出必须完整，绝不中途截断段落或表格，且正文中绝不出现任何参考文献序号或编号。",
            },
            {"role": "user", "content": prompt},
        ]

        result, used_model, finish = call_with_continuation(
            client, model_name, messages, temperature=0.3
        )
        wrapped = wrap_article(result, used_model, finish)
        save_review_to_cloud(topic_keywords, wrapped, tag="V1")
        return wrapped, result, messages
    except Exception as e:
        return f"❌ 深度综述生成失败: {e}", None, None


def wrap_article(result, used_model, finish):
    note = ""
    if finish == "length":
        note = "\n\n> ⚠️ *输出经多次自动续写后仍未写完，可在下方提交“继续补全未完成章节”，或更换模型。*"
    return (
        f"> 💡 *本篇深度综述由 AI 模型 `{used_model}` 基于全网文献生成，"
        f"内容须经专业人员核实后方可使用*\n\n" + result + note
    )


REVISE_MODE = "✏️ 修改综述（输出完整新版本）"
ASK_MODE = "💬 提问讨论（不改动文章）"


NEED_SEARCH_RE = re.compile(r"NEED_SEARCH\s*[:：]\s*([^\n]*)", re.IGNORECASE)
ASK_SEARCH_MAX = 8  # 提问时补充检索 PubMed 的文献数


def follow_up_review(api_key, model_name, base_messages, current_article,
                     qa_history, user_text, mode, enable_glossary=True, topic=""):
    """返回 (回复文本, 实际模型, finish_reason, 补充检索信息或None)。
    提问模式：先判断背景资料是否足以回答；不足则自动检索 PubMed，基于新检索到的文献回答。"""
    clean_key = str(api_key).strip()
    if not clean_key:
        raise RuntimeError("请填入有效的 OpenRouter API Key！")
    client = OpenAI(api_key=clean_key, base_url="https://openrouter.ai/api/v1")

    glossary_directive = build_glossary_section(enable_glossary, follow_up=True)
    context = (
        list(base_messages)
        + [{"role": "assistant", "content": current_article}]
        + list(qa_history[-6:])
    )

    # ---------- 修改模式 ----------
    if mode == REVISE_MODE:
        directive = (
            f"请根据下面的修改意见修订上述综述，**输出修订后的完整全文**"
            f"(保持 7 个章节结构与 Markdown 表格，不得省略、缩写或中途截断；"
            f"只能依据背景资料中的文献，严禁编造；**正文中严禁出现任何形式的参考文献序号、标号、(来源...) 或 PMID**；直接输出文章，不要额外解释)。"
            f"{glossary_directive}\n\n"
            f"修改意见：{user_text}"
        )
        text, used, finish = call_with_continuation(
            client, model_name, context + [{"role": "user", "content": directive}], temperature=0.3
        )
        return text, used, finish, None

    # ---------- 提问模式：先看背景资料够不够 ----------
    judge_directive = (
        "请先判断：上述【背景资料】与综述中的文献，是否足以回答下面的问题。\n"
        "- 如果足以：请基于背景资料简明准确地回答，不要重写全文；只能依据背景资料中的文献，严禁编造；回答中不要出现文献编号或 PMID。\n"
        "- 如果背景资料没有涉及该问题所需的证据（或只有无关内容）：**不要回答、不要凭空编造**，"
        "只输出一行：NEED_SEARCH: 用于 PubMed 检索的英文关键词（3-6 个单词，空格分隔，不含 AND/OR、括号、引号）。\n"
        f"{glossary_directive}\n\n问题：{user_text}"
    )
    text, used, finish = call_with_continuation(
        client, model_name, context + [{"role": "user", "content": judge_directive}], temperature=0.4
    )
    m = NEED_SEARCH_RE.search(text)
    if not m:
        return text, used, finish, None

    # ---------- 背景资料不足 → 检索 PubMed ----------
    kw = re.sub(r"[^A-Za-z0-9\s\-]", " ", m.group(1))
    query = " ".join(kw.split()[:6]) or clean_search_keyword(topic)
    try:
        df = search_pubmed(query, ASK_SEARCH_MAX)
    except Exception as e:
        return (
            f"背景资料中没有涉及该问题的证据，补充检索 PubMed（关键词：`{query}`）时出错：{e}",
            used, finish, {"query": query, "df": pd.DataFrame(columns=DOC_COLUMNS)},
        )
    if df.empty:
        return (
            f"背景资料中没有涉及该问题的证据，且在 PubMed 近 5 年文献中用关键词 `{query}` 也未检索到相关文献，"
            "暂时无法给出有依据的回答。可以换一种问法，或上传相关文献后再试。",
            used, finish, {"query": query, "df": df},
        )

    extra_text = "\n=== [针对本问题补充检索到的 PubMed/PMC 文献] ===\n"
    for _, row in df.iterrows():
        extra_text += (
            f"文献标题: {row.get('Title', '')} ({row.get('Year', '')})\n"
            f"资料类型: {row.get('Source', '')}\n"
            f"摘要/全文节选: {_excerpt_for_prompt(row)}\n\n"
        )
    search_directive = (
        "原背景资料中没有涉及下面这个问题，系统已补充检索 PubMed，结果如下：\n"
        f"{extra_text}\n"
        "请**仅依据这些补充文献**简明准确地回答问题（不要重写全文）；\n"
        "1. 回答开头请用一句话说明：原综述资料未涉及该问题，以下回答基于补充检索的 PubMed 文献；\n"
        "2. 若补充文献仍不足以回答，请直接说明证据不足，严禁编造文献、作者或数据；\n"
        "3. 有具体样本量或数值时请如实引用；正文中不要出现文献编号、[1] 或 PMID。"
        f"{glossary_directive}\n\n问题：{user_text}"
    )
    answer, used2, finish2 = call_with_continuation(
        client, used, context + [{"role": "user", "content": search_directive}], temperature=0.4
    )

    refs = []
    for _, row in df.iterrows():
        url, shown = doc_link(row.get("PMID/URL"))
        title = _md_escape(row.get("Title"))
        year = str(row.get("Year") or "N/A")
        refs.append(f"- [{title}]({url}) （{year}）" if url else f"- {title} （{year}）")
    answer += (
        f"\n\n---\n🔎 *背景资料未涉及该问题，已补充检索 PubMed（关键词：`{query}`），"
        f"本回答参考了以下 {len(df)} 篇文献：*\n\n" + "\n".join(refs)
    )
    return answer, used2, finish2, {"query": query, "df": df}


# ==========================================
# 6. Streamlit 主界面与交互
# ==========================================
MODE_1 = "1. 上传本地文献 + 全网 (Tavily/PubMed) 综合分析"
MODE_2 = "2. 直接输入主题全网搜查并撰写综述"
NO_PRESET = "-- 手动/自定义输入主题 --"

ss = st.session_state
ss.setdefault("web_df", pd.DataFrame(columns=DOC_COLUMNS))
ss.setdefault("searched_topic", "")
ss.setdefault("search_msgs", [])
ss.setdefault("article_topic", "")
ss.setdefault("versions", [])
ss.setdefault("base_messages", [])
ss.setdefault("current_raw", "")
ss.setdefault("qa_history", [])
ss.setdefault("extra_df", pd.DataFrame(columns=DOC_COLUMNS))
ss.setdefault("custom_topic_input", "")
ss.setdefault("manual_topic_input", "")


def _sync_preset():
    p = st.session_state.get("preset_select", NO_PRESET)
    st.session_state["manual_topic_input"] = "" if p == NO_PRESET else english_part(p)

st.title("💉 麻醉学全网文献热点追踪与深度综述生成系统")
st.markdown(
    "支持**上传本地文献** + **Tavily 全网学术搜索** + **PubMed 数据库**，"
    "一键撰写包含**核心争论、临床建议表格与未来方向**的学术综述。"
)

render_auth_header()

st.sidebar.header("⚙️ 配置面板")
work_mode = st.sidebar.radio("选择工作模式：", options=[MODE_1, MODE_2], index=0)

st.sidebar.markdown("---")

st.sidebar.header("🔑 API 凭证配置（自动加载）")
_AC = {"autocomplete": "off"} if "autocomplete" in inspect.signature(st.text_input).parameters else {}

# 修改 OpenRouter 输入框，加上 key="or_key_input"
openrouter_input = st.sidebar.text_input(
    "API Key (OpenRouter / DeepSeek)",
    value="", 
    type="password", 
    key="or_key_input",  # <--- 加上这一行，打破 Streamlit 状态缓存
    **_AC,
    placeholder="已在后台配置，留空即使用" if ENV_OPENROUTER_KEY else "请输入",
)
_typed_or = normalize_key(openrouter_input)
openrouter_api_key = _typed_or or ENV_OPENROUTER_KEY

# 修改 Tavily 输入框，加上 key="tavily_key_input"
tavily_input = st.sidebar.text_input(
    "Tavily 全网搜索 API Key",
    value="", 
    type="password", 
    key="tavily_key_input",  # <--- 加上这一行
    **_AC,
    placeholder="已在后台配置，留空即使用" if ENV_TAVILY_KEY else "请输入",
)
_typed_tavily = normalize_key(tavily_input)
tavily_api_key = _typed_tavily or ENV_TAVILY_KEY

MODEL_CUSTOM = "✏️ 自定义模型 ID…"
live_models = fetch_openrouter_models()

free_models = [m for m in live_models if m["free"]]
if free_models:
    pool = free_models
    labels = {m["id"]: m["label"] for m in pool}
    live_ids = {m["id"] for m in free_models}
    ordered = [m for m in [DEFAULT_OPENROUTER_MODEL] + FALLBACK_FREE_MODELS if m in labels]
    ordered = list(dict.fromkeys(ordered)) + sorted(i for i in labels if i not in ordered)
    _alive = [m for m in FALLBACK_FREE_MODELS if m in live_ids or m == "openrouter/free"]
    if len(_alive) > 1:
        FALLBACK_FREE_MODELS = _alive
else:
    labels = {m: m for m in FALLBACK_FREE_MODELS}
    ordered = list(dict.fromkeys([DEFAULT_OPENROUTER_MODEL] + FALLBACK_FREE_MODELS))

options = ordered + [MODEL_CUSTOM]
default_idx = options.index(DEFAULT_OPENROUTER_MODEL) if DEFAULT_OPENROUTER_MODEL in options else 0
choice = st.sidebar.selectbox(
    "选择大模型",
    options=options,
    index=default_idx,
    format_func=lambda x: x if x == MODEL_CUSTOM else labels.get(x, x),
)
if choice == MODEL_CUSTOM:
    openrouter_model = st.sidebar.text_input(
        "自定义模型 ID", value=DEFAULT_OPENROUTER_MODEL,
    ).strip() or DEFAULT_OPENROUTER_MODEL
else:
    openrouter_model = choice

st.sidebar.markdown("---")

enable_glossary_checkbox = st.sidebar.checkbox("启用麻醉学专业术语动态对齐", value=True)

max_doc_count = st.sidebar.slider(
    "检索文献数量限制",
    min_value=10, max_value=100, value=50, step=10,
)

st.sidebar.markdown("---")

# ---- 参考链接智能提取检索词（须在主题输入框之前执行，才能写入其 session_state）----
st.sidebar.markdown("🔗 **或通过参考链接智能提取检索词**")
ref_url = st.sidebar.text_input(
    "参考文章/网页链接：",
    key="ref_url_input",
    placeholder="https://pubmed.ncbi.nlm.nih.gov/40322031/",
    help="粘贴前请先清空输入框（Ctrl+A 全选后再粘贴），避免链接被插进旧文字中间。",
)
if st.sidebar.button("📥 读取链接并提取检索词"):
    if not ref_url.strip():
        ss["url_msg"] = ("warning", "请先输入有效的网址链接。")
    elif not openrouter_api_key:
        ss["url_msg"] = ("warning", "请先配置 OpenRouter API Key（用于提取关键词）。")
    else:
        _clean_url = sanitize_url(ref_url)
        if _clean_url != ref_url.strip():
            st.sidebar.info(f"已自动清洗链接：{_clean_url}")
        with st.spinner("正在读取链接并提取核心检索词..."):
            _q, _err = extract_keywords_from_url(
                _clean_url, openrouter_api_key, openrouter_model, tavily_api_key
            )
        if _q:
            ss["custom_topic_input" if work_mode == MODE_1 else "manual_topic_input"] = _q
            ss["url_msg"] = ("success", f"已提取检索词并填入主题框：{_q}" + (f"\n\n⚠️ {_err}" if _err else ""))
            st.rerun()
        else:
            ss["url_msg"] = ("warning", f"未能提取检索词：{_err}")
_url_msg = ss.pop("url_msg", None)
if _url_msg:
    getattr(st.sidebar, _url_msg[0])(_url_msg[1])

st.sidebar.markdown("---")

local_df = pd.DataFrame(columns=DOC_COLUMNS)
search_topic = ""

if work_mode == MODE_1:
    uploaded_files = st.sidebar.file_uploader(
        "上传本地文献（支持 PDF, Docx, TXT, CSV）：",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )
    custom_topic = st.sidebar.text_input(
        "补充或指定搜索的主题关键词（可选）：",
        key="custom_topic_input", placeholder="例如：Dexmedetomidine neuroprotection",
    )
    if uploaded_files:
        local_df = parse_uploaded_files(uploaded_files)
        st.sidebar.success(f"已解析 {len(local_df)} 条本地文献记录！")

    if custom_topic.strip():
        search_topic = custom_topic.strip()
    elif not local_df.empty:
        combined_content = " ".join(local_df["Abstract"].tolist() + local_df["Title"].tolist())
        search_topic = clean_search_keyword(combined_content)
else:
    selected_preset = st.sidebar.selectbox(
        "💡 从 33 项权威热点词库中选择：",
        options=[NO_PRESET] + list(CLINICAL_TOPICS.keys()),
        key="preset_select", on_change=_sync_preset,
    )
    manual_input = st.sidebar.text_input(
        "✏️ 请确认或手动修改检索主题（建议使用英文）：",
        key="manual_topic_input", placeholder="例如：Remimazolam vs Propofol sedation",
    )
    search_topic = manual_input.strip()

st.sidebar.markdown("---")
if st.sidebar.button("🔍 开始检索文献", type="primary", disabled=not search_topic):
    with st.spinner(f"正在全网（Tavily + PubMed）检索与【{search_topic}】相关的近 5 年文献..."):
        ss["web_df"], ss["search_msgs"] = fetch_web_and_pubmed_literature(
            search_topic, tavily_api_key, max_doc_count
        )
        ss["searched_topic"] = search_topic
if not search_topic:
    st.sidebar.caption("👆 请先输入或选择一个主题（模式 1 也可直接上传文件）。")

web_df = ss["web_df"]
review_topic = ss["searched_topic"] or search_topic

st.markdown("### 📊 当前数据准备状态")
c1, c2, c3, c4 = st.columns(4)
c1.metric("解析本地文献数", f"{len(local_df)} 篇")
c2.metric("全网/PubMed 关联文献", f"{len(web_df)} 篇")
_pmc_n = int(web_df["Source"].astype(str).str.contains("PMC").sum()) if not web_df.empty else 0
c3.metric("其中含 PMC 开放全文", f"{_pmc_n} 篇")
c4.metric("拟撰写综述主题", review_topic or "未指定")

for level, text in ss["search_msgs"]:
    getattr(st, level)(text)

st.markdown("---")

all_docs = pd.concat([local_df, web_df, ss["extra_df"]], ignore_index=True)
all_docs = all_docs.drop_duplicates(subset=["PMID/URL"], keep="first").reset_index(drop=True)

tab1, tab2 = st.tabs(
    ["📝 深度知识更新文章", "📚 数据库与全网文献明细"]
)

with tab1:
    st.subheader("📝 AI 深度撰写：知识更新综述")
    st.caption(
        "系统将融合本地文献与全网学术搜索（Tavily）以及 PubMed 数据库近 5 年的研究，"
        "探讨学术争论并输出临床转化表格。生成后会自动保存在服务器云端。"
    )

    _gh_token, _gh_repo, _gh_branch = _github_cfg()
    history = []  # (显示名, 类型, 路径)
    if _gh_token and _gh_repo:
        history += [(f"☁️ GitHub · {os.path.basename(p)}", "gh", p)
                    for p in github_list_reviews(_gh_token, _gh_repo, _gh_branch)]
    if os.path.exists("saved_reviews"):
        history += [(f"💾 本地 · {f}", "local", os.path.join("saved_reviews", f))
                    for f in sorted(os.listdir("saved_reviews"), reverse=True) if f.endswith(".md")]
    if history:
        with st.expander("📁 查看已自动保存的历史综述文件"):
            _pick = st.selectbox("选择历史文件：", range(len(history)), format_func=lambda i: history[i][0])
            _label, _kind, _path = history[_pick]
            try:
                if _kind == "gh":
                    cloud_content = github_read_file(_gh_token, _gh_repo, _gh_branch, _path)
                else:
                    with open(_path, "r", encoding="utf-8") as cf:
                        cloud_content = cf.read()
                st.markdown(cloud_content[:1000] + "\n\n*(预览前 1000 字符)*")
                st.download_button(
                    label=f"📥 下载此文件 ({os.path.basename(_path)})",
                    data=cloud_content,
                    file_name=os.path.basename(_path),
                    mime="text/markdown",
                    key="dl_history",
                )
            except Exception as e:
                st.error(f"读取失败: {e}")

    if st.button("🚀 撰写深度综述", type="primary"):
        if not review_topic:
            st.error("请先在左侧边栏输入或选择一个具体的综述主题！")
        elif not openrouter_api_key:
            st.error("请在侧边栏填入有效的 OpenRouter API Key！")
        elif local_df.empty and web_df.empty:
            st.warning("当前没有文献数据，请先点击“开始检索文献”或上传文件！")
        else:
            with st.spinner(f"AI 正在围绕【{review_topic}】撰写综述，可能需要 1-5 分钟（内容较长时会自动续写），请稍候..."):
                display_text, raw_text, base_msgs = generate_5000_words_review(
                    openrouter_api_key, openrouter_model, review_topic, local_df, web_df, enable_glossary_checkbox
                )
            if raw_text is None:
                st.error(display_text)
            else:
                ss["versions"] = [{"label": "V1 初稿", "text": display_text}]
                ss["base_messages"] = base_msgs
                ss["current_raw"] = raw_text
                ss["qa_history"] = []
                ss["extra_df"] = pd.DataFrame(columns=DOC_COLUMNS)
                ss["article_topic"] = review_topic

    if ss["versions"]:
        st.markdown("---")
        _notice = ss.get("save_notice")
        if _notice:
            getattr(st, _notice[0])(_notice[1])
        versions = ss["versions"]
        if len(versions) > 1:
            idx = st.selectbox(
                "查看版本：", options=range(len(versions)),
                index=len(versions) - 1,
                format_func=lambda i: versions[i]["label"],
            )
        else:
            idx = 0
        shown = versions[idx]
        st.markdown(shown["text"])

        safe_name = re.sub(r'[\\/:*?"<>|]', "_", ss["article_topic"])
        ver_tag = shown["label"].split()[0]
        st.download_button(
            label=f"📥 下载当前查看版本 ({ver_tag}) (.md)",
            data=shown["text"],
            file_name=f"{safe_name}_全网深度知识更新综述_{ver_tag}.md",
            mime="text/markdown",
        )

        st.markdown("---")
        st.subheader("💬 继续提问 / 提交修改意见")
        st.caption("“修改”模式会基于最新版生成完整新版本并保留历史版本；“提问”模式只回答问题、不改动文章：优先依据已有文献作答，若背景资料没有相关内容，会自动检索 PubMed 后再回答，并列出所参考的文献。")

        for msg in ss["qa_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        with st.form("followup_form", clear_on_submit=True):
            mode = st.radio("本次操作：", [REVISE_MODE, ASK_MODE], horizontal=True)
            user_text = st.text_area(
                "请输入内容：", height=120,
                placeholder="修改示例：把第四章的争论部分扩写，并增加关于老年患者的讨论；"
                            "提问示例：第六章表格中的肌松监测建议依据是什么？",
            )
            submitted = st.form_submit_button("提交", type="primary")

        if submitted:
            if not user_text.strip():
                st.warning("请输入修改意见或问题。")
            elif not openrouter_api_key:
                st.error("请在侧边栏填入有效的 OpenRouter API Key！")
            else:
                spinner_txt = (
                    "AI 正在修订全文，请稍候..." if mode == REVISE_MODE
                    else "AI 正在思考（若背景资料不足，将自动检索 PubMed 后再回答）..."
                )
                try:
                    with st.spinner(spinner_txt):
                        reply, used_model, finish, searched = follow_up_review(
                            openrouter_api_key, openrouter_model, ss["base_messages"],
                            ss["current_raw"], ss["qa_history"], user_text.strip(), mode,
                            enable_glossary_checkbox, ss["article_topic"],
                        )
                    if searched and not searched["df"].empty:  # 补充检索到的文献并入明细
                        _new = searched["df"].copy()
                        _new["Source"] = _new["Source"].astype(str) + " · 提问补充检索"
                        ss["extra_df"] = pd.concat([ss["extra_df"], _new], ignore_index=True)
                    if mode == REVISE_MODE:
                        n = len(ss["versions"]) + 1
                        short = re.sub(r"\s+", " ", user_text.strip())[:12]
                        new_text = wrap_article(reply, used_model, finish)
                        ss["versions"].append({
                            "label": f"V{n} {short}",
                            "text": new_text,
                        })
                        ss["current_raw"] = reply
                        ss["qa_history"].append({"role": "user", "content": f"✏️ 修改：{user_text.strip()}"})
                        ss["qa_history"].append({"role": "assistant", "content": f"已按要求生成第 {n} 版全文（见上方“查看版本”）。"})
                        save_review_to_cloud(ss["article_topic"], new_text, tag=f"V{n}")
                    else:
                        ss["qa_history"].append({"role": "user", "content": f"💬 提问：{user_text.strip()}"})
                        ss["qa_history"].append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 处理失败: {e}")

with tab2:
    st.subheader("已调用的文献数据明细")
    if not all_docs.empty:
        lit_md = build_literature_markdown(all_docs)
        st.caption(f"共 {len(all_docs)} 篇（点击标题或链接可直接访问）")
        st.markdown(lit_md)
        st.download_button(
            label="📥 下载文献清单 (.md)",
            data=f"# 文献清单：{review_topic or '未指定主题'}\n\n{lit_md}\n",
            file_name="文献清单.md",
            mime="text/markdown",
        )
    else:
        st.info("暂无文献数据。")

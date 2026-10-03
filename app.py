import datetime
import inspect
import io
import os
import re
import time
from collections import Counter

import pandas as pd
import plotly.express as px
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
# 1. 凭证加载（Streamlit Secrets -> 环境变量 -> .env）
# ==========================================
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):  # 跳过嵌套表
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # 本地没有 secrets.toml 时会抛异常，直接跳过

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

# 邮箱改为从环境变量读取，不再把个人邮箱写死在公开代码里
Entrez.email = NCBI_EMAIL or None
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
TAVILY_MAX_RESULTS = 20  # Tavily 接口上限为 20

# st.dataframe / st.plotly_chart 的撑满宽度参数在新旧版本中不同
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (0, 0)
STRETCH = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}

# 2. 权威临床与科研核心主题映射字典（共 33 项）
# ==========================================
# 药物自动识别：按 WHO 国际非专利药名（INN）的词干规则匹配，而不是写死具体药名。
# 例：-caine(局麻药)、-flurane(吸入麻醉药)、-zolam(苯二氮䓬)、-tomidine(α2激动剂)、
#     -fentanil/-fentanyl(阿片)、-ofol(丙泊酚类)、-curonium/-curium(肌松药)、-setron(止吐药) 等。
DRUG_STEMS = (
    "caine", "flurane", "zolam", "tomidine", "tomidate", "fentanil", "fentanyl",
    "ofol", "curonium", "curium", "ketamine", "setron", "phine", "morphone", "adol",
    "stigmine", "gammadex", "methasone", "rolac", "profen", "phrine", "drine",
    "olol", "pentin", "galin", "pental", "barbital", "trexone", "oxone",
)
DRUG_PATTERN = r"\b[a-z]*(?:" + "|".join(DRUG_STEMS) + r")\b"
DRUG_EXCLUDE = {"curium", "josephine", "delphine"}  # 元素“锔”及常见人名，非药物

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
    """从当前文献集（本地 + 全网 + PubMed）的标题与摘要中自动提取被讨论最多的药物。"""
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
        doc_freq.update(set(found))   # 每篇文献只计一次，避免长摘要刷屏
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
    """把 PMID/URL 转成可访问链接；本地文件返回 (None, 显示文字)。"""
    ident = str(identifier or "").strip()
    if ident.upper().startswith("PMID:"):
        pmid = ident.split(":", 1)[1].strip()
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", f"PMID:{pmid}"
    if ident.lower().startswith(("http://", "https://")):
        return ident, ident
    if ident.isdigit():  # CSV 中的纯数字 PMID
        return f"https://pubmed.ncbi.nlm.nih.gov/{ident}/", f"PMID:{ident}"
    return None, "本地上传文件（无在线链接）"


def _md_escape(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip() or "（无标题）"
    return re.sub(r"([\\`*_{}\[\]<>|])", r"\\\1", text)


def build_literature_markdown(df):
    """生成 Markdown 文献清单：标题（可点击）、年份、PMID/URL。"""
    lines = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        url, shown = doc_link(row.get("PMID/URL"))
        title = _md_escape(row.get("Title"))
        year = str(row.get("Year") or "N/A")
        head = f"[{title}]({url})" if url else title
        link_part = f"[{shown}]({url})" if url else shown
        lines.append(f"{i}. **{head}**  \n   年份：{year} ｜ 链接：{link_part}")
    return "\n\n".join(lines)


def clean_search_keyword(raw_text):
    if not raw_text:
        return "Anesthesia"
    text = re.sub(r"\.(pdf|docx|txt|csv)$", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"[_—–-]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:4]) if words else "Anesthesia"


def english_part(topic):
    """预设主题形如『中文 (English)』，检索时只用括号内的英文。"""
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
        raw = file.getvalue()  # 每次 rerun 都从头读，避免指针停在末尾读到空内容
        abstract = ""

        try:
            if ext == "pdf":
                import pypdf  # 按需导入（原代码漏了 import）

                reader = pypdf.PdfReader(io.BytesIO(raw))
                abstract = "".join(p.extract_text() or "" for p in reader.pages)[:4000]
            elif ext == "docx":
                import docx  # python-docx

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

            data.append({
                "PMID/URL": f"LOCAL_{idx}",
                "Title": filename,
                "Abstract": abstract or "未能解析出有效正文",
                "Year": "本地文件",
                "Source": "上传文件",
            })
        except Exception as e:
            st.error(f"解析文件 {filename} 失败: {e}")

    return pd.DataFrame(data, columns=DOC_COLUMNS)


# ==========================================
# 4. 检索引擎：Tavily 全网 + PubMed
#    出错时抛异常（异常不会被 st.cache_data 缓存，避免把失败结果缓存 1 小时）
# ==========================================
def tavily_request(api_key, query, max_results):
    """直接调用 Tavily REST 接口。出错时抛出带状态码和 Key 末4位的异常，便于核对。"""
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


@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(query, max_results):
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
    for article in papers.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]
            abs_list = art.get("Abstract", {}).get("AbstractText", [])
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year") or str(pub_date.get("MedlineDate", "N/A"))[:4])
            rows.append({
                "PMID/URL": f"PMID:{medline['PMID']}",
                "Title": str(art.get("ArticleTitle", "")),
                "Abstract": " ".join(str(x) for x in abs_list),
                "Year": year,
                "Source": "PubMed (数据库)",
            })
        except Exception:
            continue
    return pd.DataFrame(rows, columns=DOC_COLUMNS)


def fetch_web_and_pubmed_literature(raw_topic, tavily_key, max_results):
    """返回 (DataFrame, 提示信息列表)。"""
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
# 5. AI 综述生成引擎
# ==========================================
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
            last_error = f"模型 `{model}` 报错: {e}"
            time.sleep(1.5)  # 原代码未 import time，一旦报错 fallback 会直接崩溃

    raise RuntimeError(f"所有备选模型均未能正常生成响应。最后报错细节: {last_error}")


MAX_CONTINUATIONS = 4  # 单次最多自动续写次数


def call_with_continuation(client, primary_model, messages, temperature=0.3):
    """调用模型；若因长度上限被截断（finish_reason == "length"），自动让模型接着写并拼接。"""
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
        # 优先沿用同一个模型续写
        more, used_model, finish = call_openrouter_with_fallback(
            client, used_model, cont_messages, temperature
        )
        text += more
    return text, used_model, finish


def generate_5000_words_review(api_key, model_name, topic_keywords, local_df, web_df):
    try:
        clean_key = str(api_key).strip()
        if not clean_key:
            return "❌ 错误：请填入有效的 OpenRouter API Key！"

        client = OpenAI(api_key=clean_key, base_url="https://openrouter.ai/api/v1")

        local_text = ""
        if not local_df.empty:
            local_text += "\n=== [来源1：用户上传的本地文献资料] ===\n"
            for i, (_, row) in enumerate(local_df.head(20).iterrows(), 1):
                local_text += (
                    f"【本地文献 {i}】标题: {row.get('Title', '')}\n"
                    f"摘要/内容: {str(row.get('Abstract', ''))[:800]}...\n\n"
                )

        web_text = ""
        if not web_df.empty:
            web_text += "\n=== [来源2：全网（Tavily + PubMed）近5年高相关度文献] ===\n"
            for i, (_, row) in enumerate(web_df.head(30).iterrows(), 1):
                web_text += (
                    f"【全网/PubMed 文献 {i}】来源: {row.get('Source', '')} | "
                    f"标识: {row.get('PMID/URL', '')} | "
                    f"标题: {row.get('Title', '')} ({row.get('Year', '')})\n"
                    f"摘要: {str(row.get('Abstract', ''))[:500]}...\n\n"
                )

        combined_input = local_text + web_text

        hot_drugs = extract_hot_drugs(
            pd.concat([local_df, web_df], ignore_index=True), top_n=10
        )
        if not hot_drugs.empty:
            combined_input += "\n=== [自动统计：当前文献集中被讨论最多的药物（按涉及文献数）] ===\n"
            for _, r in hot_drugs.iterrows():
                combined_input += f"- {r['药物']}：{r['涉及文献数']} 篇文献（占 {r['占比']}）\n"
        start, today = get_past_5_years_range()

        prompt = f"""你是一名世界顶尖的麻醉学与围术期医学教授、权威学术期刊资深主编。
请基于我提供的【用户上传文献】以及【全网学术检索与 PubMed 数据库近 5 年检索到的文献】，围绕主题 **【{topic_keywords}】**，撰写一篇高度专业、结构严谨的**《麻醉学与围术期医学前沿知识更新与重难点热点深度综述》**，全文约 5000 字。

背景资料如下：
{combined_input}

---

### 📝 论文撰写规范与结构要求（必须完全输出完以下所有章节，绝对不能中断）：

#### 一、 摘要与关键词 (Abstract & Keywords)
* 撰写约 300 字的精炼摘要，涵盖研究背景、核心进展、核心争论及未来方向。

#### 二、 主题背景与近五年研究演进 (Introduction)
* 梳理该领域（【{topic_keywords}】）近 5 年（{start.year}-{today.year}年）的发展轨迹。

#### 三、 近五年核心学术突破与重大研究进展 (Major Breakthroughs)
* 分类阐述突破性研究成果（分子机制、临床试验 RCTs、新型药物或技术应用）。

#### 四、 当前热点领域的核心争论与未解决的临床困境 (Key Controversies)
* **重点章节**：深入剖析当前学术界存在的重大争论与分歧点（疗效争议、安全隐患、矛盾结论等）。

#### 五、 机制探讨与方法学瓶颈 (Mechanistic Insights)
* 讨论当前基础与临床研究面临的方法学瓶颈。

#### 六、 临床转化与围术期管理落地建议 (Clinical Implications)
* **必须包含一份完整 Markdown 表格**：
| 环节 | 建议措施 | 证据等级与推荐强度 |
（请填入具体的术前评估、麻醉诱导、血流动力学、肌松监测、镇痛策略等表格行）

#### 七、 总结与展望 (Conclusion)
* 总结全篇，提出未来 3-5 年最值得投入的科研切入点。

---

### ⚠️ 输出格式严格约束：
1. 必须完整输出完所有的 7 个章节，**特别是“六、临床转化表格”和“七、总结与展望”必须完整撰写完，绝对不能中途截断**！
2. 请使用标准 Markdown 格式，语言专业严谨。
3. 涉及药物的论述请优先围绕背景资料末尾“自动统计”中最热门的药物展开，并说明它们为何成为讨论热点。
4. 引用文献时只能使用上方【背景资料】中实际出现的文献（标注 PMID 或链接）；严禁编造文献、作者、数据或 PMID。资料不足之处请明确说明“现有资料未覆盖”。
"""

        messages = [
            {
                "role": "system",
                "content": "你是一名精通麻醉学前沿研究的权威期刊主编。你的输出必须完整，绝不中途截断段落或表格，且不得编造参考文献。",
            },
            {"role": "user", "content": prompt},
        ]

        result, used_model, finish = call_with_continuation(
            client, model_name, messages, temperature=0.3
        )
        return wrap_article(result, used_model, finish), result, messages
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


def follow_up_review(api_key, model_name, base_messages, current_article,
                     qa_history, user_text, mode):
    """多轮追问/修改。每轮只携带：原始资料 + 当前最新版全文 + 最近 6 条问答，避免上下文越滚越大。"""
    clean_key = str(api_key).strip()
    if not clean_key:
        raise RuntimeError("请填入有效的 OpenRouter API Key！")
    client = OpenAI(api_key=clean_key, base_url="https://openrouter.ai/api/v1")

    if mode == REVISE_MODE:
        directive = (
            "请根据下面的修改意见修订上述综述，**输出修订后的完整全文**"
            "（保持 7 个章节结构与 Markdown 表格，不得省略、缩写或中途截断；"
            "只能引用背景资料中出现的文献，严禁编造；直接输出文章，不要额外解释）。\n\n"
            f"修改意见：{user_text}"
        )
        temperature = 0.3
    else:
        directive = (
            "请基于上述综述与背景资料回答下面的问题，简明准确，不要重写全文；"
            "只能依据背景资料中的文献，严禁编造，资料不足请直说。\n\n"
            f"问题：{user_text}"
        )
        temperature = 0.4

    messages = (
        list(base_messages)
        + [{"role": "assistant", "content": current_article}]
        + list(qa_history[-6:])
        + [{"role": "user", "content": directive}]
    )
    return call_with_continuation(client, model_name, messages, temperature=temperature)


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
ss.setdefault("versions", [])        # [{"label":..., "text":...}]
ss.setdefault("base_messages", [])   # 初次生成时的 system + user(含全部资料)
ss.setdefault("current_raw", "")     # 最新版正文（不含页眉提示）
ss.setdefault("qa_history", [])      # 多轮提问/修改记录

st.title("💉 麻醉学全网文献热点追踪与深度综述生成系统")
st.markdown(
    "支持**上传本地文献** + **Tavily 全网学术搜索** + **PubMed 数据库**，"
    "一键撰写包含**核心争论、临床建议表格与未来方向**的学术综述。"
)

st.sidebar.header("🔍 模式设置与 Key 配置")
work_mode = st.sidebar.radio("选择工作模式：", options=[MODE_1, MODE_2])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 检索参数设置")
max_doc_count = st.sidebar.slider(
    "🔍 单次 PubMed / Tavily 检索最大文献数量（Tavily 最多 20 篇）：",
    min_value=10, max_value=100, value=50, step=10,
)

st.sidebar.markdown("---")
st.sidebar.header("🔑 API Key 设置")
# 安全：不再把后台 Secrets 预填进输入框。
# 否则 type="password" 的值仍会发送到访客浏览器，任何人都能取走你的密钥。
_AC = {"autocomplete": "off"} if "autocomplete" in inspect.signature(st.text_input).parameters else {}
tavily_input = st.sidebar.text_input(
    "Tavily 全网搜索 API Key",
    value="", type="password", **_AC,
    placeholder="已在后台配置，留空即使用" if ENV_TAVILY_KEY else "请输入",
)
openrouter_input = st.sidebar.text_input(
    "OpenRouter API Key",
    value="", type="password", **_AC,
    placeholder="已在后台配置，留空即使用" if ENV_OPENROUTER_KEY else "请输入",
)
_typed_tavily = normalize_key(tavily_input)
_typed_tavily_rejected = bool(_typed_tavily) and not _typed_tavily.startswith("tvly-")
if _typed_tavily_rejected:
    _typed_tavily = ""  # 多为浏览器自动填充的无关内容，忽略并使用后台 Key
tavily_api_key = _typed_tavily or ENV_TAVILY_KEY
openrouter_api_key = normalize_key(openrouter_input) or ENV_OPENROUTER_KEY
openrouter_model = st.sidebar.text_input("首选 AI 模型", value=DEFAULT_OPENROUTER_MODEL)

if not NCBI_EMAIL:
    st.sidebar.caption("ℹ️ 未配置 NCBI_EMAIL，建议在 Secrets 中设置（NCBI 要求提供联系邮箱）。")

local_df = pd.DataFrame(columns=DOC_COLUMNS)
search_topic = ""

if work_mode == MODE_1:
    st.sidebar.markdown("---")
    uploaded_files = st.sidebar.file_uploader(
        "上传本地文献（支持 PDF, Docx, TXT, CSV）：",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )
    custom_topic = st.sidebar.text_input(
        "补充或指定搜索的主题关键词（可选）：",
        value="", placeholder="例如：Dexmedetomidine neuroprotection",
    )
    if uploaded_files:
        local_df = parse_uploaded_files(uploaded_files)
        st.sidebar.success(f"已解析 {len(local_df)} 条本地文献记录！")

    if custom_topic.strip():
        search_topic = custom_topic.strip()
    elif uploaded_files:
        search_topic = clean_search_keyword(uploaded_files[0].name.rsplit(".", 1)[0])
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 自由指定或选择综述主题")
    selected_preset = st.sidebar.selectbox(
        "💡 从 33 项权威热点词库中选择：",
        options=[NO_PRESET] + list(CLINICAL_TOPICS.keys()),
    )
    default_text = "" if selected_preset == NO_PRESET else english_part(selected_preset)
    manual_input = st.sidebar.text_input(
        "✏️ 请确认或手动修改检索主题（建议使用英文）：",
        value=default_text, placeholder="例如：Remimazolam vs Propofol sedation",
    )
    search_topic = manual_input.strip()

# 检索改为点击按钮触发，避免每次改动输入框都消耗 Tavily/PubMed 额度
st.sidebar.markdown("---")
if st.sidebar.button("🔍 开始检索文献", type="primary", disabled=not search_topic):
    with st.spinner(f"正在全网（Tavily + PubMed）检索【{search_topic}】近 5 年文献..."):
        ss["web_df"], ss["search_msgs"] = fetch_web_and_pubmed_literature(
            search_topic, tavily_api_key, max_doc_count
        )
        ss["searched_topic"] = search_topic
if not search_topic:
    st.sidebar.caption("👆 请先输入或选择一个主题（模式 1 也可直接上传文件）。")

web_df = ss["web_df"]
review_topic = ss["searched_topic"] or search_topic

st.markdown("### 📊 当前数据准备状态")
c1, c2, c3 = st.columns(3)
c1.metric("解析本地文献数", f"{len(local_df)} 篇")
c2.metric("全网/PubMed 关联文献", f"{len(web_df)} 篇")
c3.metric("拟撰写综述主题", review_topic or "未指定")

for level, text in ss["search_msgs"]:
    getattr(st, level)(text)

st.markdown("---")

all_docs = pd.concat([local_df, web_df], ignore_index=True)

tab1, tab2, tab3 = st.tabs(
    ["📝 深度知识更新文章", "📚 数据库与全网文献明细", "🔥 课题热点图表"]
)

with tab1:
    st.subheader("📝 AI 深度撰写：知识更新综述")
    st.caption(
        "系统将融合本地文献与全网学术搜索（Tavily）以及 PubMed 数据库近 5 年的研究，"
        "探讨学术争论并输出临床转化表格。请先在侧边栏点击“开始检索文献”。"
        "初稿生成后，可在文章下方多轮提问或提交修改意见。"
    )

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
                    openrouter_api_key, openrouter_model, review_topic, local_df, web_df
                )
            if raw_text is None:
                st.error(display_text)
            else:
                ss["versions"] = [{"label": "V1 初稿", "text": display_text}]
                ss["base_messages"] = base_msgs
                ss["current_raw"] = raw_text
                ss["qa_history"] = []
                ss["article_topic"] = review_topic

    # 结果存入 session_state：否则点击下载按钮触发 rerun 后文章会消失
    if ss["versions"]:
        st.markdown("---")
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

        # ---------- 多轮提问与修改 ----------
        st.markdown("---")
        st.subheader("💬 继续提问 / 提交修改意见")
        st.caption("“修改”模式会基于最新版生成完整新版本并保留历史版本；“提问”模式只回答问题，不改动文章。")

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
                spinner_txt = "AI 正在修订全文，请稍候..." if mode == REVISE_MODE else "AI 正在思考..."
                try:
                    with st.spinner(spinner_txt):
                        reply, used_model, finish = follow_up_review(
                            openrouter_api_key, openrouter_model, ss["base_messages"],
                            ss["current_raw"], ss["qa_history"], user_text.strip(), mode,
                        )
                    if mode == REVISE_MODE:
                        n = len(ss["versions"]) + 1
                        short = re.sub(r"\s+", " ", user_text.strip())[:12]
                        ss["versions"].append({
                            "label": f"V{n} {short}",
                            "text": wrap_article(reply, used_model, finish),
                        })
                        ss["current_raw"] = reply
                        ss["qa_history"].append({"role": "user", "content": f"✏️ 修改：{user_text.strip()}"})
                        ss["qa_history"].append({"role": "assistant", "content": f"已按要求生成第 {n} 版全文（见上方“查看版本”）。"})
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

with tab3:
    st.subheader("当前文献集的 33 项热点映射分类")
    if not all_docs.empty:
        texts = all_docs["Title"].fillna("").astype(str) + " " + all_docs["Abstract"].fillna("").astype(str)
        topics_list = []
        for text in texts:
            topics_list.extend(extract_clinical_topics(text))

        if topics_list:
            df_counts = pd.DataFrame(Counter(topics_list).most_common(), columns=["热点主题", "匹配频次"])
            fig = px.bar(
                df_counts, x="匹配频次", y="热点主题", orientation="h",
                color="匹配频次", color_continuous_scale="Reds",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                height=max(400, len(df_counts) * 25),
            )
            st.plotly_chart(fig, **STRETCH)
        else:
            st.info("未发现匹配的预设热点。")

        st.markdown("---")
        st.subheader("💊 当前文献集中讨论最热门的药物（自动提取）")
        st.caption(
            "依据药物国际通用名（INN）词干规则，从标题与摘要中自动识别药物，"
            "统计被多少篇文献提及。未写死具体药名，结果随检索主题和文献集自动变化。"
        )
        hot_df = extract_hot_drugs(all_docs, top_n=15)
        if not hot_df.empty:
            fig2 = px.bar(
                hot_df, x="涉及文献数", y="药物", orientation="h",
                color="涉及文献数", color_continuous_scale="Blues",
            )
            fig2.update_layout(
                yaxis=dict(autorange="reversed"),
                height=max(350, len(hot_df) * 30),
            )
            st.plotly_chart(fig2, **STRETCH)
            st.dataframe(hot_df, hide_index=True, **STRETCH)
        else:
            st.info("当前文献集中未识别到药物名称（网页片段较短时可能出现，可增大检索数量）。")
    else:
        st.info("暂无文献数据。")

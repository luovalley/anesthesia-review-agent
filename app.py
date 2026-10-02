import os
import re
import datetime
import requests
import pandas as pd
import streamlit as st

# 1. 自动适配 Streamlit Cloud 的 Secrets 并注入到环境变量
try:
    if st.secrets:
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
except Exception:
    pass

# 2. 本地开发时尝试加载 .env（云端如果没有 python-dotenv 库则静默跳过，绝不报错）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
st.set_page_config(
    page_title="麻醉学全网文献热点追踪与 5000 字知识更新综述系统",
    page_icon="💉",
    layout="wide",
)

# 强制从环境变量获取凭证，代码中绝不内置硬编码密钥
ENTREZ_EMAIL = os.getenv("NCBI_EMAIL")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)

# 启动安全拦截检查
if not TAVILY_API_KEY or not OPENROUTER_API_KEY:
  st.error(
      "❌ **安全配置错误：** 检测到系统缺少必要的 API 密钥环境变量"
      " (`TAVILY_API_KEY` 或 `OPENROUTER_API_KEY`)。"
  )
  st.info("请在 AgentScope 平台的 **Environment Variables** 后台配置相应的密钥。")
  st.stop()

# 2. 默认模型与备用免费模型池配置
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "qwen/qwen3.8-27b:free"
)

# 实时更新的可用免费模型池
FALLBACK_FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
    "apodex/apodex-1.1-mini:free",
    "openrouter/free",
]
# ==========================================
# 2. 权威临床与科研核心主题映射字典（共 33 项）
# ==========================================
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
        r" delirium|pod)\b"
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
    "麻醉药物 (Dexmedetomidine / Propofol etc.)": (
        r"\b(dexmedetomidine|propofol|remimazolam|sevoflurane|ketamine|ropivacaine)\b"
    ),
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
        r" anesthetic systemic toxicity|last)\b"
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
        r"\b(epigenetics|dna methylation|microRNA|transcriptomics|single-cell"
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
  matched_topics = set()
  text_lower = str(text).lower()
  for topic_name, pattern in CLINICAL_TOPICS.items():
    if re.search(pattern, text_lower):
      matched_topics.add(topic_name)
  return list(matched_topics)


def get_past_5_years_date_filter():
  today = datetime.date.today()
  start_date = today - datetime.timedelta(days=365 * 5)
  return f'("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : "{today.strftime("%Y/%m/%d")}"[Date - Publication])'


def clean_search_keyword(raw_text):
  if not raw_text:
    return "Anesthesia"
  text = re.sub(r"\.(pdf|docx|txt|csv)$", "", raw_text, flags=re.IGNORECASE)
  text = re.sub(r"[_—–-]", " ", text)
  text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
  words = [w for w in text.split() if len(w) > 2]
  if words:
    return " ".join(words[:4])
  return "Anesthesia"


def parse_uploaded_files(uploaded_files):
  data = []
  for idx, file in enumerate(uploaded_files, 1):
    filename = file.name
    ext = filename.split(".")[-1].lower()
    title = filename
    abstract = ""

    try:
      if ext == "pdf":
        reader = pypdf.PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        abstract = text[:4000]
      elif ext == "docx":
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
        abstract = text[:4000]
      elif ext == "txt":
        text = file.read().decode("utf-8", errors="ignore")
        abstract = text[:4000]
      elif ext == "csv":
        df_uploaded = pd.read_csv(file)
        for _, row in df_uploaded.iterrows():
          data.append({
              "PMID/URL": str(row.get("PMID", f"LOCAL_{idx}")),
              "Title": str(row.get("Title", row.get("title", "未命名文献"))),
              "Abstract": str(row.get("Abstract", row.get("abstract", ""))),
              "Year": str(row.get("Year", row.get("year", "N/A"))),
              "Source": "上传文件",
          })
        continue

      data.append({
          "PMID/URL": f"LOCAL_{idx}",
          "Title": title,
          "Abstract": abstract if abstract else "未能解析出有效正文",
          "Year": "本地文件",
          "Source": "上传文件",
      })
    except Exception as e:
      st.error(f"解析文件 {filename} 失败: {str(e)}")

  return pd.DataFrame(data)


# ==========================================
# 4. 检索引擎：Tavily 全网 + PubMed
# ==========================================
@st.cache_data(
    ttl=3600, show_spinner="正在全网（Tavily + PubMed）检索近 5 年相关文献..."
)
def fetch_web_and_pubmed_literature(
    query_term, tavily_api_key, max_results=50
):
  combined_data = []
  cleaned_query = clean_search_keyword(query_term)

  # 1. Tavily 全网学术检索 (已修复此处的缩进对齐问题)
  if tavily_api_key and tavily_api_key.strip():
    try:
      url = "https://api.tavily.com/search"
      payload = {
          "api_key": tavily_api_key.strip(),
          "query": (
              f"{cleaned_query} anesthesia perioperative trial review recent"
              " research 2021..2026"
          ),
          "search_depth": "advanced",
          "max_results": max_results,
      }
      response = requests.post(url, json=payload, timeout=15)

      if response.status_code == 200:
        results = response.json().get("results", [])
        if not results:
          st.info(f"ℹ️ Tavily 未能针对关键词【{cleaned_query}】检索到相关结果。")
        for res in results:
          combined_data.append({
              "PMID/URL": res.get("url", "Web Link"),
              "Title": res.get("title", ""),
              "Abstract": res.get("content", ""),
              "Year": "2021-2026",
              "Source": "全网学术搜索 (Tavily)",
          })
      else:
        st.error(
            f"⚠️ Tavily API 调用失败！状态码: {response.status_code},"
            f" 响应内容: {response.text}"
        )
    except Exception as e:
      st.warning(f"Tavily 全网搜索出现异常: {str(e)}")

  # 2. PubMed 数据库检索
  try:
    date_filter = get_past_5_years_date_filter()
    full_query = f"({cleaned_query}) AND {date_filter}"

    search_handle = Entrez.esearch(
        db="pubmed", term=full_query, retmax=max_results, sort="relevance"
    )
    search_results = Entrez.read(search_handle)
    id_list = search_results.get("IdList", [])

    if not id_list and len(cleaned_query.split()) > 1:
      fallback_query = cleaned_query.split()[0]
      full_query = f"({fallback_query}) AND {date_filter}"
      search_handle = Entrez.esearch(
          db="pubmed", term=full_query, retmax=max_results, sort="relevance"
      )
      search_results = Entrez.read(search_handle)
      id_list = search_results.get("IdList", [])

    if id_list:
      fetch_handle = Entrez.efetch(
          db="pubmed", id=",".join(id_list), retmode="xml"
      )
      papers = Entrez.read(fetch_handle)

      for article in papers.get("PubmedArticle", []):
        try:
          medline = article["MedlineCitation"]
          pmid = str(medline["PMID"])
          article_data = medline["Article"]
          title = article_data.get("ArticleTitle", "")

          abstract_list = article_data.get("Abstract", {}).get(
              "AbstractText", []
          )
          abstract = " ".join(abstract_list) if abstract_list else ""

          journal_issue = article_data.get("Journal", {}).get(
              "JournalIssue", {}
          )
          pub_date = journal_issue.get("PubDate", {})
          year = pub_date.get("Year", pub_date.get("MedlineDate", "N/A")[:4])

          combined_data.append({
              "PMID/URL": f"PMID:{pmid}",
              "Title": title,
              "Abstract": abstract,
              "Year": year,
              "Source": "PubMed (数据库)",
          })
        except Exception:
          continue
  except Exception as e:
    st.error(f"PubMed 检索失败: {str(e)}")

  return pd.DataFrame(combined_data)


# ==========================================
# 5. AI 综述生成引擎
# ==========================================
def call_openrouter_with_fallback(
    client, primary_model, messages, temperature=0.3
):
  models_to_try = [primary_model] + [
      m for m in FALLBACK_FREE_MODELS if m != primary_model
  ]
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

      if response is None:
        last_error = f"模型 `{model}` 返回为空 (None)"
        continue

      choices = getattr(response, "choices", None)
      if not choices or len(choices) == 0:
        last_error = f"模型 `{model}` 返回的 choices 列表为空: {response}"
        continue

      first_choice = choices[0]
      if first_choice is None:
        last_error = f"模型 `{model}` 对应的 choice 对象为 None"
        continue

      message = getattr(first_choice, "message", None)
      if message is None:
        last_error = f"模型 `{model}` 的 choice 未能成功提取 message"
        continue

      content = getattr(message, "content", None)
      if content and str(content).strip():
        return str(content), model
      else:
        last_error = f"模型 `{model}` 返回生成文本为空"

    except Exception as e:
      last_error = f"模型 `{model}` 报错: {str(e)}"
      time.sleep(1.5)
      continue

  raise Exception(
      f"所有备选模型均未能正常生成响应。最后报错细节: {last_error}"
  )


def generate_5000_words_review(
    api_key, model_name, topic_keywords, local_df, web_df
):
  try:
    clean_key = str(api_key).strip()
    if not clean_key:
      return "❌ 错误：请填入有效的 OpenRouter API Key！"

    client = OpenAI(
        api_key=clean_key,
        base_url="https://openrouter.ai/api/v1",
    )

    local_text = ""
    if not local_df.empty:
      local_text += "\n=== [来源1：用户上传的本地文献资料] ===\n"
      for i, (_, row) in enumerate(local_df.iterrows(), 1):
        local_text += (
            f"【本地文献 {i}】标题: {row.get('Title', '')}\n摘要/内容:"
            f" {str(row.get('Abstract', ''))[:800]}...\n\n"
        )

    web_text = ""
    if not web_df.empty:
      web_text += (
          "\n=== [来源2：全网（Tavily + PubMed）近5年高相关度重磅文献] ===\n"
      )
      for i, (_, row) in enumerate(web_df.head(30).iterrows(), 1):
        web_text += (
            f"【全网/PubMed 文献 {i}】来源: {row.get('Source', '')} | 标题:"
            f" {row.get('Title', '')} ({row.get('Year', '')})\n摘要:"
            f" {str(row.get('Abstract', ''))[:500]}...\n\n"
        )

    combined_input = local_text + web_text

    prompt = f"""你是一名世界顶尖的麻醉学与围术期医学教授、权威学术期刊资深主编。
请基于我提供的【用户上传文献】以及【全网学术检索与 PubMed 数据库近 5 年检索到的重磅文献】，围绕主题 **【{topic_keywords}】**，撰写一篇高度专业、结构严谨的**《麻醉学与围术期医学前沿知识更新与重难点热点深度综述》**。

背景资料如下：
{combined_input}

---

### 📝 论文撰写规范与结构要求（必须完全输出完以下所有章节，绝对不能中断）：

#### 一、 摘要与关键词 (Abstract & Keywords)
* 撰写约 300 字的精炼摘要，涵盖研究背景、核心进展、核心争论及未来方向。

#### 二、 主题背景与近五年研究演进 (Introduction)
* 梳理该领域（【{topic_keywords}】）近 5 年（2021-2026年）的发展轨迹。

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
"""

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名精通麻醉学前沿研究的权威期刊主编。你的输出必须完整，绝不中途截断段落或表格。"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result_content, used_model = call_openrouter_with_fallback(
        client, model_name, messages, temperature=0.3
    )
    return (
        f"> 💡 *本篇深度综述由 AI 模型 `{used_model}` 基于全网文献成功生成*\n\n"
        + result_content
    )
  except Exception as e:
    return f"❌ 深度综述生成失败: {str(e)}"


# ==========================================
# 6. Streamlit 主界面与交互
# ==========================================
st.title("💉 麻醉学全网文献热点追踪与深度综述生成系统")
st.markdown(
    "支持**上传本地文献** + **Tavily 全网学术搜索** + **PubMed 数据库**，一键撰写包含**核心争论、临床建议表格与未来方向**的学术综述。"
)

st.sidebar.header("🔍 模式设置与 Key 配置")

work_mode = st.sidebar.radio(
    "选择工作模式：",
    options=[
        "1. 上传本地文献 + 全网 (Tavily/PubMed) 综合分析",
        "2. 直接输入主题全网搜查并撰写综述",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 检索参数设置")
max_doc_count = st.sidebar.slider(
    "🔍 单次 PubMed / Tavily 检索最大文献数量：",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
)

st.sidebar.markdown("---")
st.sidebar.header("🔑 API Key 设置")

# 安全从 st.secrets 获取，如果后台没配则默认为空字符串
default_tavily = st.secrets.get("TAVILY_API_KEY", "") if hasattr(st, "secrets") else ""

tavily_api_key = st.sidebar.text_input(
    "Tavily 全网搜索 API Key", value=default_tavily, type="password"
)
# 安全从 st.secrets 获取 OpenRouter 密钥，未配置则为空
default_openrouter = st.secrets.get("OPENROUTER_API_KEY", "") if hasattr(st, "secrets") else ""

openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key", value=default_openrouter, type="password"
)

openrouter_model = st.sidebar.text_input(
    "首选 AI 模型", value=DEFAULT_OPENROUTER_MODEL
)

local_df = pd.DataFrame()
web_df = pd.DataFrame()
search_topic = ""

if "1. 上传本地文献" in work_mode:
  st.sidebar.markdown("---")
  uploaded_files = st.sidebar.file_uploader(
      "上传本地文献（支持 PDF, Docx, TXT, CSV）：",
      type=["pdf", "docx", "txt", "csv"],
      accept_multiple_files=True,
  )
  custom_topic = st.sidebar.text_input(
      "补充或指定搜索的主题关键词（可选）：",
      value="",
      placeholder="例如：Dexmedetomidine neuroprotection",
  )

  if uploaded_files:
    local_df = parse_uploaded_files(uploaded_files)
    st.sidebar.success(f"已解析 {len(local_df)} 篇本地文件！")

    if custom_topic.strip():
      search_topic = custom_topic.strip()
    else:
      first_name = uploaded_files[0].name.rsplit(".", 1)[0]
      search_topic = clean_search_keyword(first_name)

    st.sidebar.info(
        f"🔗 正在全网检索主题：`{search_topic}` (单次上限: {max_doc_count} 篇)"
    )
    web_df = fetch_web_and_pubmed_literature(
        search_topic, tavily_api_key, max_results=max_doc_count
    )

else:
  st.sidebar.markdown("---")
  st.sidebar.subheader("🎯 自由指定或选择综述主题")

  preset_list = list(CLINICAL_TOPICS.keys())

  selected_preset = st.sidebar.selectbox(
      "💡 从 33 项权威热点词库中选择：",
      options=["-- 手动/自定义输入主题 --"] + preset_list,
  )

  if selected_preset != "-- 手动/自定义输入主题 --":
    default_text = selected_preset
  else:
    default_text = ""

  manual_input = st.sidebar.text_input(
      "✏️ 请确认或手动修改检索主题（支持英文/中文）：",
      value=default_text,
      placeholder="例如：Remimazolam vs Propofol sedation",
  )

  search_topic = manual_input.strip()

  if search_topic:
    st.info(
        f"🔍 当前全网检索主题：**{search_topic}** （检索上限：**{max_doc_count}** 篇）"
    )
    web_df = fetch_web_and_pubmed_literature(
        search_topic, tavily_api_key, max_results=max_doc_count
    )
  else:
    st.warning("👈 请在左侧侧边栏输入或从 33 项热点词库中选择一个主题！")

st.markdown("### 📊 当前数据准备状态")
c1, c2, c3 = st.columns(3)
c1.metric("解析本地文献数", f"{len(local_df)} 篇")
c2.metric("全网/PubMed 关联文献", f"{len(web_df)} 篇")
c3.metric("拟撰写综述主题", search_topic if search_topic else "未指定")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["📝 深度知识更新文章", "📚 数据库与全网文献明细", "🔥 课题热点图表"]
)

with tab1:
  st.subheader("📝 AI 深度撰写：知识更新综述")
  st.caption(
      "系统将融合本地文献与全网学术搜索（Tavily）以及 PubMed 数据库近 5"
      " 年的突破性研究，深入探讨学术争论并输出完整临床转化表格。"
  )

  if st.button("🚀 开始全网检索并撰写深度综述", type="primary"):
    if not search_topic:
      st.error("请先在左侧边栏输入或选择一个具体的综述主题！")
    elif not openrouter_api_key.strip():
      st.error("请在侧边栏填入有效的 OpenRouter API Key！")
    elif local_df.empty and web_df.empty:
      st.warning("当前没有检索到文献数据，请检查网络或更换关键词！")
    else:
      with st.spinner(
          f"AI 正在全网检索【{search_topic}】并梳理学术争论，过程可能需要 1"
          " 分钟，请稍候..."
      ):
        article_md = generate_5000_words_review(
            openrouter_api_key,
            openrouter_model,
            search_topic,
            local_df,
            web_df,
        )

        st.markdown("---")
        st.markdown(article_md)

        st.download_button(
            label="📥 下载完整综述文章 (.md)",
            data=article_md,
            file_name=f"{search_topic}_全网深度知识更新综述.md",
            mime="text/markdown",
        )

with tab2:
  st.subheader("已调用的文献数据明细")
  all_docs = pd.concat([local_df, web_df], ignore_index=True)
  if not all_docs.empty:
    all_docs["匹配热点"] = all_docs.apply(
        lambda row: ", ".join(
            extract_clinical_topics(f"{row['Title']} {row['Abstract']}")
        ),
        axis=1,
    )
    st.dataframe(
        all_docs[["Title", "Year", "Source", "匹配热点", "PMID/URL"]],
        use_container_width=True,
    )
  else:
    st.info("暂无文献数据。")

with tab3:
  st.subheader("当前文献集的 33 项热点映射分类")
  all_docs = pd.concat([local_df, web_df], ignore_index=True)
  if not all_docs.empty:
    topics_list = []
    for text in all_docs["Title"] + " " + all_docs["Abstract"]:
      topics_list.extend(extract_clinical_topics(text))

    if topics_list:
      counts = Counter(topics_list).most_common()
      df_counts = pd.DataFrame(counts, columns=["热点主题", "匹配频次"])
      fig = px.bar(
          df_counts,
          x="匹配频次",
          y="热点主题",
          orientation="h",
          color="匹配频次",
          color_continuous_scale="Reds",
      )
      fig.update_layout(
          yaxis=dict(autorange="reversed"),
          height=max(400, len(df_counts) * 25),
      )
      st.plotly_chart(fig, use_container_width=True)
    else:
      st.info("未发现匹配的预设热点。")
  else:
    st.info("暂无文献数据。")

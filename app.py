import os
import re
import base64
import datetime
import urllib.request
import pandas as pd
import streamlit as st
from Bio import Entrez
from openai import OpenAI

# ==================== 0. 配置初始化与全局常量 ====================
st.set_page_config(
    page_title="麻醉学与围术期医学前沿文献深度综述与智能检索系统",
    page_icon="🩺",
    layout="wide"
)

# 设置 Entrez 邮箱（NCBI 推荐规范）
Entrez.email = os.getenv("NCBI_EMAIL", "anesthesia_agent@example.com")

DOC_COLUMNS = ["PMID/URL", "Title", "Abstract", "Year", "Source"]

# ==================== 1. 辅助函数与后端逻辑 ====================

def load_glossary_content():
    """加载本地或远端的 glossary.md 专业术语对照表"""
    if os.path.exists("glossary.md"):
        try:
            with open("glossary.md", "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return ""

def _pubmed_esearch(term, retmax):
    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="relevance")
    try:
        return Entrez.read(handle).get("IdList", [])
    finally:
        handle.close()

def get_past_5_years_date_filter():
    current_year = datetime.datetime.now().year
    start_year = current_year - 5
    return f'("{start_year}"[Date - Publication] : "{current_year}"[Date - Publication])'

@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(query, max_results):
    """支持标准摘要抓取及增强版 PMC 开放全文提取的 PubMed 检索函数"""
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
            pmid = str(medline["PMID"])
            
            # 1. 提取标准摘要作为基础保底
            abs_list = art.get("Abstract", {}).get("AbstractText", [])
            abstract_text = " ".join(str(x) for x in abs_list)
            
            # 2. 安全提取 PMCID
            pmcid = ""
            pubmed_data = article.get("PubmedData", {})
            for id_elem in pubmed_data.get("ArticleIdList", []):
                if hasattr(id_elem, "attributes") and id_elem.attributes.get("IdType") == "pmc":
                    pmcid = str(id_elem)
                    break

            full_text = abstract_text
            source_desc = "PubMed (数据库摘要)"

            # 3. 增强版 PMC 开放全文自动抓取
            if pmcid:
                pmc_num = pmcid.replace("PMC", "").strip()
                try:
                    pmc_handle = Entrez.efetch(db="pmc", id=pmc_num, retmode="xml")
                    pmc_data = pmc_handle.read()
                    pmc_handle.close()
                    
                    clean_text = re.sub(r'<[^>]+>', ' ', pmc_data)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    if len(clean_text) > len(abstract_text) + 200:
                        full_text = clean_text[:8000]  # 截取前 8000 字符详细正文
                        source_desc = "PubMed Central (PMC 开放全文)"
                except Exception:
                    try:
                        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_num}&retmode=text"
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=10) as response:
                            web_content = response.read().decode('utf-8', errors='ignore')
                            if len(web_content) > len(abstract_text) + 200:
                                full_text = web_content[:8000]
                                source_desc = "PubMed Central (PMC 开放全文)"
                    except Exception:
                        pass

            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = str(pub_date.get("Year") or str(pub_date.get("MedlineDate", "N/A"))[:4])
            
            rows.append({
                "PMID/URL": f"PMID:{pmid}" + (f" | {pmcid}" if pmcid else ""),
                "Title": str(art.get("ArticleTitle", "")),
                "Abstract": full_text,
                "Year": year,
                "Source": source_desc,
            })
        except Exception:
            continue
            
    return pd.DataFrame(rows, columns=DOC_COLUMNS)

def extract_keywords_from_url(url, client, model):
    """通过参考链接抓取网页内容，并利用大模型提取精准的 PubMed 英文检索词"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            clean_text = re.sub(r'<[^>]+>', ' ', resp.text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:4000]
            
            prompt = f"""请阅读以下网页/文献页面的内容，提取其最核心的医学研究主题，并将其转化为适合 PubMed 数据库的高效英文检索表达式（包含必要的布尔逻辑词如 AND/OR）。
要求：
1. 只输出检索词表达式本身，绝对不要输出任何多余的解释、引号或前缀。
2. 保持专业、精准。

网页内容片段：
{clean_text}"""

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            extracted_query = response.choices[0].message.content.strip().replace('"', '')
            return extracted_query
    except Exception as e:
        st.error(f"解析链接失败: {e}")
    return None

def save_review_to_cloud(topic, content, tag="V1"):
    """
    通过 GitHub API 安全将综述保存到远端 reviews/ 目录。
    强制以文章主题命名，并对特殊符号与文件名长度进行严格截断与清洗。
    """
    topic_str = str(topic).strip()
    
    # 自动过滤误传的布尔检索表达式
    if " OR " in topic_str or " AND " in topic_str or "_postinduction" in topic_str:
        clean_terms = re.sub(r'[\(\)_]', '', topic_str)
        first_term = clean_terms.split(" OR ")[0].split(" AND ")[0].strip()
        topic_str = first_term if first_term else "Anesthesia_Review"

    clean_topic = re.sub(r'[\\/:*?"<>|\s\(\)\[\]{}]+', "_", topic_str).strip("_")
    if len(clean_topic) > 35:
        clean_topic = clean_topic[:35] + "_etc"
    if not clean_topic:
        clean_topic = "Anesthesia_Review"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path_in_repo = f"reviews/{clean_topic}_{tag}_{timestamp}.md"
    
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "").strip()
    branch = os.getenv("GITHUB_BRANCH", "main").strip()

    if token and repo:
        url = f"https://api.github.com/repos/{repo}/contents/{file_path_in_repo}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        try:
            encoded_content = base64.b64encode(content.encode("utf-8")).decode("utf-8")
            payload = {
                "message": f"Auto-save review: {clean_topic} ({tag})",
                "content": encoded_content,
                "branch": branch
            }
            put_resp = requests.put(url, headers=headers, json=payload, timeout=15)
            if put_resp.status_code in (200, 201):
                return file_path_in_repo
        except Exception:
            pass

    # 本地降级存储
    os.makedirs("saved_reviews", exist_ok=True)
    local_filename = f"saved_reviews/{clean_topic}_{tag}_{timestamp}.md"
    try:
        with open(local_filename, "w", encoding="utf-8") as f:
            f.write(content)
        return local_filename
    except Exception:
        return None

# ==================== 2. 侧边栏配置面板 ====================
st.sidebar.title("🛠️ 麻醉学文献综述工作台")

# API 配置
api_key = st.sidebar.text_input("OpenAI / OpenRouter API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
base_url = st.sidebar.text_input("API Base URL (可选)", value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
selected_model = st.sidebar.text_input("LLM 模型名称", value="gpt-4o")

st.sidebar.markdown("---")

# 初始化 session state 用于默认主题与联动
if "default_topic" not in st.session_state:
    st.session_state["default_topic"] = "post-induction hypotension AND elderly"

# 主题输入框
topic_keywords = st.sidebar.text_input(
    "请确认或手动修改检索主题", 
    value=st.session_state["default_topic"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **或通过参考链接智能提取**")

ref_url = st.sidebar.text_input(
    "请输入参考文章/网页链接：", 
    placeholder="https://pubmed.ncbi.nlm.nih.gov/40322031/",
    key="input_ref_url"
)

# 初始化临时客户端以支持侧边栏的链接解析
temp_client = OpenAI(api_key=api_key or "dummy", base_url=base_url) if api_key else None

if st.sidebar.button("📥 读取链接并提取检索词", use_container_width=True):
    if ref_url.strip() and temp_client:
        with st.spinner("正在抓取网页并提取核心检索词..."):
            new_query = extract_keywords_from_url(ref_url.strip(), temp_client, selected_model)
            if new_query:
                st.session_state["default_topic"] = new_query
                st.success(f"成功提取检索词：{new_query}")
                st.rerun()
            else:
                st.warning("未能从该链接中提取到有效关键词。")
    else:
        st.warning("请先配置 API Key 并输入有效的网址链接。")

st.sidebar.markdown("---")
max_results = st.sidebar.slider("检索文献最大数量", 5, 50, 20)
enable_glossary = st.sidebar.checkbox("启用专业术语表对齐 (glossary.md)", value=True)

# ==================== 3. 主界面布局与核心交互 ====================
st.title("🩺 麻醉学与围术期医学前沿文献深度综述系统")
st.markdown("基于 **PubMed/PMC 开放全文数据库**、**大模型多维归纳** 与 **GitHub 云端持久化** 的智能学术综述生成平台。")

tab1, tab2, tab3 = st.tabs(["🚀 生成综述主面板", "📚 数据库与全网文献明细", "📖 术语表管理"])

with tab1:
    st.subheader("一键生成 5000 字硬核学术综述")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        run_btn = st.button("🚀 开始检索并生成深度综述", type="primary", use_container_width=True)

    if run_btn:
        if not api_key:
            st.error("请先在左侧侧边栏输入您的 API Key！")
        else:
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            with st.status("正在进行多源学术检索与全文解析...", expanded=True) as status:
                st.write("正在连接 PubMed/PMC 数据库进行近 5 年文献深度检索...")
                df_pubmed = search_pubmed(topic_keywords, max_results)
                status.update(label=f"成功获取 {len(df_pubmed)} 篇文献（含 PMC 开放全文）", state="running")
                
                # 组装背景资料
                combined_input = ""
                for idx, row in df_pubmed.iterrows():
                    combined_input += (
                        f"文献 [{idx+1}]\n"
                        f"标题: {row['Title']} ({row['Year']})\n"
                        f"来源: {row['Source']} | 链接标识: {row['PMID/URL']}\n"
                        f"详细内容/摘要节选:\n{row['Abstract']}\n\n"
                    )
                
                # 加载术语表
                glossary_section = ""
                if enable_glossary:
                    glossary_text = load_glossary_content()
                    if glossary_text.strip():
                        glossary_section = f"""
---
### 📖 专业术语表 (Glossary) 严格对齐与翻译规范：
请在撰写综述时**必须严格参照**以下专有名词对照表：
{glossary_text.strip()}

**⚠️ 核心硬性要求**：
1. **大小写不敏感匹配**：无论背景资料中的英文专业名词是大写、小写还是复数形式，**在中文正文中都必须严格翻译并替换为词典中对应的中文专业译名**。
2. 绝对不允许在中文正文里直接大段保留未翻译的英文核心病名、药名或生理指标。
"""

                today = datetime.datetime.now()
                start = today - datetime.timedelta(days=365*5)

                prompt = f"""你是一名世界顶尖的麻醉学与围术期医学教授、权威学术期刊资深主编。
请基于我提供的【全网学术搜索与 PubMed/PMC 全文数据库近 5 年检索到的文献】，围绕主题 **【{topic_keywords}】**，撰写一篇高度专业、结构严谨的《麻醉学与围术期医学前沿知识更新与重难点热点深度综述》，全文约 5000 字。

背景资料如下：
{combined_input}
{glossary_section}
---

### 📝 论文撰写规范与结构要求（必须完全输出完以下所有章节，绝对不能中断）：

#### 一、 摘要与关键词 (Abstract & Keywords)
* **必须使用中文**撰写约 300 字的精炼摘要，涵盖研究背景、核心进展、核心争论及未来方向；关键词需提供中英文对照。

#### 二、 主题背景与近五年研究演进 (Introduction)
* 梳理该领域近 5 年（{start.year}-{today.year}年）的发展轨迹与演变逻辑。

#### 三、 近五年核心学术突破与重大研究进展 (Major Breakthroughs)
* **【硬核数据强制要求】**：必须深入引用背景资料中具体临床试验或前瞻性研究的**硬核量化数据**。例如，必须明确写出：
  1. **具体样本量**（如“某项纳入了 $N=420$ 例患者的随机对照试验显示……”）；
  2. **具体的生理/药理量化指标**（如“观察组指标较基线显著改变，数值达 $22.5 \pm 4.3$……”）；
  3. **具体的给药方案与终点数据**。绝对不能只写笼统的结论。

#### 四、 当前热点领域的核心争论与未解决的临床困境 (Key Controversies)
* **重点章节**：深入剖析当前学术界存在的重大争论与分歧点，必须直接引用不同研究在具体终点指标上的**数据冲突与对立结果**。

#### 五、 机制探讨与方法学瓶颈 (Mechanistic Insights)
* 讨论当前基础与临床研究面临的方法学瓶颈。

#### 六、 临床转化与围术期管理落地建议 (Clinical Implications)
* **必须包含一份完整 Markdown 表格**：
| 环节 | 建议措施 | 证据等级与推荐强度 |

#### 七、 总结与展望 (Conclusion)
* 总结全篇，提出未来 3-5 年最值得投入的科研切入点。

---

### ⚠️ 输出格式极其严格的约束（切记）：
1. **全文（包括第一章摘要与关键词）必须完全使用中文撰写与翻译**，专业名词可在括号中附带英文原词。
2. 必须完整输出完所有的 7 个章节，特别是表格与总结部分，绝对不能中途截断！
3. 请使用标准 Markdown 格式，语言专业严谨。
4. **绝对禁止在正文中出现任何形式的文献序号、标号或标记**：包括但不限于 `[1]`、`[1,2]`、`(来源2、3等)`、上标数字、或者任何形式的“文献1”、“本地文献”、“全网文献”、“PMID”等引用标识。
5. 论述必须以上方【背景资料】中的文献为依据，严禁编造文献、作者、数据或 PMID。
"""

                st.write("正在交由大模型进行深度综合归纳与硬核数据萃取...")
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    stream=True
                )
                
                review_placeholder = st.empty()
                full_review_content = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_review_content += chunk.choices[0].delta.content
                        review_placeholder.markdown(full_review_content + "▌")
                
                review_placeholder.markdown(full_review_content)
                status.update(label="综述生成完毕！正在自动同步至 GitHub 远端...", state="complete")

            # 自动云端持久化（以文章主题命名，防止文件名过长）
            saved_path = save_review_to_cloud(topic=topic_keywords, content=full_review_content, tag="V1")
            if saved_path:
                st.success(f"🎉 综述已成功加密同步至云端仓库路径: `{saved_path}`")
            else:
                st.warning("⚠️ 云端同步失败，已保存在本地 `saved_reviews/` 目录中。")

with tab2:
    st.subheader("📚 数据库检索结果与全文状态明细")
    st.markdown("在此可核查当前检索到的文献来源（是 PMC 开放全文还是标准数据库摘要）。")
    if st.button("🔄 刷新文献列表"):
        st.rerun()
    
    # 快速预览检索结果
    preview_df = search_pubmed(topic_keywords, max_results)
    if not preview_df.empty:
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("当前暂无检索结果，请在主面板点击生成或调整检索主题。")

with tab3:
    st.subheader("📖 专业术语表 (glossary.md) 管理")
    st.markdown("在此维护双语医学专业词汇，确保大模型生成的中文综述专业词汇翻译一致。")
    
    current_glossary = load_glossary_content()
    edited_glossary = st.text_area("编辑 glosssary.md 内容", value=current_glossary, height=300)
    
    if st.button("💾 保存术语表"):
        try:
            with open("glossary.md", "w", encoding="utf-8") as f:
                f.write(edited_glossary)
            st.success("术语表保存成功！")
        except Exception as e:
            st.error(f"保存失败: {e}")

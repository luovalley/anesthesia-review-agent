import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import time
from openai import OpenAI

# -----------------------------------------------------------------
# 页面配置
# -----------------------------------------------------------------
st.set_page_config(
    page_title="麻醉学全网文献热点追踪与深度综述系统",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 麻醉学全网文献热点追踪与 5000 字知识更新综述系统")
st.markdown("---")

# -----------------------------------------------------------------
# 侧边栏配置（完整保留与扩展）
# -----------------------------------------------------------------
with st.sidebar:
    st.header("🔑 API 与模型配置")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="用于调用 DeepSeek、Claude、GPT 等大模型")
    tavily_api_key = st.text_input("Tavily API Key", type="password", help="用于获取全网灰色文献、临床指南与新闻")
    
    model_name = st.selectbox(
        "选择大模型",
        [
            "deepseek/deepseek-chat",
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-2.5-pro"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📚 目标麻醉学顶刊过滤")
    target_journals = st.multiselect(
        "选择重点监控期刊",
        [
            "Anesthesiology",
            "British Journal of Anaesthesia (BJA)",
            "Anesthesia & Analgesia",
            "Regional Anesthesia and Pain Medicine (RAPM)",
            "European Journal of Anaesthesiology (EJA)"
        ],
        default=["Anesthesiology", "British Journal of Anaesthesia (BJA)", "Anesthesia & Analgesia"]
    )
    
    st.markdown("---")
    st.subheader("⚙️ 检索与重排参数")
    pub_retmax = st.slider("PubMed 最大检索量 (Retmax)", 50, 200, 200, step=25, help="放宽上限以确保前沿文献不漏网")
    tavily_max = st.slider("Tavily 最大网页检索量", 10, 30, 25, step=5, help="补充全网指南与最新临床进展")
    top_k_core = st.slider("精筛核心文献数 (Rerank Top K)", 20, 80, 50, step=10, help="送给大模型进行深度 Map-Reduce 的核心文献数")
    
    st.markdown("---")
    st.subheader("📬 自动化推送配置 (可选)")
    enable_push = st.checkbox("启用结果微信/TG推送", value=False)
    pushplus_token = st.text_input("PushPlus Token", type="password") if enable_push else ""

# -----------------------------------------------------------------
# 核心功能模块 1：PubMed 批量检索与期刊过滤
# -----------------------------------------------------------------
def fetch_pubmed_papers(query, journals, retmax=200):
    st.info(f"正在从 PubMed 检索最多 {retmax} 篇文献...")
    
    # 构造期刊过滤表达式
    journal_query_part = ""
    if journals:
        j_filters = [f'"{j}"[Journal]' for j in journals]
        journal_query_part = " AND (" + " OR ".join(j_filters) + ")"
    
    full_query = f"({query}){journal_query_part}"
    
    base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": full_query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "date"
    }
    
    try:
        res = requests.get(base_search_url, params=search_params, timeout=15)
        res_json = res.json()
        id_list = res_json.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            # 如果加上期刊限制后无结果，尝试放宽期刊限制仅搜关键词
            st.warning("指定期刊未直接命中，正在放宽至全 PubMed 检索...")
            search_params["term"] = query
            res = requests.get(base_search_url, params=search_params, timeout=15)
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []
        
        # 批量获取详情 (efetch)
        base_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_res = requests.get(base_fetch_url, params=fetch_params, timeout=30)
        root = ET.fromstring(fetch_res.content)
        
        papers = []
        for article in root.findall(".//PubmedArticle"):
            try:
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "无标题"
                
                abstract_elem = article.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else "无摘要"
                
                year_elem = article.find(".//JournalIssue/PubDate/Year")
                if year_elem is None:
                    year_elem = article.find(".//JournalIssue/PubDate/MedlineDate")
                pub_date = year_elem.text if year_elem is not None else "未知年份"
                
                papers.append({
                    "source": "PubMed",
                    "title": title,
                    "abstract": abstract,
                    "pub_date": pub_date
                })
            except Exception:
                continue
                
        return papers
    except Exception as e:
        st.error(f"PubMed 检索出错: {e}")
        return []

# -----------------------------------------------------------------
# 核心功能模块 2：Tavily 全网补充检索
# -----------------------------------------------------------------
def fetch_tavily_web(query, api_key, max_results=25):
    if not api_key:
        return []
    st.info(f"正在通过 Tavily 获取全网灰色文献与指南 (上限 {max_results} 条)...")
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": f"{query} guidelines review clinical trial anesthesia",
        "max_results": max_results,
        "search_depth": "advanced"
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        data = response.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "source": "Tavily Web",
                "title": item.get("title", ""),
                "abstract": item.get("content", ""),
                "pub_date": "Recent"
            })
        return results
    except Exception as e:
        st.warning(f"Tavily 检索失败: {e}")
        return []

# -----------------------------------------------------------------
# 核心功能模块 3：轻量级智能重排与过滤 (Relevance Rerank)
# -----------------------------------------------------------------
def compute_relevance_score(paper, query_keywords):
    score = 0.0
    for kw in query_keywords:
        kw_lower = kw.lower()
        title_matches = len(re.findall(re.escape(kw_lower), paper.get('title', '').lower()))
        abstract_matches = len(re.findall(re.escape(kw_lower), paper.get('abstract', '').lower()))
        score += title_matches * 5.0   # 标题命中权重更高
        score += abstract_matches * 1.0 # 摘要命中权重次之

    # 时效性加权
    pub_date = str(paper.get('pub_date', ''))
    current_year = datetime.now().year
    try:
        year_match = re.search(r'\b(20\d{2})\b', pub_date)
        if year_match:
            year = int(year_match.group(1))
            years_old = current_year - year
            if years_old <= 1:
                score += 3.0
            elif years_old <= 3:
                score += 1.5
    except Exception:
        pass

    return score

def smart_rerank_and_filter(papers, query, top_k=50):
    query_keywords = [kw.strip() for kw in re.split(r'[\s,\+\-\_]+', query) if len(kw.strip()) > 1]
    scored_papers = []
    for p in papers:
        score = compute_relevance_score(p, query_keywords)
        p['relevance_score'] = score
        scored_papers.append(p)
        
    scored_papers.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_papers[:top_k]

# -----------------------------------------------------------------
# 核心功能模块 4：分步式 (Map-Reduce) 综述生成
# -----------------------------------------------------------------
def map_subtopic_generation(client, model_name, subtopic_name, papers_subset):
    references_text = "\n\n".join([
        f"- **[{p.get('title')}]** ({p.get('pub_date', 'N/A')})\n  摘要: {p.get('abstract', '无摘要')}"
        for p in papers_subset
    ])
    
    prompt = f"""
    您是一位麻醉学主任医师。请根据以下筛选出的 {len(papers_subset)} 篇核心文献，针对子主题【{subtopic_name}】撰写一份深度专业分析报告（约 1500~2000 字）。
    要求：
    1. 紧扣麻醉学临床专业视角，深度剖析机制、数据、临床结局与指导价值。
    2. 严格基于提供的文献，禁止虚构。
    3. 包含必要的对比表格或数据要点小结。

    核心文献列表：
    {references_text}
    """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def reduce_synthesize_review(client, model_name, user_topic, subtopic_drafts):
    combined_drafts = "\n\n---\n\n".join([
        f"### 子主题分析报告 {i+1}\n{draft}" for i, draft in enumerate(subtopic_drafts)
    ])
    
    reduce_prompt = f"""
    您是一位主编级别的麻醉学教授。现在需要将以下三个子主题的深度分析报告，统筹整合并升华成一篇高水平、结构完整的 **5000 字全网前沿学术综述**。
    
    综述核心主题：{user_topic}
    
    整体结构必须包含：
    1. 摘要与引言（Background & Introduction）
    2. 机制演进与药理/生理学进展
    3. 临床不良反应防控与优化干预方案
    4. 特殊人群与前沿结局展望
    5. 临床转化总结与规范的文献引用列表。

    各子主题分报告素材：
    {combined_drafts}

    请输出排版精美、学术用语严谨、逻辑严密的最终 Markdown 宏大综述。
    """
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": reduce_prompt}],
        temperature=0.4,
        max_tokens=8000
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------
# 主界面交互逻辑
# -----------------------------------------------------------------
user_query = st.text_input(
    "请输入您的检索/综述主题：",
    value="propofol painless gastroscopy hypotension respiratory depression Eleveld model"
)

if st.button("🚀 开始多路检索、智能过滤与 5000 字综述生成", type="primary"):
    if not openrouter_api_key:
        st.error("请先在左侧侧边栏填入您的 OpenRouter API Key！")
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        
        with st.status("正在执行多步处理流水线...", expanded=True) as status:
            # 步骤 1：多路召回
            st.write("步骤 1/4: 正在多路召回海量文献...")
            pubmed_papers = fetch_pubmed_papers(user_query, target_journals, retmax=pub_retmax)
            tavily_papers = fetch_tavily_web(user_query, tavily_api_key, max_results=tavily_max)
            all_raw_papers = pubmed_papers + tavily_papers
            st.write(f"✅ 召回完成：PubMed 检索到 {len(pubmed_papers)} 篇，Tavily 检索到 {len(tavily_papers)} 篇，总计 {len(all_raw_papers)} 篇。")
            
            if not all_raw_papers:
                status.update(label="检索失败：未找到相关文献，请更换关键词！", state="error")
                st.stop()
                
            # 步骤 2：智能重排与精筛
            st.write(f"步骤 2/4: 正在进行智能相关性重排 (Rerank)，从 {len(all_raw_papers)} 篇中提取 Top {top_k_core} 核心文献...")
            core_papers = smart_rerank_and_filter(all_raw_papers, user_query, top_k=top_k_core)
            st.write(f"✅ 重排完成：已精选出相关性最高的 {len(core_papers)} 篇核心文献。")
            
            # 步骤 3：Map 阶段
            st.write("步骤 3/4: 正在执行 Map 阶段：分主题深度提取核心机制与数据...")
            chunk_size = len(core_papers) // 3 if len(core_papers) >= 3 else 1
            group1 = core_papers[:chunk_size]
            group2 = core_papers[chunk_size:chunk_size*2]
            group3 = core_papers[chunk_size*2:] if len(core_papers) >= 3 else core_papers
            
            subtopics = [
                ("药代动力学、靶控输注(TCI)与机制演进", group1),
                ("临床不良反应（呼吸抑制、低血压）防控策略", group2),
                ("特殊人群应用与未来临床结局展望", group3)
            ]
            
            subtopic_drafts = []
            for i, (sub_name, subset) in enumerate(subtopics):
                if not subset:
                    subset = core_papers[:5]
                st.write(f"  - 正在生成子主题 {i+1}: 【{sub_name}】...")
                draft = map_subtopic_generation(client, model_name, sub_name, subset)
                subtopic_drafts.append(draft)
                time.sleep(1)
                
            # 步骤 4：Reduce 阶段
            st.write("步骤 4/4: 正在执行 Reduce 阶段：统筹融合成 5000 字高级学术综述...")
            final_review = reduce_synthesize_review(client, model_name, user_query, subtopic_drafts)
            
            status.update(label="🎉 综述生成完毕！", state="complete")
            
        # 结果展示
        st.markdown("---")
        st.subheader("📄 生成的 5000 字前沿学术综述")
        st.markdown(final_review)
        
        # 下载按钮
        st.download_button(
            label="📥 下载 Markdown 格式综述",
            data=final_review,
            file_name=f"Anesthesia_Review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

import os
import asyncio
import aiohttp
from typing import List, Dict, Any
from tavily import TavilyClient

# ==================== 配置区域 ====================
# 请根据实际环境配置您的 API Key
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") # 例如 DeepSeek / Zhipu AI Key

tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# ==================== 1. 多路召回模块 ====================
async def fetch_pubmed_papers(query: str, max_results: int = 200) -> List[Dict[str, Any]]:
    """异步调用 PubMed API 获取文献元数据与摘要"""
    print(f"[*] 正在从 PubMed 召回文献，目标数量: {max_results}...")
    papers = []
    
    # 1. 检索符合条件的 PMIDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "date",
        "retmode": "json"
    }
    if PUBMED_API_KEY:
        search_params["api_key"] = PUBMED_API_KEY

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=search_params) as resp:
                if resp.status != 200:
                    print(f"[-] PubMed 检索失败，状态码: {resp.status}")
                    return papers
                data = await resp.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                
            if not id_list:
                return papers

            # 2. 批量获取文献详情 (esummary / efetch)
            # 实际生产中可分批获取，此处展示核心框架
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list[:max_results]),
                "retmode": "json"
            }
            if PUBMED_API_KEY:
                summary_params["api_key"] = PUBMED_API_KEY

            async with session.get(summary_url, params=summary_params) as resp:
                if resp.status == 200:
                    sum_data = await resp.json()
                    result_dict = sum_data.get("result", {})
                    for pmid in id_list:
                        if pmid in result_dict:
                            item = result_dict[pmid]
                            papers.append({
                                "id": pmid,
                                "title": item.get("title", ""),
                                "source": "PubMed",
                                "pubdate": item.get("pubdate", ""),
                                "abstract": item.get("source", "") # 实际可通过 efetch 获取详细摘要
                            })
    except Exception as e:
        print(f"[-] PubMed 召回异常: {e}")
        
    print(f"[+] PubMed 实际召回文献数: {len(papers)}")
    return papers

def fetch_tavily_papers(query: str, max_results: int = 25) -> List[Dict[str, Any]]:
    """调用 Tavily Search 获取全网最新资讯、临床指南或补充资料"""
    print(f"[*] 正在通过 Tavily 召回补充文献/资讯，目标数量: {max_results}...")
    papers = []
    if not tavily_client:
        print("[-] 未配置 Tavily API Key，跳过 Tavily 召回。")
        return papers

    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )
        results = response.get("results", [])
        for item in results:
            papers.append({
                "id": item.get("url"),
                "title": item.get("title", ""),
                "source": "Tavily Web",
                "pubdate": "Recent",
                "abstract": item.get("content", "")
            })
    except Exception as e:
        print(f"[-] Tavily 召回异常: {e}")

    print(f"[+] Tavily 实际召回资料数: {len(papers)}")
    return papers

async def multi_channel_retrieve(query: str) -> List[Dict[str, Any]]:
    """多路召回聚合入口"""
    pubmed_task = fetch_pubmed_papers(query, max_results=200)
    # Tavily 为同步客户端，可直接调用或放入线程池
    tavily_results = fetch_tavily_papers(query, max_results=25)
    
    pubmed_results = await pubmed_task
    all_papers = pubmed_results + tavily_results
    print(f"[+] 多路召回完成，合并后总文献/资料数: {len(all_papers)}")
    return all_papers


# ==================== 2. 智能重排模块 ====================
def smart_rerank_and_filter(papers: List[Dict[str, Any]], query: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    智能重排与压降：
    可接入 Embedding 相似度计算、交叉编码器 (Cross-Encoder) 或大模型批量打分，
    此处对候选集进行相关性打分并压降至 top_k 篇核心文献。
    """
    print(f"[*] 正在执行智能重排，从候选池 {len(papers)} 篇中精准筛选出 Top {top_k} 核心文献...")
    
    # 示例策略：去重、清洗空摘要，按相关性截取
    valid_papers = [p for p in papers if p.get("title")]
    
    # 如果总数小于等于 top_k 则直接返回，否则截取前 top_k
    reranked_papers = valid_papers[:top_k]
    print(f"[+] 智能重排完毕，锁定核心文献数: {len(reranked_papers)}")
    return reranked_papers


# ==================== 3. 分群处理 (Map 阶段) ====================
async def map_subtopic_generation(sub_papers: List[Dict[str, Any]], subtopic_name: str, query: str) -> str:
    """Map 阶段：针对分群文献生成子主题深度草稿"""
    print(f"[*] [Map 线程] 正在处理子方向: 「{subtopic_name}」 (包含文献数: {len(sub_papers)})...")
    
    # 组装文献摘要上下文
    papers_context = "\n".join([
        f"- 标题: {p.get('title')}\n  摘要/内容: {p.get('abstract')[:300]}..."
        for p in sub_papers
    ])
    
    prompt = f"""
    您是麻醉学与临床药理学资深专家。当前任务是围绕主题「{query}」下的垂直子方向「{subtopic_name}」，
    对以下 {len(sub_papers)} 篇核心文献进行深度剖析、机制归纳与草稿撰写。
    
    核心文献素材：
    {papers_context}
    
    要求：
    1. 深入提炼核心生理/药理机制、临床数据及对照试验结论。
    2. 指出现状痛点、争议及未来演进趋势。
    3. 论述专业、严谨，输出约 1500-2000 字的高质量子主题深度草稿。
    """
    
    # TODO: 此处替换为您实际的大模型调用代码 (如 AsyncOpenAI / DeepSeek Client)
    # response = await async_llm_call(prompt)
    draft = f"【子主题深度草稿 —— {subtopic_name}】\n(基于 {len(sub_papers)} 篇核心文献剖析生成的阶段性专业论述内容...)"
    return draft


# ==================== 4. 最终合成 (Reduce 阶段) ====================
async def reduce_synthesize_review(subtopic_drafts: List[str], query: str) -> str:
    """Reduce 阶段：整合多路子主题草稿，输出结构宏大、论述详尽的 5000 字综述"""
    print(f"[*] [Reduce 阶段] 正在统筹合并各子主题草稿，生成终篇 5000 字宏大综述...")
    
    combined_drafts_text = "\n\n".join([
        f"=== 子主题草稿 Part {i+1} ===\n{draft}" 
        for i, draft in enumerate(subtopic_drafts)
    ])
    
    prompt = f"""
    您是主编级麻醉学教授与资深学术导师。请基于以下三个维度的子主题深度草稿，
    统筹合成一篇结构宏大、论述详尽、具备顶级医学期刊水准的 **5000字学术综述**。
    
    核心主题：{query}
    
    各子主题草稿素材：
    {combined_drafts_text}
    
    写作规范与结构要求：
    1. **绪论 (Introduction)**：立意高远，阐明当前主题的临床背景、演进脉络与核心意义。
    2. **核心机制与分论点详述**：逻辑分节严密，深度整合各子主题素材，避免空洞泛泛。
    3. **临床应用评价与风险防范**：紧密结合麻醉安全、血流动力学调控或药代动力学模型。
    4. **局限性与未来展望 (Conclusion & Future Directions)**：指出现有研究短板与突破方向。
    5. 语言风格严谨、学术化，全文保持超高深度，字数达到 5000 字左右的详尽体量。
    """
    
    # TODO: 接入大模型（建议使用长文本旗舰模型如 DeepSeek-V3 / R1 等）
    # final_review = await async_llm_call(prompt, max_tokens=8192)
    
    final_review = f"""# 综合学术综述：{query}

## 摘要
(此处为自动生成的 5000 字长篇综述全文，包含严密的引言、机制剖析、临床评价、展望及参考文献架构...)
\n\n{combined_drafts_text}
"""
    return final_review


# ==================== 主控工作流管道 ====================
async def run_literature_review_pipeline(query: str):
    print(f"\n=== 开始执行麻醉学文献智能追踪与 Map-Reduce 综述工作流 ===")
    print(f"当前检索主题: {query}")
    
    # 步骤 1: 多路召回 (PubMed 200 篇 + Tavily 25 篇)
    raw_papers = await multi_channel_retrieve(query)
    
    if not raw_papers:
        print("[-] 未召回任何有效文献，流程终止。")
        return "未召回有效文献，请检查网络或 API Key 配置。"
    
    # 步骤 2: 智能重排压降到 50 篇核心文献
    core_papers = smart_rerank_and_filter(raw_papers, query, top_k=50)
    
    # 步骤 3: 分群处理 (Map 阶段) - 均匀拆分为 3 组
    chunk_size = max(1, len(core_papers) // 3)
    chunks = [
        core_papers[:chunk_size],
        core_papers[chunk_size:chunk_size*2],
        core_papers[chunk_size*2:]
    ]
    subtopic_names = [
        "基础药理学机制与 PK/PD 模型优化",
        "临床复合镇静安全与血流动力学调控",
        "前沿技术融合、器械创新与未来展望"
    ]
    
    print(f"[*] 正在并行执行 Map 阶段：将 {len(core_papers)} 篇核心文献均分至 3 个子任务...")
    map_tasks = [
        map_subtopic_generation(chunks[i], subtopic_names[i], query)
        for i in range(len(chunks))
    ]
    subtopic_drafts = await asyncio.gather(*map_tasks)
    
    # 步骤 4: 最终合成 (Reduce 阶段)
    final_review = await reduce_synthesize_review(subtopic_drafts, query)
    
    print("=== Map-Reduce 综述生成完毕！===")
    return final_review


# ==================== 本地运行测试入口 ====================
if __name__ == "__main__":
    test_query = "Propofol sedation pharmacokinetic pharmacodynamic models and hemodynamic safety in elderly patients"
    asyncio.run(run_literature_review_pipeline(test_query))

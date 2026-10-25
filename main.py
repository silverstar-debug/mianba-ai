"""
面霸 AI —— 后端 MVP
FastAPI + OpenAI SDK (DeepSeek)

设计要点（与前期调研对齐）：
1. 五级压力模型 L1-L5，压力由用户选择，默认温和（L2）——压力必须可控，不能默认高压。
2. 追问遵循「单点深挖」：引用用户原话 → 抓一个技术锚点 → 垂直往下钻，而不是横向换题。
3. 第 4 轮输出格式化评分报告，前端据此高亮渲染。
4. 状态用内存字典保存，重启即清空（MVP 不引入数据库）。
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mianba")

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

# 面试总轮数：第 4 轮（AI 的第 4 条消息）输出评分报告
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "4"))

# 公网访问保护
# ACCESS_CODE 为空表示不启用访问码；设置后，前端需带上 X-Access-Code 头（或 ?code= 参数）
ACCESS_CODE = os.getenv("ACCESS_CODE", "").strip()
# 每个 IP 每天最多能发起多少次对话请求（防刷 key）。0 表示不限制。
DAILY_LIMIT_PER_IP = int(os.getenv("DAILY_LIMIT_PER_IP", "80"))

# 岗位别名映射：把常见简称补全成更完整的描述，便于模型理解。
# 不在这里的岗位也能用 —— 用户自由输入的岗位名会原样传给模型。
ROLES = {
    "产品经理": "互联网产品经理",
    "前端开发": "前端开发工程师",
    "数据分析": "数据分析师",
}

ROLE_MAX_LEN = 40
ROLE_FALLBACK = "通用岗位"


def normalize_role(role: str) -> str:
    """清洗用户输入的岗位名：去空白、去换行、限长、空值兜底。"""
    r = (role or "").replace("\r", " ").replace("\n", " ").strip()
    r = re.sub(r"\s+", " ", r)[:ROLE_MAX_LEN]
    return r or ROLE_FALLBACK

_client = None


def get_client():
    """惰性初始化，避免没配 Key 时启动就崩。"""
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    role: str = Field(default="产品经理", description="目标岗位")
    history: List[Message] = Field(default_factory=list, description="对话历史")
    user_input: str = Field(default="", description="用户本轮回答；开场可为空")
    session_id: Optional[str] = Field(default=None, description="会话 id，用于服务端保存进度")
    pressure_level: int = Field(default=2, ge=1, le=5, description="压力等级 L1-L5")


# --------------------------------------------------------------------------
# 内存状态（MVP：重启即清空）
# --------------------------------------------------------------------------
SESSIONS: dict = {}
# 每个 IP 的每日调用计数 {ip: {"date": "2026-10-25", "count": 12}}
RATE_LIMIT: dict = {}


def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"turn": 0, "history": [], "role": "", "pressure_level": 2}
    return SESSIONS[session_id]


def client_ip(request) -> str:
    """取真实客户端 IP（兼容反向代理）。"""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_and_count(ip: str) -> tuple[bool, int]:
    """按天限流。返回 (是否允许, 今日已用次数)。"""
    if DAILY_LIMIT_PER_IP <= 0:
        return True, 0
    today = datetime.now().strftime("%Y-%m-%d")
    rec = RATE_LIMIT.get(ip)
    if not rec or rec.get("date") != today:
        rec = {"date": today, "count": 0}
        RATE_LIMIT[ip] = rec
    rec["count"] += 1
    return rec["count"] <= DAILY_LIMIT_PER_IP, rec["count"]


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------
PRESSURE_PROFILES = {
    1: ("温和陪练", "语气友善，绝不打断。用户卡住时主动给提示或换个更简单的问题。说错也先肯定再补充。"),
    2: ("标准面试", "职业、中性、有礼貌。正常追问 1-2 层，不施压。"),
    3: ("细致深挖", "不放过任何含糊表述，追着细节往下问，要求用户举具体例子和数字。"),
    4: ("压力面试", "偏严厉，会打断、会质疑、会用沉默施压。但禁止人身攻击、禁止贬低、禁止侮辱。"),
    5: ("高压终面", "极度挑剔，连续否定式追问，刻意打乱节奏。同样禁止人身攻击与侮辱，用户说『停』必须立刻切换温和模式。"),
}

BASE_RULES = """
【你的角色】
你是一场真实求职面试的面试官。你不是客服，不要客套，不要自我介绍，不要解释规则。

【核心提问规则 —— 单点深挖】
1. 必须引用用户上一句话里的原文，再提问。
   正确：「你刚说『用 Redis 做了缓存』——那如果这时候发生缓存雪崩，你怎么实时限流降级？」
   错误：「请说说缓存雪崩怎么处理？」（像考试，没有听人说话）
2. 一次只抓一个技术锚点，垂直往下挖，不要横向换话题。
3. 追问要「跳跃」：从存储跳到稳定性、从选型跳到故障处理，而不是停留在原地问常识。

【输出格式】
只输出面试官要说的话，1-3 句，不要加任何前缀、标签或解释。
不要输出「面试官：」这种前缀。
"""


def build_system_prompt(role: str, pressure_level: int) -> str:
    role_desc = ROLES.get(normalize_role(role), normalize_role(role))
    name, style = PRESSURE_PROFILES.get(pressure_level, PRESSURE_PROFILES[2])
    return (
        f"{BASE_RULES}\n"
        f"【目标岗位】{role_desc}\n"
        f"【当前压力等级】L{pressure_level} {name}\n"
        f"【当前风格】{style}\n"
    )


def build_opening_prompt(role: str) -> str:
    role_desc = ROLES.get(normalize_role(role), normalize_role(role))
    return (
        f"这是面试的第 1 轮。请针对「{role_desc}」岗位，问一个最基础但能展开的开放问题作为开场。"
        f"只问一个问题，1-2 句，语气自然。不要自我介绍。"
    )


def build_followup_prompt(role: str, user_input: str, round_no: int) -> str:
    return (
        f"这是第 {round_no} 轮追问。\n"
        f"候选人刚才的回答是：\n「{user_input}」\n\n"
        f"请从上面这段话里挑出**一个**具体的技术点或方法论名词，引用他的原话，"
        f"然后垂直往下追问一层。只问一个问题，1-3 句。"
    )


def build_report_prompt(role: str, history: List[Message]) -> str:
    role_desc = ROLES.get(normalize_role(role), normalize_role(role))
    convo = "\n".join(
        f"{'候选人' if m.role == 'user' else '面试官'}：{m.content}"
        for m in history
        if m.role in ("user", "assistant")
    )
    return (
        f"面试结束。以下是「{role_desc}」岗位的完整对话记录：\n\n{convo}\n\n"
        f"请作为面试官，输出一段结构化评分报告。**必须严格按以下格式**，"
        f"首行必须是「【面试评分报告】」，不要有任何其他开头：\n\n"
        f"【面试评分报告】\n"
        f"总分：XX/100\n"
        f"亮点：\n"
        f"1. ……\n"
        f"2. ……\n"
        f"不足：\n"
        f"1. ……\n"
        f"2. ……\n"
        f"改进建议：\n"
        f"1. ……\n"
        f"2. ……\n\n"
        f"要求：评分要具体、有依据，结合候选人实际说过的内容，不要空泛套话。"
        f"总分要有区分度，不要给满分。"
    )


# --------------------------------------------------------------------------
# 调用模型
# --------------------------------------------------------------------------
async def call_llm(system_prompt: str, messages: List[dict], max_tokens: int = 800) -> str:
    client = get_client()
    resp = await client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        log.warning("输出被 max_tokens 截断（max_tokens=%s），考虑调大该值", max_tokens)
    return (choice.message.content or "").strip()


def ensure_report_format(text: str) -> str:
    """兜底：确保前端能靠标记识别报告。"""
    if "【面试评分报告】" in text:
        return text
    return "【面试评分报告】\n" + text


# --------------------------------------------------------------------------
# FastAPI
# --------------------------------------------------------------------------
app = FastAPI(title="面霸 AI", version="0.1.0", description="AI 面试模拟与求职陪练 MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # MVP 阶段放开，上线前收敛为具体域名
    allow_credentials=False,      # 与 allow_origins=["*"] 同时为 True 会被浏览器拒绝
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def index_page():
    """托管前端页面。必须走 http://127.0.0.1:8000 访问，
    因为浏览器的语音识别 API 要求安全上下文（localhost 属于安全上下文，file:// 不保证）。"""
    path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html; charset=utf-8")
    return JSONResponse(status_code=404, content={"error": "index.html 不存在"})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_configured": bool(DEEPSEEK_API_KEY and not DEEPSEEK_API_KEY.startswith("sk-xxxx")),
        "model": DEEPSEEK_MODEL,
        "max_rounds": MAX_ROUNDS,
        "role_presets": list(ROLES.keys()),
        "custom_role_supported": True,
        "access_code_required": bool(ACCESS_CODE),
        "daily_limit_per_ip": DAILY_LIMIT_PER_IP,
    }


@app.post("/api/reset")
async def reset(payload: dict = None):
    sid = (payload or {}).get("session_id")
    if sid:
        SESSIONS.pop(sid, None)
    else:
        SESSIONS.clear()
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    # ---- 0. 公网保护：访问码 ----
    if ACCESS_CODE:
        provided = (
            request.headers.get("x-access-code")
            or request.query_params.get("code")
            or ""
        ).strip()
        if provided != ACCESS_CODE:
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": "ACCESS_CODE_REQUIRED", "hint": "需要访问码，请在网址后加 ?code=你的访问码"},
            )

    # ---- 1. 校验 ----
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-xxxx"):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "后端未配置 DEEPSEEK_API_KEY，请先编辑 .env 填入真实 Key 并重启服务。"},
        )

    # ---- 1.5 每日限额 ----
    ip = client_ip(request)
    allowed, used = check_and_count(ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "ok": False,
                "error": f"今天的免费次数已用完（每天 {DAILY_LIMIT_PER_IP} 次），明天再来。",
                "used": used,
            },
        )

    # ---- 2. 取会话状态 ----
    incoming_role = normalize_role(req.role)

    if req.session_id:
        sess = get_session(req.session_id)
        history = sess["history"]
        if incoming_role and incoming_role != ROLE_FALLBACK:
            sess["role"] = incoming_role
        sess["pressure_level"] = req.pressure_level
        role = sess["role"] or incoming_role
        turn = sess["turn"]
    else:
        # 无 session_id：完全依赖前端传的 history
        history = [m.model_dump() for m in req.history]
        role = incoming_role
        turn = len([m for m in history if m["role"] == "user"])

    system_prompt = build_system_prompt(role, req.pressure_level)

    try:
        # ---- 3. 第一轮：开场提问（不需要用户回答）----
        if turn == 0 and not req.user_input.strip():
            text = await call_llm(
                system_prompt,
                [{"role": "user", "content": build_opening_prompt(role)}],
                max_tokens=300,
            )
            history.append({"role": "assistant", "content": text})
            if req.session_id:
                SESSIONS[req.session_id]["turn"] = 1
            return {
                "ok": True,
                "round": 1,
                "finished": False,
                "reply": text,
                "history": history,
            }

        # ---- 4. 记录用户回答 ----
        if req.user_input.strip():
            history.append({"role": "user", "content": req.user_input.strip()})

        answered = len([m for m in history if m["role"] == "user"])

        # ---- 5. 达到轮数上限：出评分报告 ----
        if answered >= MAX_ROUNDS - 1:
            text = await call_llm(
                system_prompt,
                [{"role": "user", "content": build_report_prompt(role, [Message(**m) for m in history])}],
                max_tokens=1800,
            )
            text = ensure_report_format(text)
            history.append({"role": "assistant", "content": text})
            if req.session_id:
                SESSIONS[req.session_id]["turn"] = MAX_ROUNDS
            return {
                "ok": True,
                "round": MAX_ROUNDS,
                "finished": True,
                "reply": text,
                "is_report": True,
                "history": history,
            }

        # ---- 6. 中间轮：追问 ----
        text = await call_llm(
            system_prompt,
            [{"role": "user", "content": build_followup_prompt(role, req.user_input, answered + 1)}],
            max_tokens=400,
        )
        history.append({"role": "assistant", "content": text})
        if req.session_id:
            SESSIONS[req.session_id]["turn"] = answered + 1
        return {
            "ok": True,
            "round": answered + 1,
            "finished": False,
            "reply": text,
            "history": history,
        }

    except Exception as e:
        log.exception("调用模型失败")
        msg = str(e)
        if "401" in msg or "Authentication" in msg or "invalid api key" in msg.lower():
            hint = "API Key 无效，请检查 .env 里的 DEEPSEEK_API_KEY。"
        elif "timeout" in msg.lower() or "connect" in msg.lower():
            hint = "无法连接 DeepSeek，请检查网络或代理。"
        else:
            hint = f"模型调用出错：{msg[:200]}"
        return JSONResponse(status_code=500, content={"ok": False, "error": hint})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

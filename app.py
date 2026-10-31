"""
    FastAPI 接口：app.py

    职责：只做「装配与转发」——创建应用、挂载静态资源、把 HTTP / WebSocket 请求
        转交给 QASystem 处理，不承载业务逻辑（问答与会话逻辑均在 qa_system.py）。

    接口清单：
        GET  /                     问答页面
        GET  /health               健康检查
        GET  /api/kb_types         知识库类型清单
        POST /api/create_session   创建会话
        GET  /api/history/{id}     查询会话历史
        DELETE /api/history/{id}   清除会话历史
        POST /api/query            非流式问答
        WS   /api/stream           流式问答
"""
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from base.config import config
from base.logger import logger
from qa_system import QASystem
# 行级权限：把调用方角色翻译成过滤条件（需求 4.2）
from core.access_control import build_acl, acl_summary
from core.loaders import loader_report

# 创建应用实例
app = FastAPI(title="食安通 · 食品安全知识库系统",
              description="基于 RAG 的食品安全法规、SOP、事故处置智能问答服务")

# 配置 CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局问答系统实例（进程内单例，避免重复加载模型）
qa_system = QASystem()

# 知识库类型清单（唯一来源在 base/config.py，此处仅做接口格式转换）
KB_TYPES = [{"key": k, "name": v} for k, v in config.KB_TYPES.items()]


# 请求模型
class QueryRequest(BaseModel):
    query: str
    kb_type: Optional[str] = None      # 知识库过滤：regulation / sop / case / nutrition / supplier / faq
    session_id: Optional[str] = None
    # 行级权限（需求 4.2）：role 决定可见密级；department 可选，不传则按角色的部门清单。
    # 两者都为空时使用 config.ini [access] default_role（默认 staff）——
    # 即「不声明身份就拿最小可见范围」，而不是默认全库可见。
    role: Optional[str] = None         # public / staff / kitchen / quality / supplier / admin
    department: Optional[str] = None   # all / quality / kitchen / purchasing / management


# 响应模型
class QueryResponse(BaseModel):
    answer: str
    sources: list
    kb_type: Optional[str] = None
    session_id: str
    processing_time: float
    # 回显本次实际使用的权限上下文，便于前端展示与排障（需求 4.2）
    acl: Optional[dict] = None


# 挂载静态资源目录
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
async def read_root():
    """问答页面"""
    return FileResponse(f"{config.STATIC_DIR}/index.html")


@app.get("/health")
async def health_check():
    """
    健康检查。

    除了存活状态，还回显权限模型与文档加载器的可用性——
    这两项都容易出现「配置写了但没生效」的情况（权限映射为空、PDF 依赖没装），
    放在 /health 里能一眼看出来，不用去翻日志。
    """
    from core.incremental import load_manifest, describe_manifest
    return {
        "status": "healthy",
        "access_control": acl_summary(),
        "loaders": loader_report()["active"],
        "supported_extensions": loader_report()["supported_extensions"],
        "kb_manifest": describe_manifest(load_manifest()),
    }


@app.get("/api/kb_types")
async def get_kb_types():
    """返回知识库类型清单"""
    return {"kb_types": KB_TYPES}


@app.post("/api/create_session")
async def create_session():
    """创建新会话"""
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """查询会话历史"""
    try:
        return {"session_id": session_id, "history": qa_system.get_history(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {e}")


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """清除会话历史"""
    qa_system.clear_history(session_id)
    return {"status": "success", "message": "历史记录已清除"}


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """非流式问答接口"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    # 行级权限：角色 → 可见密级 + 可见部门（需求 4.2）
    acl = build_acl(role=request.role, department=request.department)
    logger.info(f"收到问答请求: query='{request.query}', kb_type={request.kb_type}, "
                f"session={session_id}, acl={acl.describe()}")

    answer, sources, _ = qa_system.ask(request.query, kb_filter=request.kb_type,
                                       session_id=session_id, acl=acl)
    return QueryResponse(answer=answer, sources=sources, kb_type=request.kb_type,
                         session_id=session_id, processing_time=time.time() - start_time,
                         acl=acl.describe())


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    """流式问答 WebSocket 接口"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            query = request_data.get("query")
            kb_type = request_data.get("kb_type")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            # 行级权限：前端可在每条消息里带 role / department（需求 4.2）
            acl = build_acl(role=request_data.get("role"),
                            department=request_data.get("department"))
            start_time = time.time()

            # 发送开始标志（带上权限上下文，前端可据此提示"当前以哪个角色的可见范围作答"）
            await websocket.send_json({"type": "start", "session_id": session_id,
                                       "acl": acl.describe()})

            # 流式返回 RAG 结果（问候语短路逻辑在 QASystem.ask_stream 内，此处不再重复判断）
            sources = []
            for token, is_last, meta in qa_system.ask_stream(query, kb_filter=kb_type,
                                                             session_id=session_id, acl=acl):
                if is_last:
                    sources = meta.get("sources", [])
                if token:
                    await websocket.send_json({"type": "token", "token": token, "session_id": session_id})
            await websocket.send_json({"type": "end", "session_id": session_id, "is_complete": True,
                                       "sources": sources, "processing_time": time.time() - start_time})

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket 断开: code={e.code}")
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    # 运行策略：对话模型为远程 API（非本地部署、按 token 计费），
    # 启动时明确打印本次运行的限制，避免误跑全量流程（详见 base/run_policy.py）
    from base.run_policy import run_policy
    run_policy.print_banner()

    print("=" * 60)
    print("食安通 · 食品安全知识库系统")
    print(f"访问地址： http://127.0.0.1:{config.APP_PORT}")
    print(f"API 文档： http://127.0.0.1:{config.APP_PORT}/docs")
    print(f"（监听 {config.APP_HOST}:{config.APP_PORT}；0.0.0.0 是监听地址，"
          f"浏览器请用上面的 127.0.0.1）")
    print("=" * 60)
    uvicorn.run(app=app, host=config.APP_HOST, port=config.APP_PORT, reload=False)

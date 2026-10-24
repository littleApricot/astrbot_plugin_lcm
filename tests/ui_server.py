"""Serve disposable fixtures through actual read-only plugin handlers."""

import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import uvicorn
from astrbot.api.web import PluginRequest, bind_request_context
from astrbot.dashboard.services.plugin_page_service import PluginPageService
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from astrbot_plugin_lcm.main import LCM

folder = tempfile.TemporaryDirectory(prefix="lcm-isolated-ui-")
with patch(
    "astrbot_plugin_lcm.main.StarTools.get_data_dir", return_value=Path(folder.name)
):
    plugin = LCM(
        SimpleNamespace(
            register_web_api=lambda *a: None,
            get_config=lambda **k: {"timezone": "Asia/Shanghai"},
        ),
        {"base_tokens": 40},
    )
scope = plugin.store.scope(
    "QQ 讨论群 · 小比的日常", "viewer-demo", "QQ 讨论群 · 小比的日常"
)
for i in range(12):
    plugin.store.append(
        scope,
        {
            "role": "user",
            "content": f"我们讨论过 LCM：第 {i + 1} 个话题，确认保留原始聊天、完整工具执行和时间。",
        },
        sender="小幸 / user-1",
        tokens=20,
    )
for i in range(5):
    job_id = plugin.store.plan_job(scope, 40)
    job = plugin.store.next_job()
    plugin.store.job_result(
        job_id,
        "讨论确认：保留完整原始消息与工具调用关系；使用后台分层压缩，最近原文继续用于对话。"
        if i == 0
        else "继续讨论了记忆与上下文的边界，确认摘要仅作为历史资料，不改变人格。",
    )
    # Make fixture summaries large enough to demonstrate second-level provenance.
    plugin.store.count = lambda v: 25
    plugin.store.publish(job)
    plugin.store.count = lambda v: 20
job_id = plugin.store.plan_job(scope, 40)
job = plugin.store.next_job()
if job:
    plugin.store.count = lambda v: 15
    plugin.store.job_result(
        job_id,
        "群友讨论了上下文管理方案：保留原文、按完整轮次后台压缩，并提供来源查看入口。",
    )
    plugin.store.publish(job)
    plugin.store.count = lambda v: 20
mid = plugin.store.append(
    scope,
    {"role": "user", "content": "那查看器能逐级展开到最初的聊天吗？"},
    sender="琛 / user-2",
    tokens=20,
)
plugin.store.reserve(scope, [mid], "demo")
run = plugin.store.begin(scope, "demo", plugin.store.view(scope, exclude=[mid]), [mid])
plugin.store.snapshot(
    scope,
    run,
    {
        "contexts": [
            {"role": "system", "content": "你是小比。当前记忆：正在讨论上下文查看器。"},
            {"role": "user", "content": "那查看器能逐级展开到最初的聊天吗？"},
        ],
        "tools": [{"name": "lcm_expand", "description": "查看完整来源"}],
        "model": "deterministic-fixture",
        "provider": "isolated",
        "token_method": "estimate",
    },
)
plugin.store.append(
    scope,
    {
        "role": "assistant",
        "content": "可以，从高级摘要逐层展开到原始完整轮次，时间、发送者和工具结果都会保留。",
    },
    round_id=run,
    sender="小比",
    tokens=22,
)
plugin.store.finish(run)
before = plugin.messages(plugin.store.view(scope)["nodes"])
after = copy.deepcopy(before)
after[-1]["content"] = "历史修订示例：保留发出的图片与最后回复。"
plugin.store.sync_history(scope, before, after)
plugin.store.scope("另一个群 · 读书闲聊", "separate-session", "另一个群 · 读书闲聊")
app = FastAPI()
root = Path(__file__).resolve().parents[1] / "pages/context"
metadata = SimpleNamespace(
    name="astrbot_plugin_lcm",
    activated=True,
    root_dir_name="astrbot_plugin_lcm",
    reserved=False,
    i18n={},
    display_name="LCM 上下文与历史",
)
page_service = PluginPageService(
    SimpleNamespace(
        context=SimpleNamespace(get_all_stars=lambda: [metadata]),
        plugin_store_path=str(root.parents[2]),
    )
)


@app.get("/api/{endpoint}")
@app.get("/api/v1/plugins/extensions/astrbot_plugin_lcm/{endpoint}")
async def api(endpoint: str, request: Request):
    with bind_request_context(
        PluginRequest(
            request, plugin_name="astrbot_plugin_lcm", username="isolated-test"
        )
    ):
        return await plugin.api()


@app.get("/view/viewer.js")
async def js():
    return FileResponse(root / "viewer.js", media_type="application/javascript")


@app.get("/api/plugin/page/bridge-sdk.js")
async def sdk():
    payload = await page_service.serve_bridge_sdk(
        asset_token="", locale="zh-CN", theme="light"
    )
    return Response(payload.content, media_type="application/javascript")


@app.get("/view/")
async def page():
    payload = await page_service.serve_page_content(
        plugin_name=metadata.name,
        view_name="context",
        asset_path="index.html",
        asset_token="",
        username=None,
        locale="zh-CN",
        theme="light",
    )
    return HTMLResponse(payload.content)


@app.get("/")
async def host():
    # The production Vue host performs these postMessage requests with its
    # authenticated HTTP client. This fixture retains the actual native SDK.
    return HTMLResponse("""<!doctype html><html><head><style>body{margin:0}iframe{border:0;width:100%;height:100vh;display:block}</style></head><body><script>
window.addEventListener('message',async event=>{
 const frame=document.querySelector('iframe');const m=event.data;
 if(event.origin!==location.origin||event.source!==frame.contentWindow||m?.channel!=='astrbot-plugin-page')return;
 const send=data=>frame.contentWindow.postMessage({channel:'astrbot-plugin-page',...data},location.origin);
 if(m.kind==='ready')send({kind:'context',context:{pluginName:'astrbot_plugin_lcm',locale:'zh-CN',isDark:false}});
 if(m.kind==='request')try{
  if(m.action!=='api:get'||!/^[a-z]+$/.test(m.endpoint))throw Error('invalid fixture API request');
  const response=await fetch('/api/v1/plugins/extensions/astrbot_plugin_lcm/'+m.endpoint+'?'+new URLSearchParams(m.params||{}));
  const data=await response.json();if(!response.ok)throw Error(data.message||'API failed');
  send({kind:'response',requestId:m.requestId,ok:true,data});
 }catch(e){send({kind:'response',requestId:m.requestId,ok:false,error:e.message});}
});
</script><iframe name="viewer" title="LCM 原生 SDK 测试页面" src="/view/"></iframe></body></html>""")


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 6197
    )

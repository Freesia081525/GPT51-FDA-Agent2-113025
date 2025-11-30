import os
import json
import re
from typing import Dict, Any, List, Optional
import streamlit as st
import yaml

# --- LLM client libraries ---
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

# -----------------------------------------------------------
# FDA 510(k) Theme Configuration
# -----------------------------------------------------------

FDA_THEMES = {
    "light": {
        "primary": "#0052CC",
        "secondary": "#00838F",
        "background": "#F7FAFC",
        "text": "#1A202C",
        "accent": "#FF7F50",  # coral for key highlights
    },
    "dark": {
        "primary": "#63B3ED",
        "secondary": "#00B5D8",
        "background": "#1A202C",
        "text": "#E2E8F0",
        "accent": "#FF7F50",  # coral for key highlights
    }
}

REVIEW_CONTEXT_STYLES = {
    "General 510(k)": {
        "icon": "📁",
        "description": "一般 510(k) 傳統醫療器材審查情境",
        "color": "#2B6CB0",
    },
    "Orthopedic": {
        "icon": "🦴",
        "description": "骨科植入物與器材審查情境",
        "color": "#805AD5",
    },
    "Cardiovascular": {
        "icon": "❤️",
        "description": "心血管裝置與支架審查情境",
        "color": "#E53E3E",
    },
    "Radiology": {
        "icon": "🩻",
        "description": "影像診斷設備與 AI 讀片輔助審查情境",
        "color": "#3182CE",
    },
    "In Vitro Diagnostic": {
        "icon": "🧪",
        "description": "體外診斷 (IVD) 試劑與儀器審查情境",
        "color": "#38A169",
    },
    "Digital Health": {
        "icon": "📱",
        "description": "數位健康、SaMD 與遠距監測系統審查情境",
        "color": "#D53F8C",
    },
    "Surgical": {
        "icon": "🔪",
        "description": "手術器械與能量設備審查情境",
        "color": "#DD6B20",
    },
    "Dental": {
        "icon": "🦷",
        "description": "牙科裝置與材料審查情境",
        "color": "#319795",
    },
    "Anesthesiology": {
        "icon": "💤",
        "description": "麻醉與呼吸治療設備審查情境",
        "color": "#4A5568",
    },
    "Combination Product": {
        "icon": "💊",
        "description": "藥械組合產品與邊界產品審查情境",
        "color": "#B83280",
    },
}

TRANSLATIONS = {
    "en": {
        "title": "FDA 510(k) Multi-Agent Review Studio",
        "subtitle": "Role: Professional Regulatory AI Orchestrator",
        "theme": "UI Theme",
        "language": "Language",
        "art_style": "Review Context Style",
        "health": "Compliance Health",
        "mana": "AI Resource Capacity",
        "experience": "Case Experience",
        "api_keys": "API Keys",
        "input": "Case Inputs",
        "pipeline": "Review Pipelines",
        "smart_replace": "Smart Editing",
        "notes": "AI Note Keeper",
        "dashboard": "Dashboard",
        "run": "Run Pipeline",
        "level": "Maturity Level",
        "quest_log": "Case Log",
        "achievements": "Milestones",
    },
    "zh": {
        "title": "FDA 510(k) 多代理審查工作室",
        "subtitle": "專業角色：FDA 醫療器材 510(k) 審查協作代理系統",
        "theme": "介面主題",
        "language": "語言",
        "art_style": "審查情境風格",
        "health": "合規健康度",
        "mana": "AI 資源容量",
        "experience": "案件經驗值",
        "api_keys": "API 金鑰",
        "input": "案件輸入",
        "pipeline": "審查流程",
        "smart_replace": "智能編輯",
        "notes": "AI 筆記助手",
        "dashboard": "儀表板",
        "run": "執行流程",
        "level": "審查成熟度等級",
        "quest_log": "案件紀錄",
        "achievements": "重要里程碑",
    }
}

# -----------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "theme": "dark",
        "language": "zh",
        "art_style": "General 510(k)",
        "player_level": 1,
        "health": 100,
        "mana": 100,
        "experience": 0,
        "quests_completed": 0,
        "achievements": [],
        "combat_log": [],
        "template": "## 案件模板\n\n在此撰寫或貼上 510(k) 案件相關模板內容...",
        "observations": "在此新增臨床、風險或技術觀察備註...",
        "pipeline_history": [],
        "note_raw_text": "",
        "note_markdown": "",
        "note_formatted": "",
        "note_keywords_output": "",
        "note_entities_json_data": [],
        "note_mindmap_json_text": "",
        "note_wordgraph_json_text": "",
        "note_chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

@st.cache_data
def load_agents_config(path: str = "agents.yaml") -> Dict[str, Any]:
    """Load agents configuration from YAML file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"agents": [], "pipelines": []}

def get_translation(key: str) -> str:
    """Get translated text based on current language"""
    lang = st.session_state.get("language", "zh")
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)

def apply_custom_css():
    """Apply FDA 510(k)-themed custom CSS"""
    theme = st.session_state.get("theme", "dark")
    style = st.session_state.get("art_style", "General 510(k)")
    colors = FDA_THEMES[theme]
    accent_color = REVIEW_CONTEXT_STYLES.get(style, REVIEW_CONTEXT_STYLES["General 510(k)"])["color"]
    
    css = f"""
    <style>
    /* Main theme colors */
    .stApp {{
        background-color: {colors['background']};
        color: {colors['text']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {colors['primary']};
        border-bottom: 3px solid {accent_color};
        padding-bottom: 6px;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(145deg, {accent_color}, {colors['secondary']});
        color: white;
        border: 1px solid {colors['primary']};
        border-radius: 6px;
        font-weight: 600;
        padding: 6px 16px;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.18);
    }}
    
    /* Status bars container */
    .status-bar {{
        background: linear-gradient(90deg, {accent_color}, transparent);
        border: 1px solid {colors['primary']};
        border-radius: 8px;
        padding: 4px 6px;
        margin: 4px 0;
    }}
    
    /* Card style */
    .review-card {{
        background: {colors['background']};
        border: 2px solid {accent_color};
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: rgba(0,0,0,0.05);
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {colors['secondary']};
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid {colors['primary']};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(145deg, {accent_color}, {colors['primary']});
    }}
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: rgba(0,0,0,0.02);
        color: {colors['text']};
        border-radius: 6px;
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: {colors['background']};
        border-right: 2px solid {accent_color};
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background-color: {accent_color};
    }}
    
    /* Expander header */
    .streamlit-expanderHeader {{
        background-color: {colors['secondary']};
        color: white;
        border-radius: 6px;
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def update_player_stats(action: str):
    """
    Update abstracted 'player' stats, re-interpreted as review metrics:
    - level: 審查成熟度等級
    - health: 合規健康度
    - mana: AI 資源容量
    """
    if action == "quest_complete":
        st.session_state.experience += 10
        st.session_state.quests_completed += 1
        if st.session_state.experience >= st.session_state.player_level * 50:
            st.session_state.player_level += 1
            st.session_state.experience = 0
            st.toast(f"🎯 審查成熟度提升！目前等級：{st.session_state.player_level}")
    elif action == "use_mana":
        st.session_state.mana = max(0, st.session_state.mana - 20)
    elif action == "regenerate":
        st.session_state.mana = min(100, st.session_state.mana + 10)
        st.session_state.health = min(100, st.session_state.health + 5)

def add_combat_log(message: str, message_type: str = "info"):
    """Add entry to review activity log"""
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "spell": "🧠",
    }
    log_entry = {
        "icon": icons.get(message_type, "ℹ️"),
        "message": message,
        "timestamp": st.session_state.get("quests_completed", 0),
    }
    if "combat_log" not in st.session_state:
        st.session_state.combat_log = []
    st.session_state.combat_log.append(log_entry)
    if len(st.session_state.combat_log) > 50:
        st.session_state.combat_log.pop(0)

# -----------------------------------------------------------
# API Key Management
# -----------------------------------------------------------

def get_api_key_from_env_or_ui(
    provider_name: str,
    env_var: str,
    session_key: str,
    label: str,
) -> Optional[str]:
    """Get API key from environment or user input"""
    env_val = os.getenv(env_var)
    if env_val:
        st.caption(f"🔑 {label}: 已從環境變數載入")
        st.session_state[session_key] = env_val
        return env_val

    key = st.text_input(
        label,
        value=st.session_state.get(session_key, ""),
        type="password",
    )
    if key:
        st.session_state[session_key] = key
        st.caption(f"🔑 {label} 已暫存於工作階段")
        return key
    return None

# -----------------------------------------------------------
# LLM Call Router
# -----------------------------------------------------------

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Route LLM calls to appropriate provider"""
    provider = provider.lower().strip()
    
    add_combat_log(f"呼叫 {provider} 模型：{model}", "spell")
    update_player_stats("use_mana")

    if provider == "openai":
        api_key = st.session_state.get("openai_api_key")
        if not api_key:
            raise RuntimeError("OpenAI API key is not set.")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        api_key = st.session_state.get("gemini_api_key")
        if not api_key:
            raise RuntimeError("Gemini API key is not set.")
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(
            system_prompt + "\n\nUSER MESSAGE:\n" + user_prompt
        )
        return resp.text

    elif provider == "xai":
        api_key = st.session_state.get("xai_api_key")
        if not api_key:
            raise RuntimeError("xAI API key is not set.")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        api_key = st.session_state.get("anthropic_api_key")
        if not api_key:
            raise RuntimeError("Anthropic API key is not set.")
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if resp.content and len(resp.content) > 0:
            block = resp.content[0]
            if hasattr(block, "text"):
                return block.text
        return json.dumps(resp.model_dump(), indent=2)

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def run_agent(
    agent_cfg: Dict[str, Any],
    user_prompt: str,
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
    override_system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run a single configured agent"""
    provider = override_provider or agent_cfg.get("provider", "openai")
    model = override_model or agent_cfg.get("default_model", "gpt-4o-mini")
    system_prompt = override_system_prompt or agent_cfg.get("system_prompt", "")
    return call_llm(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# -----------------------------------------------------------
# Status Indicators
# -----------------------------------------------------------

def render_status_indicators():
    """Render review status indicators"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"### {get_translation('level')} {st.session_state.player_level}")
        
    with col2:
        st.markdown(f"### {get_translation('health')}")
        st.progress(st.session_state.health / 100)
        st.caption(f"{st.session_state.health}/100")
        
    with col3:
        st.markdown(f"### {get_translation('mana')}")
        st.progress(st.session_state.mana / 100)
        st.caption(f"{st.session_state.mana}/100")
        
    with col4:
        st.markdown(f"### {get_translation('experience')}")
        max_xp = st.session_state.player_level * 50
        st.progress(st.session_state.experience / max_xp)
        st.caption(f"{st.session_state.experience}/{max_xp}")

def render_activity_log():
    """Render review activity log"""
    st.markdown("### 📑 活動紀錄")
    with st.expander("檢視近期動作", expanded=False):
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log[-20:]):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("目前尚無活動紀錄")

# -----------------------------------------------------------
# Review Context Selector
# -----------------------------------------------------------

def render_review_context_selector():
    """Render interactive review context selector"""
    st.markdown("### 🏥 審查情境選擇器")
    
    cols = st.columns(5)
    styles = list(REVIEW_CONTEXT_STYLES.keys())
    
    for idx, style in enumerate(styles):
        with cols[idx % 5]:
            style_data = REVIEW_CONTEXT_STYLES[style]
            button_label = f"{style_data['icon']} {style}"
            
            if st.button(
                button_label,
                key=f"style_{style}",
                help=style_data["description"],
                use_container_width=True
            ):
                st.session_state.art_style = style
                add_combat_log(f"切換審查情境為：{style}", "success")
                st.rerun()
    
    current_style = st.session_state.get("art_style", "General 510(k)")
    style_data = REVIEW_CONTEXT_STYLES[current_style]
    st.markdown(
        f"<div class='review-card' style='text-align: center; "
        f"background: linear-gradient(145deg, {style_data['color']}, transparent);'>"
        f"<h3>{style_data['icon']} 目前情境：{current_style}</h3>"
        f"<p>{style_data['description']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------
# Enhanced Sidebar
# -----------------------------------------------------------

def render_enhanced_sidebar(config: Dict[str, Any]):
    """Render FDA 510(k)-themed sidebar with controls"""
    st.sidebar.markdown(f"# {get_translation('title')}")
    st.sidebar.markdown(f"*{get_translation('subtitle')}*")
    
    st.sidebar.markdown("---")
    
    # Theme and Language Selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theme = st.selectbox(
            get_translation("theme"),
            ["light", "dark"],
            index=1 if st.session_state.theme == "dark" else 0,
            key="theme_selector"
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            st.rerun()
    
    with col2:
        lang = st.selectbox(
            get_translation("language"),
            ["zh", "en"],
            index=0 if st.session_state.language == "zh" else 1,
            key="lang_selector"
        )
        if lang != st.session_state.language:
            st.session_state.language = lang
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Review Status
    st.sidebar.markdown("### 📊 審查狀態總覽")
    render_status_indicators()
    
    st.sidebar.markdown("---")
    
    # API Keys
    st.sidebar.markdown(f"### 🔑 {get_translation('api_keys')}")
    
    with st.sidebar.expander("設定 API 金鑰"):
        get_api_key_from_env_or_ui(
            "OpenAI", "OPENAI_API_KEY", "openai_api_key", "OpenAI API Key"
        )
        get_api_key_from_env_or_ui(
            "Gemini", "GEMINI_API_KEY", "gemini_api_key", "Gemini API Key"
        )
        get_api_key_from_env_or_ui(
            "xAI", "XAI_API_KEY", "xai_api_key", "xAI (Grok) API Key"
        )
        get_api_key_from_env_or_ui(
            "Anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key", "Anthropic API Key"
        )
    
    st.sidebar.markdown("---")
    
    # Model Settings
    st.sidebar.markdown("### ⚙️ 模型呼叫設定")
    
    provider = st.sidebar.selectbox(
        "模型供應商",
        ["openai", "gemini", "xai", "anthropic"],
        key="default_provider",
    )
    
    provider_models = {
        "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "xai": ["grok-4-fast-reasoning", "grok-3-mini"],
        "anthropic": ["claude-3-5-sonnet-latest", "claude-3-opus-latest"],
    }
    
    st.sidebar.selectbox(
        "模型版本",
        provider_models[provider],
        key="default_model",
    )
    
    st.sidebar.slider(
        "最大輸出 Token 數",
        64, 4096, 1024, 64,
        key="default_max_tokens",
    )
    
    st.sidebar.slider(
        "溫度（隨機性）",
        0.0, 1.0, 0.7, 0.05,
        key="default_temperature",
    )
    
    st.sidebar.markdown("---")
    
    # Case Log
    st.sidebar.markdown(f"### 📁 {get_translation('quest_log')}")
    st.sidebar.metric("已完成案件數", st.session_state.quests_completed)
    
    if st.sidebar.button("🔄 恢復資源"):
        update_player_stats("regenerate")
        add_combat_log("AI 資源與合規健康度已適度恢復", "success")
        st.rerun()

# -----------------------------------------------------------
# Input Tab
# -----------------------------------------------------------

def render_input_tab():
    """Render case input tab"""
    st.markdown(f"## 📝 {get_translation('input')}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_area(
            "📄 510(k) 案件模板 / 主要內容",
            key="template",
            height=260,
            help="例如：設備描述、適應症說明、實質等同性比較、風險管理摘要等"
        )
        
        st.text_area(
            "🔍 審查觀察與備註",
            key="observations",
            height=260,
            help="記錄審查歷程中的疑問、風險點、需追問之資料等"
        )
    
    with col2:
        render_activity_log()
        
        st.markdown("### ⚡ 快速動作")
        if st.button("💾 儲存當前輸入", use_container_width=True):
            add_combat_log("目前案件輸入已儲存（暫存於 session）", "success")
            st.success("已暫存目前內容。")
        
        if st.button("🧹 清空欄位", use_container_width=True):
            st.session_state.template = ""
            st.session_state.observations = ""
            add_combat_log("案件輸入欄位已清空", "info")
            st.rerun()

# -----------------------------------------------------------
# Pipeline Tab
# -----------------------------------------------------------

def render_pipeline_tab(config: Dict[str, Any]):
    """Render multi-agent 510(k) review pipeline tab"""
    st.markdown(f"## 🔄 {get_translation('pipeline')}")
    
    if not config or "pipelines" not in config:
        st.warning("⚠️ agents.yaml 中未找到任何審查流程 (pipelines) 設定。")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pipeline_options = {p["name"]: p for p in config["pipelines"]}
        selected_name = st.selectbox("🔎 選擇審查流程", list(pipeline_options.keys()))
        pipeline = pipeline_options[selected_name]
        
        st.markdown(f"**流程 ID：** `{pipeline['id']}`")
        st.markdown(f"**說明：** {pipeline.get('description', '')}")
        
        st.markdown("### 📂 流程步驟")
        for idx, step in enumerate(pipeline["steps"], start=1):
            st.markdown(f"- 第 {idx} 步：`{step['agent_id']}`")
        
        st.markdown("---")
        
        override_prompt = st.text_area(
            "📌 其他補充說明 / 特別指示",
            "例如：此案件風險偏高，請提高風險評估與法規比對的嚴謹度。",
            height=120,
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            provider = st.selectbox(
                "模型供應商覆寫（選填）",
                ["(使用預設)", "openai", "gemini", "xai", "anthropic"],
            )
        with col_b:
            model_override = st.text_input("模型名稱覆寫（選填）", "")
        
        if st.button(f"▶️ {get_translation('run')}", use_container_width=True):
            if st.session_state.mana < 20:
                st.error("❌ AI 資源不足，請先按左側『恢復資源』。")
                return
            
            template = st.session_state.get("template", "")
            observations = st.session_state.get("observations", "")
            current_input = (
                "【510(k) 案件輸入】\n"
                f"{template}\n\n"
                "【審查觀察與備註】\n"
                f"{observations}\n\n"
                "【額外指示】\n"
                f"{override_prompt}"
            )
            
            outputs = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, step in enumerate(pipeline["steps"]):
                agent_id = step["agent_id"]
                agent_cfg = next((a for a in config["agents"] if a["id"] == agent_id), None)
                
                if not agent_cfg:
                    st.error(f"❌ 找不到代理設定：{agent_id}")
                    return
                
                progress = (idx + 1) / len(pipeline["steps"])
                progress_bar.progress(progress)
                status_text.text(f"執行代理：{agent_cfg['name']} ...")
                
                try:
                    result = run_agent(
                        agent_cfg=agent_cfg,
                        user_prompt=current_input,
                        override_provider=None if provider.startswith("(") else provider,
                        override_model=model_override or None,
                        max_tokens=st.session_state.get("default_max_tokens", 1024),
                        temperature=st.session_state.get("default_temperature", 0.7),
                    )
                    outputs.append({"agent_id": agent_id, "output": result})
                    current_input = result
                    update_player_stats("regenerate")
                except Exception as e:
                    st.error(f"❌ 模型呼叫失敗：{e}")
                    add_combat_log(f"審查流程在代理 {agent_id} 中斷。", "error")
                    return
            
            progress_bar.progress(1.0)
            status_text.text("✅ 審查流程完成。")
            
            st.success("🎉 審查流程已成功完成並產出結果。")
            update_player_stats("quest_complete")
            add_combat_log(f"已完成審查流程：{selected_name}", "success")
            
            st.session_state.pipeline_history.append(outputs)
            
            st.markdown("### 📘 流程輸出結果")
            for idx, item in enumerate(outputs, start=1):
                with st.expander(f"步驟 {idx} – 代理 `{item['agent_id']}`", expanded=(idx == len(outputs))):
                    st.markdown(item["output"])
    
    with col2:
        render_activity_log()
        st.markdown("### 📊 流程統計")
        st.metric("已執行流程次數", len(st.session_state.pipeline_history))

# -----------------------------------------------------------
# Smart Replace Tab (placeholder, original feature kept)
# -----------------------------------------------------------

def render_smart_replace_tab():
    """Placeholder for smart editing (original feature kept)"""
    st.markdown(f"## ✨ {get_translation('smart_replace')}")
    st.info("此區可整合既有文字改寫與比對工具（保留原始設計空間）。")

# -----------------------------------------------------------
# AI Note Keeper: helpers
# -----------------------------------------------------------

def highlight_keywords_in_text(text: str, keywords: List[str], color: str) -> str:
    """Highlight given keywords in text using HTML span with specified color"""
    if not text or not keywords:
        return text
    result = text
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(
            lambda m: f"<span style='color:{color}'>{m.group(0)}</span>",
            result,
        )
    return result

# -----------------------------------------------------------
# AI Note Keeper Tab
# -----------------------------------------------------------

def render_notes_tab():
    """Render AI Note Keeper with multiple AI tools"""
    st.markdown(f"## 📔 {get_translation('notes')}")
    st.info(
        "將 510(k) 或醫療器材相關文字貼上，利用多代理 AI 進行 **Markdown 結構化、格式優化、關鍵字標示、實體抽取、心智圖與詞彙關聯圖**。"
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.text_area(
            "🧾 原始文本貼上區",
            key="note_raw_text",
            height=260,
            help="例如：510(k) 摘要、風險管理報告片段、技術說明、回覆 FDA 問答等",
        )
        if st.button("📄 轉換為 Markdown 結構", use_container_width=True):
            if not st.session_state.note_raw_text.strip():
                st.warning("請先貼上原始文本。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "你是一名專業的 FDA 醫療器材 510(k) 審查筆記整理助理，"
                        "請將使用者提供的原始文字轉換為 **結構清楚的 Markdown 文件**，"
                        "要求：\n"
                        "1. 嚴格保留所有原始資訊內容（不刪減、不改寫實質意義）。\n"
                        "2. 允許重新分段、加入標題階層 (##, ###) 與條列點，使內容更易讀。\n"
                        "3. 不要加入任何多餘說明，只輸出 Markdown 內容本身。"
                    )
                    user_prompt = st.session_state.note_raw_text
                    md = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=st.session_state.get("default_max_tokens", 1024),
                        temperature=0.1,
                    )
                    st.session_state.note_markdown = md
                    add_combat_log("完成原始文本的 Markdown 結構化。", "success")
                except Exception as e:
                    st.error(f"轉換為 Markdown 時發生錯誤：{e}")
    
    with col2:
        st.markdown("### 📑 Markdown 預覽")
        if st.session_state.note_markdown:
            st.markdown(st.session_state.note_markdown)
        else:
            st.caption("尚未產生 Markdown，請先於左側貼上文字並按下「轉換為 Markdown」。")
    
    st.markdown("---")
    
    tab_fmt, tab_kw, tab_ent, tab_mind, tab_word = st.tabs(
        ["AI 格式優化", "AI 關鍵字標示", "AI 實體抽取", "AI 心智圖", "AI 詞彙關聯圖"]
    )
    
    # --- AI Formatting ---
    with tab_fmt:
        st.markdown("### 🧹 AI 格式優化（保留原文，強化結構與重點）")
        st.caption(
            "說明：在**不刪除任何原文句子**的前提下，重新編排段落與標題，並用珊瑚色標註重要術語。"
        )
        if st.button("⚙️ 執行 AI 格式優化", use_container_width=True, key="btn_ai_format"):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "你是一名 FDA 510(k) 專業審查文件編輯助理。請對使用者提供的文字進行：\n"
                        "1. 嚴格保留所有原始句子內容，不刪除任何句子。\n"
                        "2. 允許重新排序段落、分群主題、加入適當 Markdown 標題 (##, ###)。\n"
                        "3. 針對重要法規、技術、風險與臨床相關關鍵詞，以 "
                        "<span style=\"color:coral\">...關鍵詞...</span> 的 HTML span 形式標示（僅改變呈現，不改變文字）。\n"
                        "4. 僅輸出 Markdown + HTML span 格式，不要額外解釋。"
                    )
                    user_prompt = base_text
                    formatted = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=st.session_state.get("default_max_tokens", 2048),
                        temperature=0.4,
                    )
                    st.session_state.note_formatted = formatted
                    add_combat_log("完成 AI 格式優化與重點標示。", "success")
                except Exception as e:
                    st.error(f"AI 格式優化失敗：{e}")
        
        if st.session_state.note_formatted:
            st.markdown("#### 格式優化結果")
            st.markdown(st.session_state.note_formatted, unsafe_allow_html=True)
    
    # --- AI Keywords ---
    with tab_kw:
        st.markdown("### 🎯 AI 關鍵字標示")
        st.caption("可自訂欲強調的關鍵詞與顏色，在 Markdown 內容中自動高亮。")
        
        kw_text = st.text_input(
            "輸入欲標示的關鍵字（以逗號分隔）",
            value="510(k), 實質等同性, 風險管理, 性能測試, FDA",
        )
        kw_color = st.color_picker("關鍵字顏色", value="#FF7F50")
        
        if st.button("🔍 標示關鍵字", use_container_width=True):
            base_text = (
                st.session_state.note_formatted
                or st.session_state.note_markdown
                or st.session_state.note_raw_text
            )
            if not base_text.strip():
                st.warning("尚無可處理的文本，請先產生 Markdown 或貼上文字。")
            else:
                keywords = [k for k in kw_text.split(",") if k.strip()]
                highlighted = highlight_keywords_in_text(base_text, keywords, kw_color)
                st.session_state.note_keywords_output = highlighted
                add_combat_log("完成自訂關鍵字標示。", "success")
        
        if st.session_state.note_keywords_output:
            st.markdown("#### 關鍵字標示結果")
            st.markdown(st.session_state.note_keywords_output, unsafe_allow_html=True)
    
    # --- AI Entities ---
    with tab_ent:
        st.markdown("### 🧬 AI 實體抽取（最多 20 個）")
        st.caption(
            "從文本中抽取最重要的法規、技術、臨床與風險相關實體，並產生結構化表格與 JSON。"
        )
        if st.button("📊 抽取 20 個關鍵實體", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "你是一名 FDA 510(k) 審查資訊抽取專家。"
                        "請從使用者提供的文字中，選出 **最多 20 個最關鍵的實體 (entity)**，"
                        "實體可以是：法規條文、標準、文件區段（如 Indications for Use）、"
                        "設備模組、風險類別、性能測試項目、臨床端點等。\n\n"
                        "請**只輸出 JSON**，格式為：\n"
                        "[\n"
                        "  {{\"id\": 1, \"name\": \"...\", \"type\": \"regulation|section|risk|test|clinical|other\", "
                        "\"description\": \"簡潔說明\", \"source_snippet\": \"原文中的代表性片段\"}},\n"
                        "  ... 共最多 20 筆\n"
                        "]\n"
                        "不要輸出任何額外文字。"
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    # 嘗試解析 JSON
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    entities = json.loads(raw_str)
                    if not isinstance(entities, list):
                        raise ValueError("回傳內容並非 JSON 陣列。")
                    st.session_state.note_entities_json_data = entities
                    add_combat_log("完成文本實體抽取（最多 20 個）。", "success")
                except Exception as e:
                    st.error(f"實體抽取與 JSON 解析失敗：{e}")
        
        if st.session_state.note_entities_json_data:
            st.markdown("#### 實體表格")
            # 建立 Markdown 表格
            table_md = "| id | name | type | description | source_snippet |\n"
            table_md += "|---|------|------|-------------|----------------|\n"
            for ent in st.session_state.note_entities_json_data:
                table_md += (
                    f"| {ent.get('id','')} "
                    f"| {ent.get('name','')} "
                    f"| {ent.get('type','')} "
                    f"| {ent.get('description','').replace('|','/')} "
                    f"| {ent.get('source_snippet','').replace('|','/')} |\n"
                )
            st.markdown(table_md)
            
            st.markdown("#### JSON 檢視")
            st.json(st.session_state.note_entities_json_data)
    
    # --- AI Mind-Map ---
    with tab_mind:
        st.markdown("### 🧠 AI 心智圖")
        st.caption(
            "根據文本內容自動產生節點與關係的 JSON，您可手動調整後，即時視覺化為心智圖。"
        )
        if st.button("🧠 產生心智圖 JSON", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "你是一名知識圖譜設計助理。請根據使用者提供的文字內容，"
                        "建立一份簡潔的 **心智圖結構 JSON**，格式如下：\n"
                        "{\n"
                        "  \"nodes\": [\n"
                        "    {\"id\": \"NodeID\", \"label\": \"顯示名稱\", \"type\": \"device|risk|test|regulation|clinical|other\"},\n"
                        "    ...\n"
                        "  ],\n"
                        "  \"edges\": [\n"
                        "    {\"source\": \"NodeID\", \"target\": \"NodeID\", \"relation\": \"文字描述此關係\"},\n"
                        "    ...\n"
                        "  ]\n"
                        "}\n"
                        "請將節點數控制在 8–15 個之間，邊數 10–25 個之間。只輸出 JSON，不要額外文字。"
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.3,
                    )
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    # 僅存文字，由使用者可再修改
                    st.session_state.note_mindmap_json_text = raw_str
                    add_combat_log("已產生心智圖 JSON 結構。", "success")
                except Exception as e:
                    st.error(f"心智圖 JSON 產生失敗：{e}")
        
        mindmap_text = st.text_area(
            "心智圖 JSON 可於此調整後重新繪製",
            value=st.session_state.note_mindmap_json_text,
            height=220,
        )
        if st.button("📈 根據 JSON 顯示心智圖", use_container_width=True):
            try:
                data = json.loads(mindmap_text)
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                dot = "digraph G {\nrankdir=LR;\n"
                # 節點
                for n in nodes:
                    nid = n.get("id", "")
                    label = n.get("label", nid)
                    dot += f"  \"{nid}\" [label=\"{label}\"];\n"
                # 邊
                for e in edges:
                    src = e.get("source", "")
                    tgt = e.get("target", "")
                    rel = e.get("relation", "")
                    dot += f"  \"{src}\" -> \"{tgt}\" [label=\"{rel}\"];\n"
                dot += "}"
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"解析或繪製心智圖時發生錯誤：{e}")
    
    # --- AI Wordgraph ---
    with tab_word:
        st.markdown("### 📚 AI 詞彙關聯圖 (Wordgraph)")
        st.caption(
            "根據文本自動分析重要術語之間的關聯，產生詞彙關聯圖 JSON 並視覺化。"
        )
        if st.button("📚 產生詞彙關聯 JSON", use_container_width=True):
            base_text = st.session_state.note_markdown or st.session_state.note_raw_text
            if not base_text.strip():
                st.warning("請先貼上文字並至少完成一次 Markdown 轉換。")
            else:
                try:
                    provider = st.session_state.get("default_provider", "openai")
                    model = st.session_state.get("default_model", "gpt-4o-mini")
                    system_prompt = (
                        "你是一名文字探勘與知識圖譜專家。請從使用者提供的文本中，"
                        "找出最重要的 10–15 個技術／法規／臨床術語，並建立詞彙關聯圖 JSON：\n"
                        "{\n"
                        "  \"nodes\": [\n"
                        "    {\"id\": \"TermID\", \"label\": \"顯示名稱\", \"frequency\": 數字},\n"
                        "    ...\n"
                        "  ],\n"
                        "  \"edges\": [\n"
                        "    {\"source\": \"TermID\", \"target\": \"TermID\", \"weight\": 共現強度 (1-5), \"note\": \"關聯說明\"},\n"
                        "    ...\n"
                        "  ]\n"
                        "}\n"
                        "只輸出 JSON，不要額外文字。"
                    )
                    user_prompt = base_text
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=1024,
                        temperature=0.4,
                    )
                    raw_str = raw.strip().strip("```json").strip("```").strip()
                    st.session_state.note_wordgraph_json_text = raw_str
                    add_combat_log("已產生詞彙關聯圖 JSON 結構。", "success")
                except Exception as e:
                    st.error(f"詞彙關聯 JSON 產生失敗：{e}")
        
        wordgraph_text = st.text_area(
            "詞彙關聯圖 JSON 可於此調整後重新繪製",
            value=st.session_state.note_wordgraph_json_text,
            height=220,
        )
        if st.button("📊 根據 JSON 顯示詞彙關聯圖", use_container_width=True):
            try:
                data = json.loads(wordgraph_text)
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])
                dot = "graph G {\n"
                # 節點（以頻率控制大小）
                for n in nodes:
                    nid = n.get("id", "")
                    label = n.get("label", nid)
                    freq = n.get("frequency", 1)
                    size = 10 + freq * 2
                    dot += f"  \"{nid}\" [label=\"{label}\", fontsize={size}];\n"
                # 無向邊
                for e in edges:
                    src = e.get("source", "")
                    tgt = e.get("target", "")
                    w = e.get("weight", 1)
                    note = e.get("note", "")
                    penwidth = 1 + w
                    dot += (
                        f"  \"{src}\" -- \"{tgt}\" "
                        f"[label=\"{note}\", penwidth={penwidth}];\n"
                    )
                dot += "}"
                st.graphviz_chart(dot)
            except Exception as e:
                st.error(f"解析或繪製詞彙關聯圖時發生錯誤：{e}")

# -----------------------------------------------------------
# Dashboard Tab
# -----------------------------------------------------------

def render_dashboard_tab():
    """Render interactive dashboard"""
    st.markdown(f"## 📊 {get_translation('dashboard')}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("審查成熟度等級", st.session_state.player_level)
    with col2:
        st.metric("已完成案件數", st.session_state.quests_completed)
    with col3:
        st.metric("LLM 呼叫次數", len(st.session_state.combat_log))
    with col4:
        st.metric("已執行流程數", len(st.session_state.pipeline_history))
    
    st.markdown("---")
    
    dash_tab1, dash_tab2, dash_tab3 = st.tabs(["案件歷程", "活動紀錄", "里程碑"])
    
    with dash_tab1:
        st.markdown("### 📁 案件 / 流程歷程")
        history = st.session_state.get("pipeline_history", [])
        if not history:
            st.info("尚未執行任何審查流程。")
        else:
            for run_idx, run in enumerate(reversed(history), start=1):
                with st.expander(f"案件流程 #{len(history) - run_idx + 1}"):
                    for step_idx, item in enumerate(run, start=1):
                        st.markdown(f"**步驟 {step_idx}** – 代理 `{item['agent_id']}`")
                        st.markdown(item["output"][:300] + "...")
    
    with dash_tab2:
        st.markdown("### 📑 完整活動紀錄")
        if st.session_state.combat_log:
            for entry in reversed(st.session_state.combat_log):
                st.markdown(f"{entry['icon']} {entry['message']}")
        else:
            st.info("尚無活動紀錄。")
    
    with dash_tab3:
        st.markdown("### 🏅 審查里程碑")
        
        achievements = []
        if st.session_state.player_level >= 5:
            achievements.append("🎖️ 進階審查官：審查成熟度等級達 5。")
        if st.session_state.quests_completed >= 10:
            achievements.append("📜 案件達人：完成 10 件以上案件流程。")
        if len(st.session_state.combat_log) >= 50:
            achievements.append("📈 高度互動：已執行超過 50 次模型呼叫或操作。")
        if st.session_state.player_level >= 10:
            achievements.append("👑 資深審查架構師：審查成熟度等級達 10。")
        
        if achievements:
            for ach in achievements:
                st.success(ach)
        else:
            st.info("持續累積案件與流程，可解鎖更多審查里程碑。")

# -----------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="FDA 510(k) Multi-Agent Review Studio",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    apply_custom_css()
    config = load_agents_config()
    render_enhanced_sidebar(config)
    
    st.markdown(f"# 🏥 {get_translation('title')}")
    st.markdown(f"_{get_translation('subtitle')}_")
    
    render_review_context_selector()
    
    st.markdown("---")
    
    tab_input, tab_pipeline, tab_smart, tab_notes, tab_dashboard = st.tabs([
        f"📝 {get_translation('input')}",
        f"🔄 {get_translation('pipeline')}",
        f"✨ {get_translation('smart_replace')}",
        f"📔 {get_translation('notes')}",
        f"📊 {get_translation('dashboard')}",
    ])
    
    with tab_input:
        render_input_tab()
    
    with tab_pipeline:
        render_pipeline_tab(config)
    
    with tab_smart:
        render_smart_replace_tab()
    
    with tab_notes:
        render_notes_tab()
    
    with tab_dashboard:
        render_dashboard_tab()

if __name__ == "__main__":
    main()
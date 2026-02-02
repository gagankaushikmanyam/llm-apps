"""
Streamlit launcher for the LLM Lab.

- Auto-discovers apps in ./applications/*.py (excluding __init__.py)
- Each app must expose:
    - APP_NAME: str
    - run() -> None
  Optionally:
    - APP_DESCRIPTION: str

This design allows you to add new apps (LoRA/QLoRA/RAG/MCP) by simply adding files
to the applications/ folder without changing this launcher.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Optional

import streamlit as st


@dataclass(frozen=True)
class DiscoveredApp:
    """Represents a discovered Streamlit sub-app."""

    key: str
    name: str
    description: str
    module: ModuleType


def _discover_apps(package_name: str = "applications") -> List[DiscoveredApp]:
    """
    Discover application modules under the given package.

    Rules:
    - Scan package modules via pkgutil
    - Ignore __init__
    - A module is considered valid if it has APP_NAME (str) and run() callable
    """
    apps: List[DiscoveredApp] = []

    try:
        package = importlib.import_module(package_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import package '{package_name}'. "
            f"Make sure the folder exists and contains __init__.py. Details: {exc}"
        ) from exc

    for modinfo in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module_name = modinfo.name
        if module_name.endswith(".__init__"):
            continue

        short_key = module_name.split(".")[-1]
        if short_key == "__init__":
            continue

        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            # Fail gracefully: include an "error app" entry rather than crashing the whole UI.
            err_name = f"{short_key} (import error)"
            apps.append(
                DiscoveredApp(
                    key=short_key,
                    name=err_name,
                    description=f"Could not import {module_name}: {exc}",
                    module=_make_error_module(module_name, exc),
                )
            )
            continue

        app_name = getattr(mod, "APP_NAME", None)
        run_fn = getattr(mod, "run", None)
        app_desc = getattr(mod, "APP_DESCRIPTION", "") or ""

        if isinstance(app_name, str) and callable(run_fn):
            apps.append(
                DiscoveredApp(
                    key=short_key,
                    name=app_name,
                    description=app_desc.strip(),
                    module=mod,
                )
            )

    # Sort by human-friendly app name
    apps.sort(key=lambda a: a.name.lower())
    return apps


def _make_error_module(module_name: str, exc: Exception) -> ModuleType:
    """
    Create a tiny module-like object with a run() function that displays an error.
    This avoids breaking the whole launcher if one app fails to import.
    """

    class _ErrorModule:
        APP_NAME = f"{module_name} (error)"
        APP_DESCRIPTION = "This app failed to load."

        @staticmethod
        def run() -> None:
            st.error("This app could not be imported.")
            st.code(f"Module: {module_name}\nError: {exc}")

    return _ErrorModule()  # type: ignore[return-value]


def _render_shell_header(title: str, subtitle: str = "") -> None:
    """Render a clean, professional header."""
    st.markdown(
        """
        <style>
          .llm-lab-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
          .llm-lab-subtitle { font-size: 1.0rem; color: #666; margin-top: 0; }
          .llm-lab-card {
            padding: 0.9rem 1.0rem;
            border-radius: 14px;
            border: 1px solid rgba(120,120,120,0.25);
            background: rgba(250,250,250,0.6);
            margin-bottom: 1.0rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='llm-lab-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<div class='llm-lab-subtitle'>{subtitle}</div>", unsafe_allow_html=True
        )
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


def _render_app_info(app: DiscoveredApp) -> None:
    """Render selected app name + description in a tidy card."""
    desc = app.description or "No description provided."
    st.markdown(
        f"""
        <div class="llm-lab-card">
          <div style="font-size: 1.2rem; font-weight: 650; margin-bottom: 0.35rem;">
            {app.name}
          </div>
          <div style="font-size: 0.98rem; color: #555;">
            {desc}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Main entrypoint for the Streamlit launcher."""
    st.set_page_config(
        page_title="LLM Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_shell_header(
        "🧪 LLM Lab",
        "A scalable Streamlit launcher for mini LLM apps (fine-tuning, LoRA, QLoRA, RAG, MCP…).",
    )

    try:
        apps = _discover_apps("applications")
    except Exception as exc:
        st.error("Failed to discover apps.")
        st.exception(exc)
        st.stop()

    if not apps:
        st.warning("No apps found. Add .py files to the applications/ folder.")
        st.stop()

    # Sidebar: app selector
    st.sidebar.title("Apps")
    app_names = [a.name for a in apps]
    name_to_app: Dict[str, DiscoveredApp] = {a.name: a for a in apps}

    default_name: str = app_names[0]
    selected_name: str = st.sidebar.radio(
        "Select an app",
        options=app_names,
        index=app_names.index(default_name),
        label_visibility="collapsed",
    )

    selected_app = name_to_app[selected_name]

    # Main area: app header + app UI
    _render_app_info(selected_app)

    try:
        selected_app.module.run()
    except Exception as exc:
        st.error("The selected app crashed while running.")
        st.exception(exc)


if __name__ == "__main__":
    main()

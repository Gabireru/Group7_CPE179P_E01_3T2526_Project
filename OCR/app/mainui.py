# mainui.py

import flet as ft

from loginui import get_login_layout
from mainpgui import get_main_layout
from uploadgui import get_upload_layout   # replaces camgui
from model_runner import ModelRunner


def main(page: ft.Page):
    page.title = "Formula Scanner"
    page.window.width = 1280
    page.window.height = 720
    page.padding = 0
    page.bgcolor = ft.Colors.BLACK

    # Load model in background as soon as the app starts so it's ready
    # by the time the user navigates to the upload page.
    runner = ModelRunner()

    def _load_model():
        runner.load()
        if runner.error:
            print(f"[ModelRunner] ERROR: {runner.error}")
        else:
            print("[ModelRunner] Model loaded and ready.")

    page.run_thread(_load_model)

    # ── Navigation ──────────────────────────────────────────────────────
    def navigate_to(route):
        page.controls.clear()

        if route == "/":
            page.add(get_login_layout(page, navigate_to))
        elif route == "/main":
            page.add(get_main_layout(page, navigate_to))
        elif route == "/upload":
            page.add(get_upload_layout(page, navigate_to, model_runner=runner))

        page.update()

    navigate_to("/")


ft.run(main, assets_dir=".")

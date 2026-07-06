import flet as ft

from loginui import get_login_layout
from mainpgui import get_main_layout
from camgui import get_camera_layout

def main(page: ft.Page):
    page.title = "Multi Linear Regression Detection"
    page.window.width = 1280
    page.window.height = 720
    page.padding = 0
    page.bgcolor = ft.Colors.BLACK 

    def navigate_to(route):
        page.controls.clear()
        if route == "/":
            page.add(get_login_layout(page, navigate_to))
        elif route == "/main":
            page.add(get_main_layout(page, navigate_to))
        elif route == "/camera":
            page.add(get_camera_layout(page, navigate_to))
            
        page.update()
    navigate_to("/")

ft.run(main, assets_dir=".")
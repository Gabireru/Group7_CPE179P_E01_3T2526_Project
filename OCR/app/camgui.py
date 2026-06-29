#camgui.py

import flet as ft

def get_camera_layout(page: ft.Page, navigate_to):
    back_btn = ft.Container(
        content=ft.Text("BACK", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD, size=14),
        bgcolor=ft.Colors.WHITE,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: navigate_to("/main") 
    )

    return ft.Stack(
        controls=[
            ft.Image(
                src="UI images/campg.jpg", 
                width=page.window.width,
                height=page.window.height,
                fit="contain" 
            ),
            ft.Container(content=back_btn, left=70, top=620)
        ],
        expand=True
    )
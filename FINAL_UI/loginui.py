import flet as ft

def get_login_layout(page: ft.Page, navigate_to):
    start_btn = ft.Container(
        content=ft.Text("START", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.WHITE,
        border_radius=50, 
        padding=ft.padding.Padding(left=35, right=35, top=15, bottom=15), 
        ink=True, 
        on_click=lambda _: navigate_to("/main") 
    )
    
    return ft.Stack(
        controls=[
            ft.Image(
                src="UI images/coverlog.png",
                width=page.window.width,
                height=page.window.height,
                fit="contain" 
            ),
            ft.Container(
                content=start_btn,
                left=160,  
                top=450    
            )
        ],
        expand=True
    )
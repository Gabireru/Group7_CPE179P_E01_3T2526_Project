import flet as ft

def get_main_layout(page: ft.Page, navigate_to):
    dark_color = "#050a1f" 

    scan_btn = ft.Container(
        content=ft.Text("SCAN", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        bgcolor=dark_color,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: navigate_to("/camera")
    )

    copy_btn = ft.Container(
        content=ft.Text("COPY", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        bgcolor=dark_color,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: print("copy equation")
    )

    # FIXED: A tiny async function to properly handle Flet's window closing coroutine
    async def close_app(e):
        await page.window.close()

    exit_btn = ft.Container(
        content=ft.Text("EXIT", color=dark_color, weight=ft.FontWeight.BOLD, size=13), 
        border=ft.border.Border(bottom=ft.border.BorderSide(2, dark_color)),               
        padding=ft.padding.Padding(left=0, top=0, right=0, bottom=2),
        ink=True,
        on_click=close_app # <--- Pass the async function here
    )

    return ft.Stack(
        controls=[
            ft.Image(
                src="UI images/mainpg.jpg",
                width=page.window.width,
                height=page.window.height,
                fit="contain" 
            ),
            ft.Container(content=scan_btn, left=160, top=560),
            ft.Container(content=copy_btn, left=495, top=560),
            ft.Container(content=exit_btn, left=840, top=570)
        ],
        expand=True
    )
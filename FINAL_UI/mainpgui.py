import flet as ft

def get_main_layout(page: ft.Page, navigate_to):
    dark_color = "#050a1f" 
    upload_btn = ft.Container(
        content=ft.Text("UPLOAD", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        bgcolor=dark_color,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: navigate_to("/camera")
    )
    async def copy_from_main(e):
        saved_equation = page.session.store.get("latest_equation") or r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2"
        try:
            await ft.Clipboard().set(saved_equation)
        except Exception:
            pass 
        page.snack_bar = ft.SnackBar(ft.Text("Equation copied to clipboard!"))
        page.snack_bar.open = True
        page.update()

    copy_btn = ft.Container(
        content=ft.Text("COPY", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=13),
        bgcolor=dark_color,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=copy_from_main
    )
    
    async def close_app(e):
        await page.window.close()

    exit_btn = ft.Container(
        content=ft.Text("EXIT", color=dark_color, weight=ft.FontWeight.BOLD, size=13), 
        border=ft.border.Border(bottom=ft.border.BorderSide(2, dark_color)),              
        padding=ft.padding.Padding(left=0, top=0, right=0, bottom=2),
        ink=True,
        on_click=close_app
    )
    
    return ft.Stack(
        controls=[
            ft.Image(
                src="UI images/mainpg.jpg",
                width=page.window.width,
                height=page.window.height,
                fit="contain" 
            ),
            ft.Container(content=upload_btn, left=160, top=560),
            ft.Container(content=copy_btn, left=495, top=560),
            ft.Container(content=exit_btn, left=840, top=570)
        ],
        expand=True
    )
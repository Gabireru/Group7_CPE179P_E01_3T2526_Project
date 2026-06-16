import flet as ft

def main(page: ft.Page):
    page.title = "Multi Linear Regression Detection"
    page.window_width = 1280
    page.window_height = 720
    page.padding = 0
    page.bgcolor = ft.Colors.BLACK 

    login_btn = ft.Container(
        content=ft.Text("LOGIN", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.WHITE,
        border_radius=50, 
        padding=ft.padding.Padding(left=35, right=35, top=15, bottom=15), 
        ink=True, 
        on_click=lambda _: print("Login clicked")
    )
    signup_btn = ft.Container(
        content=ft.Text("SIGNUP", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=16), 
        border=ft.border.Border(bottom=ft.border.BorderSide(2, ft.Colors.WHITE)),               
        padding=ft.padding.Padding(left=0, top=0, right=0, bottom=2),
        on_click=lambda _: print("Signup clicked")
    )
    button_row = ft.Row(
        controls=[login_btn, signup_btn],
        spacing=40,
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER
    )
    main_layout = ft.Stack(
        controls=[
            ft.Image(
                src="UI images/coverlog.png",
                width=page.window_width,
                height=page.window_height,
                fit="contain" 
            ),
            ft.Container(
                content=button_row,
                left=160,  
                top=450    
            )
        ],
        expand=True
    )
    page.add(main_layout)
ft.run(main, assets_dir=".")
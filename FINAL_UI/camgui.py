import flet as ft
import shutil
import os

def get_camera_layout(page: ft.Page, navigate_to):
    uploaded_image = ft.Image(
        src="", 
        width=600,
        height=350,
        fit="contain",
        visible=False,
        border_radius=10
    )
    equation_field = ft.TextField(
        value="Upload an image to detect equation...",
        text_align=ft.TextAlign.CENTER,
        width=500,
        color=ft.Colors.WHITE,
        bgcolor="#050a1f",
        border_color=ft.Colors.BLUE,
        visible=False 
    )
    async def handle_upload(e):
        files = await ft.FilePicker().pick_files(
            allow_multiple=False, 
            allowed_extensions=["png", "jpg", "jpeg"]
        )
        if files and len(files) > 0:
            original_path = files[0].path
            save_folder = "MLR_Scans"
            os.makedirs(save_folder, exist_ok=True)
            filename = os.path.basename(original_path)
            new_saved_path = os.path.join(save_folder, filename)
            shutil.copy(original_path, new_saved_path)
            uploaded_image.src = new_saved_path
            uploaded_image.visible = True
            equation_field.value = r"\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2"
            equation_field.visible = True
            page.session.store.set("latest_equation", equation_field.value)
            page.update()
    async def copy_to_clipboard(e):
        try:
            await ft.Clipboard().set(equation_field.value)
        except Exception:
            pass
        page.snack_bar = ft.SnackBar(ft.Text("Equation copied to clipboard!"))
        page.snack_bar.open = True
        page.update()

    upload_btn = ft.Container(
        content=ft.Text("UPLOAD", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=14),
        bgcolor="#050a1f", 
        border_radius=50,
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True,
        on_click=handle_upload, 
        left=70,
        top=550 
    )
    copy_btn = ft.Container(
        content=ft.Text("COPY EQUATION", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD, size=14),
        bgcolor="#050a1f", 
        border_radius=50,
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True,
        on_click=copy_to_clipboard,
        left=200, 
        top=550 
    )
    back_btn = ft.Container(
        content=ft.Text("BACK", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD, size=14),
        bgcolor=ft.Colors.WHITE,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: navigate_to("/main"),
        left=70,
        top=620  
    )
    center_cluster = ft.Column(
        controls=[uploaded_image, equation_field],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20
    )
    return ft.Stack(
        controls=[
            ft.Image(
                src="UI images/campg.jpg", 
                width=page.window.width,
                height=page.window.height,
                fit="contain" 
            ),
            ft.Container(
                content=center_cluster,
                alignment=ft.Alignment(0, 0),
                width=page.window.width,
                height=page.window.height,
            ),
            upload_btn,
            copy_btn,
            back_btn
        ],
        expand=True
    )
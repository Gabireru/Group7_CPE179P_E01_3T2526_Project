#mainui.py

import flet as ft

# Import the layouts from your other files
from loginui import get_login_layout
from mainpgui import get_main_layout
from camgui import get_camera_layout

def main(page: ft.Page):
    page.title = "Multi Linear Regression Detection"
    # Modern Flet window sizing
    page.window.width = 1280
    page.window.height = 720
    page.padding = 0
    page.bgcolor = ft.Colors.BLACK 

    # --- The Screen Swapper ---
    def navigate_to(route):
        page.controls.clear() # Wipe the screen
        
        # Snap the correct UI onto the screen
        if route == "/":
            page.add(get_login_layout(page, navigate_to))
        elif route == "/main":
            page.add(get_main_layout(page, navigate_to))
        elif route == "/camera":
            page.add(get_camera_layout(page, navigate_to))
            
        page.update() # Tell Flet to redraw

    # Start the app on the login screen
    navigate_to("/")

# Run the app
ft.run(main, assets_dir=".")
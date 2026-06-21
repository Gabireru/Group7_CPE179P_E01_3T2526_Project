import flet as ft
import cv2
import base64
import threading
import time

def main(page: ft.Page):
    page.title = "Camera View Demo"
    page.window_width = 1280
    page.window_height = 720
    page.padding = 0
    page.bgcolor = ft.Colors.BLACK

    # 1. The Live Camera Screen Widget
    # FIXED: We use the required 'src' argument, but feed it the transparent pixel as a Data URI
    camera_feed = ft.Image(
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=", 
        width=800, 
        height=600,
        fit="contain"
    )

    # 2. The BACK Button 
    back_btn = ft.Container(
        content=ft.Text("BACK", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD, size=14),
        bgcolor=ft.Colors.WHITE,
        border_radius=50, 
        padding=ft.padding.Padding(left=25, right=25, top=10, bottom=10), 
        ink=True, 
        on_click=lambda _: page.window.close() 
    )

    # 3. Layout Stack
    main_layout = ft.Stack(
        controls=[
            ft.Image(
                src="campg.jpg", 
                width=page.window_width,
                height=page.window_height,
                fit="contain" 
            ),
            ft.Container(
                content=camera_feed,
                left=240, 
                top=80    
            ),
            ft.Container(
                content=back_btn,
                left=70,  
                top=620    
            )
        ],
        expand=True
    )

    page.add(main_layout)

    # --- THE BACKGROUND THREAD ---
    # --- THE BACKGROUND THREAD ---
    def capture_camera():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

        while True:
            ret, frame = cap.read()
            if ret:
                # 1. Save the actual image file directly into your assets folder
                cv2.imwrite("assets/live_frame.jpg", frame)
                
                # 2. The Cache Buster: Add a timestamp so Flet thinks it's a brand new file every single frame
                camera_feed.src = f"live_frame.jpg?time={time.time()}"
                
                try:
                    camera_feed.update()
                except Exception as e:
                    break
                    
            time.sleep(0.03) 
            
        cap.release()

    # Start the camera loop in a "Daemon" thread (it dies automatically when you close the app)
    cam_thread = threading.Thread(target=capture_camera, daemon=True)
    cam_thread.start()

# Run the app
ft.run(main, assets_dir="assets")
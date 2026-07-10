# uploadgui.py
import io
import base64
import threading

import flet as ft
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# LaTeX rendering helper
# ---------------------------------------------------------------------------

def render_latex_to_base64(latex_str: str, fontsize: int = 20) -> str | None:
    """
    Render a LaTeX string to a PNG image and return it as a base64 string
    for use in ft.Image(src_base64=...).

    Uses matplotlib's built-in math text renderer — no full LaTeX install
    needed, works on Pi with just matplotlib installed.

    Returns None if rendering fails (e.g. syntax error in the LaTeX).
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        print(repr(latex_str))
        display_str = f"${latex_str}$"

        fig = plt.figure(figsize=(8, 1.5))
        fig.patch.set_facecolor("#0a1030")

        fig.text(
            0.5, 0.5,
            display_str,
            fontsize=fontsize,
            ha="center",
            va="center",
            color="white",
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=120,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print("========== LATEX ==========")
        print(repr(display_str))
        print("===========================")
        print(e)


# ---------------------------------------------------------------------------
# Upload layout
# ---------------------------------------------------------------------------

def get_upload_layout(page: ft.Page, navigate_to, model_runner=None):
    dark_color = "#050a1f"
    bg_color = "#0a1030"

    # ── State ──────────────────────────────────────────────────────────────
    _state = {
        "pil_image": None,      # PIL image from the picked file
        "scanning": False,
    }

    # ── Controls ───────────────────────────────────────────────────────────

    preview_image = ft.Image(
        src="UI images/campg.jpg",      # placeholder until a file is picked
        width=480,
        height=300,
        fit=ft.BoxFit.CONTAIN,          # ft.ImageFit removed in flet 0.85+
        border_radius=12,
    )

    pick_label = ft.Text(
        "No file selected.",
        color=ft.Colors.WHITE60,
        size=11,
        italic=True,
    )

    status_text = ft.Text(
        "Pick an image to begin.",
        color=ft.Colors.WHITE60,
        size=11,
        italic=True,
    )

    result_label = ft.Text(
        "",
        color=ft.Colors.WHITE70,
        size=11,
        text_align=ft.TextAlign.CENTER,
    )

    result_text = ft.Text(
        "",
        color=ft.Colors.WHITE,
        size=13,
        selectable=True,
        text_align=ft.TextAlign.CENTER,
        max_lines=4,
        overflow=ft.TextOverflow.ELLIPSIS,
    )

    rendered_image = ft.Image(
        src="",
        visible=False,
        width=640,
        height=120,
        fit=ft.BoxFit.CONTAIN,
    )

    render_error = ft.Text(
        "",
        color=ft.Colors.ORANGE_300,
        size=11,
        italic=True,
        visible=False,
    )

    scan_btn = ft.ElevatedButton(
        content=ft.Text("SCAN", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
    )

    copy_btn = ft.ElevatedButton(
        content=ft.Text("COPY LaTeX", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        disabled=True,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
    )

    back_btn = ft.ElevatedButton(
        content=ft.Text("BACK", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
        on_click=lambda _: navigate_to("/main"),
    )

    # ── File picker ────────────────────────────────────────────────────────
    # In Flet 0.85, FilePicker.pick_files() is async and returns files
    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    async def on_pick_clicked(e):
        files = await file_picker.pick_files(
            dialog_title="Select an equation image",
            allowed_extensions=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            allow_multiple=False,
        )

        if not files:
            return

        picked = files[0]
        pick_label.value = picked.name

        try:
            pil = Image.open(picked.path).convert("RGB")
            _state["pil_image"] = pil

            # Show thumbnail preview
            thumb = pil.copy()
            thumb.thumbnail((480, 300))
            buf = io.BytesIO()
            thumb.save(buf, format="PNG")
            preview_image.src = picked.path

            status_text.value = "Image loaded. Press SCAN."
            scan_btn.disabled = False

            # Clear previous results
            result_label.value = ""
            result_text.value = ""
            rendered_image.visible = False
            render_error.visible = False
            copy_btn.disabled = True

        except Exception as exc:
            status_text.value = f"Could not open image: {exc}"
            scan_btn.disabled = True

        page.update()

    pick_btn = ft.ElevatedButton(
        content=ft.Text("PICK IMAGE", color=ft.Colors.WHITE, weight=ft.FontWeight.BOLD),
        bgcolor=dark_color,
        style=ft.ButtonStyle(shape=ft.StadiumBorder()),
        on_click=on_pick_clicked,
    )

    # ── Inference ──────────────────────────────────────────────────────────

    def _do_inference():
        """Runs in a worker thread — never call page.update() from here directly,
        just mutate controls then call page.update() at the end."""
        try:
            if model_runner is None:
                raise RuntimeError("No model runner provided.")
            if not model_runner.ready:
                raise RuntimeError(
                    model_runner.error or "Model is still loading — please wait a moment."
                )
            if _state["pil_image"] is None:
                raise RuntimeError("No image loaded.")

            latex = model_runner.predict(_state["pil_image"])

            # Store for the COPY button on the main page too
            page.session.store.set("last_latex", latex)

            result_label.value = "Predicted LaTeX (raw):"
            result_text.value = latex
            copy_btn.disabled = False
            status_text.value = "Done."

        except Exception as exc:
            result_label.value = "Error:"
            result_text.value = str(exc)
            rendered_image.visible = False
            status_text.value = ""
            copy_btn.disabled = True

        finally:
            _state["scanning"] = False
            scan_btn.disabled = False
            scan_btn.content.value = "SCAN"
            page.update()

    def on_scan(e):
        if _state["scanning"] or _state["pil_image"] is None:
            return
        _state["scanning"] = True
        scan_btn.disabled = True
        scan_btn.content.value = "SCANNING…"
        result_label.value = ""
        result_text.value = ""
        rendered_image.visible = False
        render_error.visible = False
        status_text.value = "Running model…"
        copy_btn.disabled = True
        page.update()
        page.run_thread(_do_inference)

    scan_btn.on_click = on_scan

    # ── Copy ───────────────────────────────────────────────────────────────

    # Clipboard registered as a service so it has an active session
    clipboard = ft.Clipboard()
    page.services.append(clipboard)

    def _show_snack(message: str, color: str):
        """
        Display a SnackBar and ensure it is removed from the dialog stack
        after dismissal, preventing the black rectangle artifact that appears
        when dialogs linger in the stack after their animation ends.
        """
        snack = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.WHITE),
            bgcolor=color,
            duration=2000,
            persist=False,
            on_dismiss=lambda _: page.pop_dialog(),
        )
        page.show_dialog(snack)

    async def on_copy(e):
        latex = page.session.store.get("last_latex")
        if latex:
            await clipboard.set(latex)
            _show_snack("LaTeX copied to clipboard!", "#1a2a6c")
        else:
            _show_snack("No result yet — run SCAN first.", "#6c1a1a")

    copy_btn.on_click = on_copy

    # ── Layout ─────────────────────────────────────────────────────────────

    return ft.Column(
        controls=[
            # ── Top bar ──
            ft.Container(
                content=ft.Row(
                    controls=[
                        back_btn,
                        ft.Text(
                            "Formula Scanner — Upload",
                            color=ft.Colors.WHITE,
                            size=16,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=16,
                bgcolor=dark_color,
            ),

            # ── Image preview ──
            ft.Container(
                content=ft.Column(
                    controls=[
                        preview_image,
                        pick_label,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=6,
                ),
                alignment=ft.Alignment.CENTER,
                bgcolor=bg_color,
                padding=16,
            ),

            # ── Result area ──
            ft.Container(
                content=ft.Column(
                    controls=[
                        status_text,
                        result_label,
                        result_text,
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
                padding=ft.Padding(left=20, top=8, right=20, bottom=8),
                bgcolor=bg_color,
            ),

            # ── Action bar ──
            ft.Container(
                content=ft.Row(
                    controls=[pick_btn, scan_btn, copy_btn],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=12,
                ),
                padding=16,
                bgcolor=dark_color,
            ),
        ],
        spacing=0,
        expand=True,
    )